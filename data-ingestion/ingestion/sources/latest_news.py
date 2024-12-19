from newsdataapi import NewsDataApiClient
from concurrent.futures import ThreadPoolExecutor
import logging

from ingestion.config import (
    NEWS_API_KEY,
    NEWS_API_TIMEFRAME,
    NEWS_API_QUERY_SIZE,
    NEWS_API_COUNTRIES,
    NEWS_API_KEYWORDS,
    DEFAULT_NEWS_MONGO_COLLECTION,
)
from ingestion.utils import translate_word
from ingestion.log import log_function
from ingestion.db import database, batch_upsert
from ingestion.models import create_article_record, Article
from ingestion.kafka import submit_data

news_data = NewsDataApiClient(NEWS_API_KEY)
news_collection = database[DEFAULT_NEWS_MONGO_COLLECTION]
logger = logging.getLogger(__name__)


def retrieve_latest_news(kw: str, ln: str):
    next_token = None
    articles: list[Article] = []
    start = news_collection.count_documents({})

    try:
        while True:
            res = news_data.latest_api(
                q=kw,
                language=ln,
                size=NEWS_API_QUERY_SIZE,
                removeduplicate=True,
                timeframe=NEWS_API_TIMEFRAME,
                page=next_token,
            )
            next_token = res.get("nextPage", None)
            if not next_token:
                break
            if res.get("status", "failure") != "success":
                raise Exception(
                    f"Failed to retrieve for  with keyword {kw} and lang {ln}"
                    f"Triggered at page {next_token}"
                )
            articles.extend(
                [Article(**create_article_record(a, kw)) for a in res["results"]]
            )
        submit_data(articles)
        batch_upsert(news_collection, articles)
        end = news_collection.count_documents({})
        logger.info(f"{end - start} inserted successfully")
    except ValueError as ve:
        logger.error(f"JSON decoding error {ve} with keyword {kw} and lang {ln}")
    except Exception as e:
        logger.error(f"An error occured: {e} with keyword {kw} and lang {ln}")


@log_function
def update_latest_news_database():
    countries = NEWS_API_COUNTRIES
    keywords = NEWS_API_KEYWORDS

    with ThreadPoolExecutor(max_workers=4) as exec:
        for kw in keywords:
            for ln in countries.values():
                exec.submit(retrieve_latest_news, translate_word(kw, ln), ln)
