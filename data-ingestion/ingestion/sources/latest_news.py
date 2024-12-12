from newsdataapi import NewsDataApiClient
from concurrent.futures import Future, ThreadPoolExecutor

from ingestion.config import (
    NEWS_API_KEY,
    NEWS_API_TIMEFRAME,
    NEWS_API_QUERY_SIZE,
    NEWS_API_COUNTRIES,
    NEWS_API_KEYWORDS,
    DEFAULT_RAW_DATA_COLLECTION,
)
from ingestion.utils import translate_word
from ingestion.db import database
from ingestion.models import create_article_record, Article

news_data = NewsDataApiClient(NEWS_API_KEY)
news_collection = database[DEFAULT_RAW_DATA_COLLECTION]


def retrieve_latest_news(kw: str, ln: str):
    next_token = None
    articles: list[Article] = []

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
        return articles
    except ValueError as ve:
        raise Exception(
            f"RETRIEVAL ERROR: JSON decoding error {ve} with keyword {kw} and lang {ln}"
        )
    except Exception as e:
        raise Exception(f"RETRIEVAL ERROR: {e} with keyword {kw} and lang {ln}")


def update_latest_news_database():
    countries = NEWS_API_COUNTRIES
    keywords = NEWS_API_KEYWORDS

    futures: list[tuple[str, Future[list[Article] | None]]] = []
    with ThreadPoolExecutor(max_workers=4) as exec:
        for kw in keywords:
            for ln in countries.values():
                future = exec.submit(retrieve_latest_news, translate_word(kw, ln), ln)
                futures.append((ln, future))

    articles: dict[str, list[Article]] = {}

    for ln, f in futures:
        if f.result() is not None:
            articles[ln] = f.result()

    return articles
