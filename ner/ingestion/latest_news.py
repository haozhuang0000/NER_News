from newsdataapi import NewsDataApiClient
from concurrent.futures import ThreadPoolExecutor
from pymongo import UpdateOne

from ner.config import NEWS_API_KEY
from ner.utils import translate_word
from ner.db import database
from ner.models import create_dict


news_data = NewsDataApiClient(NEWS_API_KEY)
news_collection = database["News"]


def retrieve_latest_news(kw: str, ln: str):
    next_token = None
    upsert_ops = []
    start = news_collection.count_documents({})

    try:
        while True:
            res = news_data.latest_api(
                q=kw,
                language=ln,
                size=50,
                removeduplicate=True,
                timeframe=12,
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
            upsert_ops.extend(
                [
                    UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
                    for doc in (create_dict(a, kw) for a in res["results"])
                ]
            )

        news_collection.bulk_write(upsert_ops)
        end = news_collection.count_documents({})
        print(f"{end - start} inserted successfully")
    except ValueError as ve:
        print(f"JSON decoding error {ve} with keyword {kw} and lang {ln}")
    except Exception as e:
        print(f"An error occured: {e} with keyword {kw} and lang {ln}")


def update_latest_news_database():
    countries = {
        "cn": "zh",
        "id": "id",
        "jp": "jp",
        "kr": "ko",
        "my": "ms",
        "th": "th",
        "us": "en",
    }
    keywords = ["finance", "debt"]

    with ThreadPoolExecutor(max_workers=4) as exec:
        for kw in keywords:
            for ln in countries.values():
                exec.submit(retrieve_latest_news, translate_word(kw, ln), ln)
