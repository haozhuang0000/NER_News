from bson.binary import Binary
from dataclasses import dataclass
from ingestion.db import create_id
from ingestion.config import DEFAULT_NEWS_MONGO_COLLECTION


@dataclass
class Article:
    _id: Binary
    article_id: str
    Title: str
    Author: list[str]
    Description: str
    Date: str
    Content: str
    Category: str
    Language: str
    Keywords: list[str]
    Country: list[str]
    Search_q: list[str]
    Url: dict[str, str]
    Source: dict[str, str]


def create_article_record(data: dict[str, any], kw: str) -> dict[str, any]:

    return {
        "_id": create_id(
            DEFAULT_NEWS_MONGO_COLLECTION,
            data.get("title", "") + data.get("pubDate", "") + data.get("link", ""),
        ),
        "article_id": data.get("article_id", ""),
        "Title": data.get("title", ""),
        "Author": data.get("creator", []),
        "Description": data.get("description", ""),
        "Date": data.get("pubDate", ""),
        "Content": data.get("content", ""),
        "Category": data.get("keywords", []),
        "Language": data.get("language", ""),
        "Keywords": data.get("keywords", []),
        "Country": data.get("keywords", []),
        "Search_q": kw.split(),
        "pubDateTZ": data.get("pubDateTZ", ""),
        "Url": {
            "article_url": data.get("link", ""),
            "image_url": data.get("image_url", ""),
            "video_url": data.get("video_url", ""),
        },
        "Source": {
            "source_id": data.get("source_id", ""),
            "source_priority": data.get("source_priority", ""),
            "source_name": data.get("source_name", ""),
            "source_url": data.get("source_url", ""),
            "source_icon": data.get("source_icon", ""),
        },
    }
