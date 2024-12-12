from bson.binary import Binary
from dataclasses import dataclass
from ingestion.db import create_id
from ingestion.config import DEFAULT_RAW_DATA_COLLECTION


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
            DEFAULT_RAW_DATA_COLLECTION,
            data.get("title", "") + data.get("pubDate", "") + data.get("link", ""),
        ),
        "article_id": data.get("article_id") or "",
        "Title": data.get("title") or "",
        "Author": data.get("creator") or [],
        "Description": data.get("description") or "",
        "Date": data.get("pubDate") or "",
        "Content": data.get("content") or "",
        "Category": data.get("category") or [],
        "Language": data.get("language") or "",
        "Keywords": data.get("keywords") or [],
        "Country": data.get("country") or [],
        "Search_q": kw.split() or [],
        "Url": {
            "article_url": data.get("link") or "",
            "image_url": data.get("image_url") or "",
            "video_url": data.get("video_url") or "",
        },
        "Source": {
            "source_id": data.get("source_id") or "",
            "source_priority": data.get("source_priority") or "",
            "source_name": data.get("source_name") or "",
            "source_url": data.get("source_url") or "",
            "source_icon": data.get("source_icon") or "",
        },
    }
