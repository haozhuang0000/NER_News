from bson.binary import Binary
from dataclasses import dataclass
from ner.db import create_id


@dataclass
class Article:
    _id: Binary
    article_id: str
    News_Id: Binary
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


def create_dict(response: dict[str, any], kw: str) -> dict[str, any]:

    return {
        "_id": create_id(
            response.get("title", ""),
            response.get("pubDate", ""),
            response.get("link", ""),
        ),
        "article_id": response.get("article_id", ""),
        "Title": response.get("title", ""),
        "Author": response.get("creator", []),
        "Description": response.get("description", ""),
        "Date": response.get("pubDate", ""),
        "Content": response.get("content", ""),
        "Category": response.get("keywords", []),
        "Language": response.get("language", ""),
        "Keywords": response.get("keywords", []),
        "Country": response.get("keywords", []),
        "Search_q": kw.split(),
        "pubDateTZ": response.get("pubDateTZ", ""),
        "Url": {
            "article_url": response.get("link", ""),
            "image_url": response.get("image_url", ""),
            "video_url": response.get("video_url", ""),
        },
        "Source": {
            "source_id": response.get("source_id", ""),
            "source_priority": response.get("source_priority", ""),
            "source_name": response.get("source_name", ""),
            "source_url": response.get("source_url", ""),
            "source_icon": response.get("source_icon", ""),
        },
    }
