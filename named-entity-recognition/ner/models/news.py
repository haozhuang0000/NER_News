from bson.binary import Binary
from dataclasses import dataclass


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
