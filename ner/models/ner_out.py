from bson.binary import Binary
from dataclasses import dataclass


@dataclass
class NEROut:
    _id: Binary
    article_id: str
    Title: str
    Description: str
    Date: str
    Content: str
    Category: str
    Language: str
    Country: list[str]
    Search_q: list[str]
    News_Id: Binary
    Category: list[str]
    Sentence_list: list[str]
    Companies_list: list[str]
    Econs_list: list[str]
