from bson.binary import Binary
from dataclasses import dataclass


@dataclass
class NEROut:
    _id: Binary
    article_id: str
    News_Id: Binary
    Title: str
    Date: str
    Content: str
    Category: str
    Search_q: list[str]
    Sentence_list: list[str]
    Companies_list: list[str]
    Econs_list: list[str]
