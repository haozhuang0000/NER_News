from bson.binary import Binary
from dataclasses import dataclass


@dataclass
class SentenceInput:
    _id: Binary
    article_id: str
    News_Id: Binary
    Title: str
    Date: str
    Companies_econs: str
    Sentence_id: str
    Sentence_1: str
    Sentence_2: list[str]
