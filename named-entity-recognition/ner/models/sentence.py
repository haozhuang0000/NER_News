from bson.binary import Binary
from dataclasses import dataclass


@dataclass
class SentenceSplit:
    _id: Binary
    article_id: str
    News_Id: Binary
    Title: str
    Date: str
    Companies_econs: list[str]
    Sentence_id: list[str]
    Sentence_1: list[str]
    Sentence_2: list[str]
