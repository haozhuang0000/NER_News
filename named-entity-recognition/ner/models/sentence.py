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


@dataclass
class SelectedSentence(SentenceSplit):
    Entity_id: list
    Bingo_entity: list
    Similarity: list
    Similarity_1st: list
    Similarity_2nd: list
    first_cleaned_ner_entity: list
    first_matched_cleaned_comp: list
    second_ner_entity: list
    second_matched_comp: list
    NER_Mapping_ID: list
    Updated: int = 0
