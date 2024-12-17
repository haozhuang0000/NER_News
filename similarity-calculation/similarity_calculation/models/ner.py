from bson.binary import Binary
from dataclasses import dataclass


@dataclass
class NERMapped:
    _id: Binary
    U3_Company_Number: int
    Company_Name: Binary
    NER_Name: str
    NER_Original_Name: str
    Similarity: int
    Similarity_1st: list
    Similarity_2nd: list
    first_cleaned_ner_entity: list
    first_matched_cleaned_comp: list
    second_ner_entity: list
    second_matched_comp: list
    Pre_Defined: int
    Updated: int
    update_date: int
