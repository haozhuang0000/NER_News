import time
import os
import numpy as np
import functools
from typing import TypeVar, Generator
import string
import pandas as pd
import bson
from bson.binary import UuidRepresentation
import uuid

script_dir = os.path.dirname(os.path.abspath(__file__))
suffixs = pd.read_csv(
    os.path.abspath(os.path.join(script_dir, "../../_static/mapping_file/suffix.csv"))
)

T = TypeVar("T")


# ---------------------- Function for similarity calculation ---------------------- ##
def mysort(x, y):
    if x[0] > y[0]:
        return 1
    else:
        return -1


def get_entity_names(
    company_list_name: str,
    econ_list_name: str,
    logger,
    include_companies: bool = True,
    include_econs: bool = True,
) -> list[str]:
    """
    Retrieves entity names based on the specified parameters.

    Parameters:
    - include_companies (bool): Whether to include company names.
    - include_econs (bool): Whether to include economic entity names.
    - parent_dir (str): The directory where the entity lists are stored.
    - company_list_name (str): The filename of the company list.
    - econ_list_name (str): The filename of the economic entities list.
    - logger: Logger for logging information and errors.

    Returns:
    - list[str]: A list of entity names based on the input parameters.
    """
    # input company names
    if include_companies:
        start_time = time.time()
        logger.info("loading company list...")

        comp_df = pd.read_csv(
            os.path.abspath(
                os.path.join(
                    script_dir, f"../../_static/mapping_file/{company_list_name}"
                )
            ),
            index_col=0,
        )
        comp_df["Company_name"] = comp_df["Company_name"].astype(str)
        comp_list = comp_df["Company_name"].tolist()
        u3id_list = comp_df["CompanyID"].tolist()

        for idx, u3id in enumerate(u3id_list):
            if u3id < 0:
                del u3id_list[idx]
                del comp_list[idx]

        # ------- Sorted each different type of list for consistency ------- #
        # Full company list
        combined_full_comp_list = []
        for idx in range(len(comp_list)):
            combined_full_comp_list.append((comp_list[idx], u3id_list[idx]))

        comp_list.sort()
        combined_full_comp_list.sort(key=functools.cmp_to_key(mysort))
        logger.info("load company list time: {}".format(time.time() - start_time))
    else:
        combined_full_comp_list = []
        # Input econs names
    if include_econs:
        start_time = time.time()
        logger.info("loading econs list...")
        econ_infor = pd.read_csv(
            os.path.abspath(
                os.path.join(script_dir, f"../../_static/mapping_file/{econ_list_name}")
            )
        )
        econ_infor = econ_infor[[econ_infor.columns[0], econ_infor.columns[1]]]
        econ_infor.columns = ["econ_name", "econ_id"]
        econ_list = []
        econ_id = []
        for i in range(econ_infor.shape[0]):
            if not econ_infor.isna()["econ_name"][i]:
                econ_list.append(econ_infor["econ_name"][i])
                econ_id.append(econ_infor["econ_id"][i])
        logger.info("load econ list time: {}".format(time.time() - start_time))
    else:
        econ_list = []
        econ_id = []

    return combined_full_comp_list, econ_list, econ_id


def first_word(comp):
    return comp.split()[0]


def rm_none_entity(entity_list):
    new_entity_list = []
    for entity in entity_list:
        if entity != "":
            new_entity_list.append(entity)
    return new_entity_list


def is_shortName(short, long):
    long = long.lower()
    short = short.lower()
    if short in long:
        if len(long) <= len(short):
            return False  # same name
        elif len(long.replace(short, "").strip()) < len(long.replace(short, "")):
            return True
    return False


def is_abbrev(short, long):
    FirstCharacters = ""
    for i in long.split(" "):
        if i != "":
            FirstCharacters = FirstCharacters + i[0]
    if FirstCharacters.upper() == short.upper():
        return True
    else:
        return False


def split_iter(list1: list[T], batch_num: int) -> Generator[list[T], None, None]:
    split_points = np.linspace(0, len(list1), batch_num + 1, dtype="uint64")
    for i in range(batch_num):
        yield list1[split_points[i] : split_points[i + 1]]


def compare(list1, list2):
    list1 = [word.lower() for word in list1]
    return list1 == list2


def clean_string(company):
    """
    company - String, the company name to clean
    suffixs - list, all the suffixs that have to be removed

    Suffixs are found by:
    (1) Ignore the cases
    (2) Remove all the chars after "/" (e.g. "Inc/Old" to "Inc")
    (3) Count the frequencies of suffixs (1-word, 2-word and 3-word)
    (4) Add the word to suffixs if frequency >= 20
    """
    if company.find("/") != -1:
        company = company[: company.find("/")]
    company_name = company.split(" ")
    for i in range(len(company_name) - 1):
        for suffix in suffixs:
            suffix_name = suffix.split(" ")
            if len(company_name) <= 1:
                break
            if compare(company_name[-len(suffix_name) :], suffix_name):
                company_name = company_name[: -len(suffix_name)]
    company_name = " ".join(company_name)
    return company_name


def clean_string_len1(company):
    suffixs_len1 = []

    for suffix in suffixs:
        if len(suffix.split(" ")) == 1:
            suffixs_len1.append(suffix)

    company = company.strip(string.punctuation)
    if company.find("/") != -1:
        company = company[: company.find("/")]
    company_name = company.split(" ")
    for i in range(len(company_name) - 1):
        for suffix in suffixs_len1:
            suffix_name = suffix.split(" ")
            if len(company_name) <= 1:
                break
            if compare(company_name[-len(suffix_name) :], suffix_name):
                company_name = company_name[: -len(suffix_name)]
    company_name = " ".join(company_name)
    return company_name


# ---------------------- Function for common mapping ---------------------- ##


def create_id(x):

    _id = uuid.uuid3(uuid.NAMESPACE_DNS, x)
    _id = bson.Binary.from_uuid(
        _id, uuid_representation=UuidRepresentation.PYTHON_LEGACY
    )
    return _id


def insert_into_db(data, collection):
    try:
        _id = uuid.uuid3(uuid.NAMESPACE_DNS, data["NER_Name"])
        _id = bson.Binary.from_uuid(
            _id, uuid_representation=UuidRepresentation.PYTHON_LEGACY
        )
        data["_id"] = _id
        collection.insert_one(data)
    except Exception as e:
        print(e)
        pass


def insert_mapped_dict(col, **kwargs):
    required_fields = [
        "match_u3id",
        "match_company",
        "ner_comp_similarity2",
        "ner_comp",
        "match_similarity",
        "Similarity_1st",
        "Similarity_2nd",
        "cleaned_ner_entity_1st",
        "matched_cleaned_comp_1st",
        "ner_entity_2nd",
        "matched_comp_2nd",
    ]

    # Ensure all required fields are present
    for field in required_fields:
        if field not in kwargs:
            raise ValueError(f"Missing required argument: {field}")

    # Create the mapping dictionary
    Mapping_dict = {
        "U3_Company_Number": int(kwargs["match_u3id"]),
        "Company_Name": kwargs["match_company"],
        "NER_Name": kwargs["ner_comp"],
        "NER_Original_Name": kwargs["ner_comp_similarity2"],
        "Similarity": kwargs["match_similarity"],
        "Similarity_1st": kwargs["Similarity_1st"],
        "Similarity_2nd": kwargs["Similarity_2nd"],
        "1st_cleaned_ner_entity": kwargs["cleaned_ner_entity_1st"],
        "1st_matched_cleaned_comp": kwargs["matched_cleaned_comp_1st"],
        "2nd_ner_entity": kwargs["ner_entity_2nd"],
        "2nd_matched_comp": kwargs["matched_comp_2nd"],
        "Pre_Defined": 0,
        "Updated": 0,
    }

    # Insert the dictionary into the database
    insert_into_db(Mapping_dict, col)
