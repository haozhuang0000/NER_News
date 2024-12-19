import numpy as np
from typing import TypeVar, Generator
import string

from similarity_calculation.db import database

T = TypeVar("T")
SUFFIX_COLLECTION = "suffix"
COMPANY_NAME_COLLECTION = "company_names"
ECONOMIES_COLLECTION = "economies"

suffix_collection = database[SUFFIX_COLLECTION]
# ---------------------- Function for similarity calculation ---------------------- ##


def get_entity_names(
    company_list_name: str = COMPANY_NAME_COLLECTION,
    econ_list_name: str = ECONOMIES_COLLECTION,
    include_companies: bool = True,
    include_econs: bool = True,
) -> list[str]:
    """
    Retrieves entity names based on the specified parameters.

    @param include_companies (bool): Whether to include company names.
    @param include_econs (bool): Whether to include economic entity names.
    @param parent_dir (str): Directory where the entity lists are stored.
    @param company_list_name (str): Filename of the company list.
    @param econ_list_name (str): Filename of the economic entities list.
    @param logger: Logger for logging information and errors.

    @return: list[str] - List of entity names based on the input parameters.
    """
    combined_full_comp_list = []
    econ_list = []
    econ_id = []

    # input company names
    if include_companies:
        companies_collection = database[company_list_name]
        # ------- Sorted each different type of list for consistency ------- #
        # Full company list

        combined_full_comp_list.extend(
            [
                (doc["Company_name"], doc["CompanyID"])
                for doc in companies_collection.find(
                    {},
                    {"CompanyID": 1, "Company_name": 1, "_id": 0},
                )
            ]
        )
        combined_full_comp_list.sort(key=lambda x: x[0])

    if include_econs:
        economies_collection = database[econ_list_name]

        econ_list.extend(
            [
                (doc["econ_name"])
                for doc in economies_collection.find(
                    {},
                    {"econ_name": 1, "_id": 0},
                )
            ]
        )
        econ_id.extend(
            [
                (doc["econ_id"])
                for doc in economies_collection.find(
                    {},
                    {"econ_id": 1, "_id": 0},
                )
            ]
        )
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
    return [word.lower() for word in list1] == list2


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

    suffix_cursor = suffix_collection.find({}, {"after_delete": 1})
    if len(company_name) > 1:
        for _ in range(len(company_name) - 1):
            for suffix in suffix_cursor:
                suffix_name = suffix.split(" ")
                length = len(suffix_name)

                if compare(company_name[-length:], suffix_name):
                    company_name = company_name[:-length]
        company_name = " ".join(company_name)
    return company_name


def clean_string_len1(company):
    suffixs_len1 = []
    suffix_cursor = suffix_collection.find({}, {"after_delete": 1})

    for suffix in suffix_cursor:
        if len(suffix.split(" ")) == 1:
            suffixs_len1.append(suffix)

    company = company.strip(string.punctuation)
    if company.find("/") != -1:
        company = company[: company.find("/")]
    company_name = company.split(" ")
    if len(company_name) > 1:
        for _ in range(len(company_name) - 1):
            for suffix in suffixs_len1:
                suffix_name = suffix.split(" ")
                length = len(suffix_name)

                if compare(company_name[-length:], suffix_name):
                    company_name = company_name[:-length]
        company_name = " ".join(company_name)
    return company_name
