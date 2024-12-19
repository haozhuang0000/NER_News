# -*- coding: utf-8 -*-
"""
NER Process the text fetched from the crawler.

- Able to recognize list of company, economy, sector or instrucment
"""

# import modules
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

import os
import re
import time

from logger import Log
from nltk import sent_tokenize
from tqdm import tqdm

from ner.config import (
    DEFAULT_RAW_DATA_COLLECTION,
    DEFAULT_TEXT_PROCESSED_COLLECTION,
)
from ner.db import create_id, database, pull_mongo_data, batch_upsert
from ner.models import Article, NEROut
from ner.processing.Named_Entity_Recognition.ner_ruler import NER_Ruler


class NER_TextProcessor(NER_Ruler):

    def __init__(
        self,
        in_col: str = DEFAULT_RAW_DATA_COLLECTION,
        out_col: str = DEFAULT_TEXT_PROCESSED_COLLECTION,
        inserted_threshold: int = 1000,
    ) -> None:

        super().__init__()
        self.get_model()
        self.db = database

        self.in_col = self.db[in_col]
        self.out_col = self.db[out_col]
        self.inserted_threshold = inserted_threshold
        self.logger = Log(f"{os.path.basename(__file__)}").getlog()

    def is_entity(self, doc: str) -> bool:
        possible_num_entity = [
            "888",
            "104",
            "365",
            "20-20",
            "24",
            "9278",
            "525",
            "521",
            "7",
            "360",
            "1700",
            "724",
            "8888",
            "77",
            "600",
            "2",
            "36",
            "66",
            "8990",
            "786",
            "524",
            "1933",
            "577",
            "789",
            "1369",
            "111",
            "1834",
            "471",
            "5:01",
            "1844",
            "908",
            "11",
            "715",
            "1&1",
            "7936567",
            "8088",
            "8000",
            "1111",
            "1847",
            "88",
            "92",
            "727",
            "79",
            "235",
            "141",
            "37",
            "401",
            "87",
            "118000",
        ]
        if doc in possible_num_entity:
            return True

        if "$" in doc and doc not in [
            "Mid Amer Life Assur $4 Par",
            "Lot$ OFF Corp",
            "Buck-A-Roo$ Holding Corp",
            "Mid Amer Life Assur $4 Par",
            "Lot$ OFF",
            "Buck-A-Roo$ Holding",
        ]:
            return False
        doc = re.sub("Q1|Q2|Q3|Q4|FY|bps|am|pm", "", doc)

        if re.search("[a-zA-Z]", doc) is None:
            return False

        return True

    def find_company_name(self, content_text: str) -> tuple[str, str, str]:
        company_entity = []
        econ_entity = []
        ner_words = self.nlp_model(content_text)

        for ent in ner_words.ents:
            if not self.is_entity(ent.text):
                continue
            if ent.label_ == "ORG":
                company_entity.append(ent.text)
            elif ent.label_ == "GPE":
                econ_entity.append(ent.text)

        return content_text, company_entity, econ_entity

    def replace_company_name(self, tokenized_text: str) -> tuple[list, list, list]:
        sentence_list = []
        companies_list = []
        econs_list = []

        for text in tokenized_text:
            after_del, company_list, econ_list = self.find_company_name(text)
            sentence_list.append(after_del)
            companies_list.append(company_list)
            econs_list.append(econ_list)

        return sentence_list, companies_list, econs_list

    def generate_text_sentence(self, record: Article) -> NEROut:
        text = record.Content
        re_websites = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        websites_list = list(set(re.findall(re_websites, text)))

        for website in websites_list:
            text = text.replace(website, "")

        get_dot_list = list(
            set(
                re.findall(
                    r"[\([0-9A-Za-z]+\.[0-9A-Za-z]+\)|\(\.[0-9A-Za-z]+\)|\([0-9A-Za-z]+\.\)]",
                    text,
                )
            )
        )
        replace_dot_list = list(map(lambda x: x.replace(".", "@@"), get_dot_list))
        for i in range(len(get_dot_list)):
            text = text.replace(get_dot_list[i], replace_dot_list[i])

        text = re.sub(r"([^0-9])\.([^0-9])", r"\1. \2", text)
        text = re.sub("-{2,}", "..", text)
        text = re.sub(r"([0-9A-Za-z])\n([0-9A-Za-z])", r"\1.\n\n \2", text)
        text = re.sub(r"([0-9A-Za-z])\n\n([0-9A-Za-z])", r"\1.\n\n \2", text)

        tokenized_text = sent_tokenize(text)
        for i in range(len(tokenized_text)):
            if "@@" in tokenized_text[i]:
                for comp in replace_dot_list:
                    if comp in tokenized_text[i]:
                        idx = replace_dot_list.index(comp)
                        tokenized_text[i] = tokenized_text[i].replace(
                            comp, get_dot_list[idx]
                        )

        sentence_list, companies_list, econs_list = self.replace_company_name(
            tokenized_text
        )

        for i in range(len(get_dot_list)):
            text = text.replace(replace_dot_list[i], get_dot_list[i])

        new_record_id = create_id(
            DEFAULT_TEXT_PROCESSED_COLLECTION, record.Title + record.Date + record._id
        )

        ner_dict = {
            "_id": new_record_id,
            "article_id": record.article_id,
            "News_id": record._id,
            "Title": record.Title,
            "Date": record.Date,
            "Content": text,
            "Category": record.Category,
            "Search_q": record.Search_q,
            "Sentence_list": sentence_list,
            "Companies_list": companies_list,
            "Econs_list": econs_list,
        }
        return NEROut(**ner_dict)

    def batch_helper(self, batch: list[Article]):
        # TODO: Do an early return/submit processed IDs to the global queue
        #       Then write to database
        processed_ner_objects: list[NEROut] = []
        try:
            for pair in tqdm(batch):
                ner_out_record = self.generate_text_sentence(pair)
                processed_ner_objects.append(ner_out_record)
            batch_upsert(self.out_col, processed_ner_objects)
            return processed_ner_objects
        except Exception as e:
            print(e)
            return []

    def run(self):

        logger = Log(f"{os.path.basename(__file__)}").getlog()
        logger.info("NER in a full-run.")
        start_time = time.time()
        article_data = pull_mongo_data(self.in_col, self.out_col, "Article")
        logger.info("the list of dict size is {}".format(len(article_data)))
        logger.info("load data from mongodb time: {}".format(time.time() - start_time))

        try:
            start_time = time.time()
            logger.info("processing data...")

            result = self.batch_helper(article_data)
            logger.info(
                "{} finished in time: {}".format("NER", time.time() - start_time)
            )
            return result
        except Exception as e:
            logger.info(f"An exception has occured at {e}")
            return []
