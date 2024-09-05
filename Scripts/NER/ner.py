# -*- coding: utf-8 -*-
"""
NER Process the text fetched from the crawler.

- Able to recognize list of company, economy, sector or instrucment
"""

# import modules
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
from nltk import sent_tokenize
from logger import Log
from tqdm import tqdm
import argparse
import time
import spacy
import re
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from Mongodb.mongodb import MongoDBHandler
from typing import List, Dict
import json
_ = load_dotenv(find_dotenv())

# set nlp_model as global variable

# define functions
class NER_Ruler:
    """
    A class to enhance a spaCy NLP model with custom Named Entity Recognition (NER) rules.
    """

    def __init__(self, ruler_path: str = None):
        """
        Initializes the NER_Ruler with a specified ruler path.

        :param ruler_path: Path to the directory containing the patterns.jsonl file.
                           Defaults to environment variable 'RULER_PATH'.
        """
        self.ruler_path = ruler_path or os.getenv('RULER_PATH')
        if not self.ruler_path:
            raise ValueError("RULER_PATH must be specified either as a parameter or as an environment variable.")

        self.nlp_model = spacy.load("en_core_web_sm")
        # self._add_ruler()

    def _load_patterns(self, file_path: str) -> List[Dict]:
        """
        Loads patterns from a JSONL file.

        :param file_path: Path to the JSONL file containing patterns.
        :return: A list of patterns.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return [json.loads(line) for line in file]
        except Exception as e:
            raise RuntimeError(f"Failed to load patterns from {file_path}: {e}")

    def _add_ruler(self):
        """
        Adds a custom entity ruler to the NLP model using patterns defined in the specified JSONL file.
        """
        ruler = self.nlp_model.add_pipe('entity_ruler', before='ner')
        patterns_path = os.path.join(self.ruler_path, 'patterns.jsonl')
        patterns = self._load_patterns(patterns_path)
        ruler.add_patterns(patterns)

    def get_model(self) -> spacy.Language:
        """
        Returns the spaCy NLP model with the custom NER ruler.

        :return: The spaCy NLP model.
        """
        return self.nlp_model

# Initialize and retrieve the NLP model
# ner_ruler = NER_Ruler()
# nlp_model = ner_ruler.get_model()



class NER_TextProcessor(MongoDBHandler, NER_Ruler):

    def __init__(self, in_col: str, out_col: str, inserted_threshold: int=1000) -> None:

        super().__init__()
        self.get_model()
        self.raw_db = self.get_database('local')
        self.db = self.get_database('Text_Preprocessed')

        self.in_col = self.raw_db[in_col]
        self.out_col = self.db[out_col]

        self.inserted_threshold = inserted_threshold
        self.logger = Log(f'{os.path.basename(__file__)}').getlog()

    def transform_data_to_results_dict(self, find_cursor: list) -> dict:

        results_dict = {"_id": [], "Date": [], "Title": [], "Author": [], "Content": [], "Category": []}

        for result in find_cursor:
            results_dict["_id"].append(result["_id"])
            results_dict["Date"].append(result["Date"])
            results_dict["Title"].append(' '.join(result["Title"].split()))
            results_dict["Author"].append(result["Author"])
            results_dict["Content"].append(' '.join(result["Content"].split()))
            results_dict["Category"].append(result["Category"])

        return results_dict

    def get_raw_text_from_mongo(self) -> list:

        input_col = self.in_col
        check_col_for_ner = self.out_col

        pipeline = [{"$group": {"_id": "$_id"}}]
        check_col_ner_ids = list(check_col_for_ner.aggregate(pipeline))
        input_col_ids = list(input_col.aggregate(pipeline))

        check_col_ner_ids_list = [item["_id"] for item in check_col_ner_ids]
        input_col_ids_list = [item["_id"] for item in input_col_ids]

        clear_input_list = list(set(input_col_ids_list).difference(set(check_col_ner_ids_list)))

        results_dict = {"_id": [], "Date": [], "Title": [], "Author": [], "Content": [], "Category": []}

        if len(clear_input_list) < 5e5:
            find_cursor = input_col.find({"_id": {'$in': clear_input_list}})
            results_dict = self.transform_data_to_results_dict(find_cursor)
        else:
            n_cores = 20  # number of splits
            total_size = len(clear_input_list)
            batch_size = round(total_size / n_cores + 0.5)
            skips = range(0, n_cores * batch_size, batch_size)
            for skip_n in skips:
                find_cursor = input_col.find({"_id": {'$in': clear_input_list[skip_n: skip_n + batch_size]}})
                temp_dict = self.transform_data_to_results_dict(find_cursor)
                for key in temp_dict.keys():
                    results_dict[key].extend(temp_dict[key])

        return list(pd.DataFrame(results_dict).to_dict('index').values())

    def is_entity(self, doc: str) -> bool:
        ##
        possible_num_entity = [
            '888', '104', '365', '20-20', '24', '9278', '525', '521', '7', '360', '1700', '724', '8888', '77', '600', '2',
            '36', '66', '8990', '786', '524', '1933', '577', '789', '1369', '111', '1834', '471', '5:01', '1844', '908',
            '11', '715', '1&1', '7936567', '8088', '8000', '1111', '1847', '88', '92', '727', '79', '235', '141', '37',
            '401', '87', '118000'
        ]
        if doc in possible_num_entity:
            return True

        if '$' in doc and doc not in ['Mid Amer Life Assur $4 Par', 'Lot$ OFF Corp', 'Buck-A-Roo$ Holding Corp', 'Mid Amer Life Assur $4 Par', 'Lot$ OFF', 'Buck-A-Roo$ Holding']:
            return False
        doc = re.sub('Q1|Q2|Q3|Q4|FY|bps|am|pm', '', doc)

        if re.search('[a-zA-Z]', doc) is None:
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

    def generate_text_sentence(self, pair: dict) ->dict:
        curr_dict = pair
        text = curr_dict["Content"]
        re_websites = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        websites_list = list(set(re.findall(re_websites, text)))

        for website in websites_list:
            text = text.replace(website, '')

        get_dot_list = list(set(re.findall(r'[\([0-9A-Za-z]+\.[0-9A-Za-z]+\)|\(\.[0-9A-Za-z]+\)|\([0-9A-Za-z]+\.\)]', text)))
        replace_dot_list = list(map(lambda x: x.replace('.', '@@'), get_dot_list))
        for i in range(len(get_dot_list)):
            text = text.replace(get_dot_list[i], replace_dot_list[i])

        text = re.sub(r'([^0-9])\.([^0-9])', r'\1. \2', text)
        text = re.sub("-{2,}", "..", text)
        text = re.sub(r'([0-9A-Za-z])\n([0-9A-Za-z])', r'\1.\n\n \2', text)
        text = re.sub(r'([0-9A-Za-z])\n\n([0-9A-Za-z])', r'\1.\n\n \2', text)

        tokenized_text = sent_tokenize(text)
        for i in range(len(tokenized_text)):
            if '@@' in tokenized_text[i]:
                for comp in replace_dot_list:
                    if comp in tokenized_text[i]:
                        idx = replace_dot_list.index(comp)
                        tokenized_text[i] = tokenized_text[i].replace(comp, get_dot_list[idx])

        sentence_list, companies_list, econs_list = self.replace_company_name(tokenized_text)

        for i in range(len(get_dot_list)):
            text = text.replace(replace_dot_list[i], get_dot_list[i])

        ner_dict = {
            "_id": curr_dict["_id"],
            "Date": curr_dict["Date"],
            "Content": text,
            "Category": curr_dict["Category"],
            "Title": curr_dict["Title"],
            "Sentence_list": sentence_list,
            "Companies_list": companies_list,
            "Econs_list": econs_list
        }
        return ner_dict

    def batch_helper(self, batch: list) -> int:

        ner_col = self.out_col
        ner_input_dict_list = []
        inserted_num = 0
        inserted_per_round = 0
        try:
            for pair in tqdm(batch):
                ner_dict = self.generate_text_sentence(pair)
                ner_check = ner_col.find_one({"_id": ner_dict["_id"]})
                if ner_check is None:
                    ner_input_dict_list.append(ner_dict)
                    inserted_per_round += 1
                    if len(ner_input_dict_list) >= self.inserted_threshold:
                        inserted_num += len(ner_input_dict_list)
                        ner_col.insert_many(ner_input_dict_list)
                        ner_input_dict_list = []
                        inserted_per_round = 0
            if inserted_per_round > 0:
                inserted_num += inserted_per_round
                ner_col.insert_many(ner_input_dict_list)
        except Exception as e:
            print(e)
        return inserted_num

    def run(self) -> None:

        logger = Log(f'{os.path.basename(__file__)}').getlog()
        logger.info(f"NER in a full-run.")
        start_time = time.time()
        list_of_dict = self.get_raw_text_from_mongo()
        logger.info("the list of dict size is {}".format(len(list_of_dict)))
        logger.info("load data from mongodb time: {}".format(time.time() - start_time))

        # ----- PARALLEL RUNNING BATCH --------
        if len(list_of_dict) == 0:
            logger.info("No new data!")
        else:
            start_time = time.time()
            logger.info("processing data...")
            # parallel_results = Parallel(n_jobs=5)(delayed(batch_helper)(batch) for batch in split_iter(list_of_dict, 5))
            # logger.info("{} articles added to ner database".format(sum(parallel_results)))
            logger.info("{} finished in time: {}".format("NER", time.time() - start_time))
            self.batch_helper(list_of_dict)
        logger.info("all done")

# implement
if __name__ == "__main__":

    ## set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        nargs='News',
                        type=str,
                        help="collection name in MongoDB")
    parser.add_argument('--output',
                        default="ner_out",
                        type=str,
                        help="Specify output ner collection")
    args, unknown = parser.parse_known_args()
    ner_rawdata_processor = NER_TextProcessor(in_col=args.input, out_col=args.output)
    ner_rawdata_processor.run()
