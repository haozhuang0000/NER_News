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
from joblib import Parallel, delayed
from nltk import sent_tokenize
from nltk.stem.porter import *
import json
from logger import Log
from tqdm import tqdm
import numpy as np
import argparse
import time
import spacy
import re
import os
from dotenv import load_dotenv, find_dotenv
from mongodb import MongoDBHandler
from typing import List, Dict

_ = load_dotenv(find_dotenv())

# set nlp_model as global variable
# nlp_model = spacy.load(r"en_core_web_sm")
# nlp_model = spacy.load(r"\\dirac2\CRI3\nlp\data\API_Output\full_run\Others\final_v2")
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

    def __init__(self, in_col, out_col, inserted_threshold):
        MongoDBHandler.__init__(self)
        NER_Ruler.__init__(self)
        self.get_model()
        self.in_col = in_col
        self.out_col = out_col
        self.out_db = 'Text_Preprocessed'
        self.inserted_threshold = inserted_threshold


    def is_entity(self, doc):
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


    def find_company_name(self, content_text):
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

    def replace_company_name(self, tokenized_text):
        sentence_list = []
        companies_list = []
        econs_list = []

        for text in tokenized_text:
            after_del, company_list, econ_list = self.find_company_name(text)
            sentence_list.append(after_del)
            companies_list.append(company_list)
            econs_list.append(econ_list)

        return sentence_list, companies_list, econs_list

    def generate_text_sentence(self, pair):
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
        text = re.sub("-{2,}", ".", text)
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

    def batch_run(self, batch):

        out_db = self.get_database(self.out_db)
        ner_col = out_db[self.out_col]
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

    def split_iter(self, list1, batch_num):
        split_points = np.linspace(0, len(list1), batch_num + 1, dtype='uint64')
        for i in range(batch_num):
            yield list1[split_points[i]: split_points[i + 1]]
# implement
if __name__ == "__main__":

    ## set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--collections',
                        nargs='+',
                        type=str,
                        help="collection name in MongoDB")
    parser.add_argument('--output',
                        default=r"ner_out",
                        type=str,
                        help="Specify output ner collection")
    args, unknown = parser.parse_known_args()
    collections = args.collections
    out_col = args.output
    collections = ['News']

    logger = Log(f'{os.path.basename(__file__)}').getlog()
    logger.info(f"NER in a full-run.")

    for collection_name in collections:
        # load data
        start_time = time.time()
        ner_text_processor = NER_TextProcessor(in_col=collection_name, out_col=out_col, inserted_threshold=1000)
        logger.info(f"{collection_name} :loading data from mongodb...")
        list_of_dict = ner_text_processor.get_raw_text_from_mongo(collection_name, out_col)
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
            # logger.info("{} finished in time: {}".format(collection_name, time.time() - start_time))
            ner_text_processor.batch_run(list_of_dict)
    logger.info("all done")
