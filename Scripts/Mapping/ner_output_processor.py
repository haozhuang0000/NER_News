"""
This script serves as an NER output processor, preparing input data for similarity mapping.
"""

from joblib import Parallel, delayed
from logger import Log
from tqdm import tqdm
import pandas as pd
import argparse
import time
import os
from Mongodb.mongodb import MongoDBHandler
from Mapping.helper_functions import split_iter

class NerOutputProcessor(MongoDBHandler):

    def __init__(self, in_col: str, out_col: str, inserted_threshold: int=1000) -> None:

        super().__init__()
        self.db = self.get_database('Text_Preprocessed')
        self.in_col = self.db[in_col]
        self.out_col = self.db[out_col]
        self.inserted_threshold = inserted_threshold
        self.logger = Log(f'{os.path.basename(__file__)}').getlog()

    def transform_data_to_results_dict(self, find_cursor: list) -> dict:
        """
        Transforms MongoDB cursor output into a dictionary of results.

        Parameters:
        - find_cursor (list): A list of documents returned by MongoDB.

        Returns:
        - dict: A dictionary where each key is a unique identifier (e.g., _id)
                and the value is the corresponding document.
        """

        results_dict = {"_id": [], "Date": [], "Title": [], "Content": [], "Category": [], "Sentence_list": [],
                        "Companies_list": [], "Econs_list": []}
        for result in tqdm(find_cursor):
            results_dict["_id"].append(result["_id"])
            results_dict["Date"].append(result["Date"])
            results_dict["Title"].append(result["Title"])
            results_dict["Content"].append(result["Content"])
            results_dict["Category"].append(result["Category"])
            results_dict["Sentence_list"].append(result["Sentence_list"])
            results_dict["Companies_list"].append(result["Companies_list"])
            results_dict["Econs_list"].append(result["Econs_list"])
        return results_dict

    def get_ner_data(self) -> list:
        """
        Retrieves NER data from collection `ner_out`.

        Returns:
        - list: A list of data
        """
        input_col = self.in_col
        check_col_for_bert = self.out_col

        pipeline = [
            {"$group": {"_id": "$_id"}}
        ]
        check_col_bert_ids = check_col_for_bert.aggregate(pipeline)
        input_col_ids = input_col.aggregate(pipeline)
        check_col_bert_ids_list = []
        input_col_ids_list = []
        for item in check_col_bert_ids:
            check_col_bert_ids_list.append(item["_id"])
        for item in input_col_ids:
            input_col_ids_list.append(item["_id"])
        clear_input_list = list(set(input_col_ids_list).difference(set(check_col_bert_ids_list)))
        results_dict = {"_id": [], "Date": [], "Title": [], "Content": [], "Category": [], "Sentence_list": [],
                        "Companies_list": [], "Econs_list": []}
        ## define the result dict
        if len(clear_input_list) < 5e5:
            find_cursor = input_col.find({"_id": {'$in': clear_input_list}})
            results_dict = self.transform_data_to_results_dict(find_cursor)
        else:
            n_cores = 20  # number of splits
            total_size = len(clear_input_list)
            batch_size = round(total_size / n_cores + 0.5)
            skips = range(0, n_cores * batch_size, batch_size)
            for skip_n in tqdm(skips):
                find_cursor = input_col.find({"_id": {'$in': clear_input_list[skip_n: skip_n + batch_size]}})
                temp_dict = self.transform_data_to_results_dict(find_cursor)
                for key in list(temp_dict.keys()):
                    results_dict[key].extend(temp_dict[key])

        return list(pd.DataFrame(results_dict).to_dict('index').values())

    def replace_company_name(self, pair: dict) -> list:
        """
        Replaces company names in the provided dictionary with their corresponding locations.

        Parameters:
        - pair (dict): A dictionary from get_ner_data()

        Returns:
        - list: A list of updated items where company names are replaced with locations.
        """
        sentence_list, companies_list, econs_list = pair['Sentence_list'], pair['Companies_list'], pair['Econs_list']
        tagged_text_list = []
        for num in range(len(sentence_list)):
            tagged_text = sentence_list[num]
            # company_econ_list = dict(companies_list[num], **econs_list[num])
            for i, item in enumerate(companies_list[num]):
                tagged_text = tagged_text.replace(item, 'locationC' + str(i + 1), 1)
            for i, item in enumerate(econs_list[num]):
                tagged_text = tagged_text.replace(item, 'locationE' + str(i + 1), 1)
            tagged_text_list.append(tagged_text)
        return tagged_text_list

    def generate_sentences_info(self, pair: dict) -> dict:
        """
        Generates information for sentences based on company and economic entities.

        Parameters:
        - pair (dict): A dictionary containing data such as company and economic entity lists.

        Returns:
        - dict: A dictionary containing sentence IDs, the sentences themselves, and associated company/economic entities.
        """

        sentence_id = []
        sentence_1 = []
        sentence_2 = []
        Companies_econs = []

        tagged_text_list = self.replace_company_name(pair)
        companies_list, econs_list = pair['Companies_list'], pair['Econs_list']

        for i in range(len(companies_list)):
            company_econ_list = companies_list[i] + econs_list[i]
            if len(company_econ_list) == 0:
                sentence_1.append(tagged_text_list[i])
                sentence_2.append('')
                Companies_econs.append('')
                sentence_id.append(i)
            else:
                for j in range(len(companies_list[i])):
                    sentence_id.append(i)
                    sentence_1.append(tagged_text_list[i])
                    sentence_2.append('locationC' + str(j + 1))
                    Companies_econs.append(companies_list[i][j])
                for j in range(len(econs_list[i])):
                    sentence_id.append(i)
                    sentence_1.append(tagged_text_list[i])
                    sentence_2.append('locationE' + str(j + 1))
                    Companies_econs.append(econs_list[i][j])

        result_dict = {'_id': pair['_id'], "Title": pair['Title'], "Date": pair['Date'], 'Companies_econs': Companies_econs,
                       'Sentence_id': sentence_id, 'Sentence_1': sentence_1, 'Sentence_2': sentence_2}

        return result_dict

    def batch_helper(self, batch: list) -> int:
        """
        Processes a batch of data, generates sentence information, and inserts it into a MongoDB collection.

        Parameters:
        - batch (list): A list of dictionaries, where each dictionary contains data needed for sentence generation.

        Returns:
        - int: The number of documents successfully inserted into the MongoDB collection.
        """
        sentence_col = self.out_col
        input_dict_list = []
        inserted_num = 0
        inserted_per_round = 0

        try:
            for pair in tqdm(batch):
                info_dict = self.generate_sentences_info(pair)
                check = sentence_col.find_one({"_id": info_dict["_id"]})
                if check == None:
                    input_dict_list.append(info_dict)
                    inserted_per_round += 1
                    if len(input_dict_list) >= self.inserted_threshold:
                        inserted_num = inserted_num + len(input_dict_list)
                        sentence_col.insert_many(input_dict_list)
                        input_dict_list = []
                        inserted_per_round = 0
            if inserted_per_round > 0:
                inserted_num = inserted_num + inserted_per_round
                sentence_col.insert_many(input_dict_list)
        except Exception as e:
            self.logger.info(e)

        return inserted_num

    def run(self) -> None:
        """
        Main method to run the NER output processor.
        It loads data from MongoDB, processes it in batches, and inserts the results back into the database.
        """

        self.logger.info(f"Ner Output processor is running...")
        start_time = time.time()
        self.logger.info("loading data from mongodb...")
        list_of_dict = self.get_ner_data()
        self.logger.info("the list of dict size is {}".format(len(list_of_dict)))
        self.logger.info("load data from mongodb time: {}".format(time.time() - start_time))

        if len(list_of_dict) == 0:
            self.logger.info("No new data")
        else:
            # ----- RAY RUNNING BATCH --------
            start_time = time.time()
            self.logger.info("processing data...")
            parallel_results = Parallel(n_jobs=1)(delayed(self.batch_helper)(batch) for batch in split_iter(list_of_dict, 1))
            self.logger.info("{} sentences added to database".format(sum(parallel_results)))

            self.logger.info("process time: {}".format(time.time() - start_time))
        self.logger.info("all done")



if __name__ == "__main__":
    # set parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        default=r"ner_out",
                        type=str,
                        help="Specify input ner collection.")
    parser.add_argument("--output",
                        default=r"sentence_split",
                        type=str,
                        help="Specify output sentence_split collection.")

    args, unknown = parser.parse_known_args()
    in_col = args.input
    out_col = args.output

    nerout_processor = NerOutputProcessor(in_col, out_col)
    nerout_processor.run()

