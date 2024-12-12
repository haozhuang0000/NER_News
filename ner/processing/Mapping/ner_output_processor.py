"""
This script serves as an NER output processor, preparing input data for similarity
mapping.
"""

from joblib import Parallel, delayed
from logger import Log
from tqdm import tqdm
import pandas as pd
import argparse
import time

import os
from ner.db import database, batch_upsert, create_id
from ner.processing.Mapping.helper_functions import split_iter
from ner.models import NEROut, SentenceInput


class NerOutputProcessor(database):

    def __init__(
        self, in_col: str, out_col: str, inserted_threshold: int = 1000
    ) -> None:

        super().__init__()
        self.db = database
        self.in_col = self.db[in_col]
        self.out_col = self.db[out_col]
        self.inserted_threshold = inserted_threshold
        self.logger = Log(f"{os.path.basename(__file__)}").getlog()

    def transform_data_to_results_dict(self, find_cursor: list) -> dict:
        """
        Transforms MongoDB cursor output into a dictionary of results.

        Parameters:
        - find_cursor (list): A list of documents returned by MongoDB.

        Returns:
        - dict: A dictionary where each key is a unique identifier (e.g., _id)
                and the value is the corresponding document.
        """

        results_dict = {
            "_id": [],
            "Date": [],
            "Title": [],
            "Content": [],
            "Category": [],
            "Sentence_list": [],
            "Companies_list": [],
            "Econs_list": [],
        }
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
        # [DEPECRATED] Data is capture via NER_Data, not read from DB.
        """
        Retrieves NER data from collection `ner_out`.

        Returns:
        - list: A list of data
        """
        input_col = self.in_col
        check_col_for_bert = self.out_col

        pipeline = [{"$group": {"_id": "$_id"}}]
        check_col_bert_ids = check_col_for_bert.aggregate(pipeline)
        input_col_ids = input_col.aggregate(pipeline)
        check_col_bert_ids_list = []
        input_col_ids_list = []
        for item in check_col_bert_ids:
            check_col_bert_ids_list.append(item["_id"])
        for item in input_col_ids:
            input_col_ids_list.append(item["_id"])
        clear_input_list = list(
            set(input_col_ids_list).difference(set(check_col_bert_ids_list))
        )
        results_dict = {
            "_id": [],
            "Date": [],
            "Title": [],
            "Content": [],
            "Category": [],
            "Sentence_list": [],
            "Companies_list": [],
            "Econs_list": [],
        }
        # define the result dict
        if len(clear_input_list) < 5e5:
            find_cursor = input_col.find({"_id": {"$in": clear_input_list}})
            results_dict = self.transform_data_to_results_dict(find_cursor)
        else:
            n_cores = 20  # number of splits
            total_size = len(clear_input_list)
            batch_size = round(total_size / n_cores + 0.5)
            skips = range(0, n_cores * batch_size, batch_size)
            for skip_n in tqdm(skips):
                find_cursor = input_col.find(
                    {"_id": {"$in": clear_input_list[skip_n : skip_n + batch_size]}}
                )
                temp_dict = self.transform_data_to_results_dict(find_cursor)
                for key in list(temp_dict.keys()):
                    results_dict[key].extend(temp_dict[key])

        return list(pd.DataFrame(results_dict).to_dict("index").values())

    def replace_company_name(self, pair: NEROut) -> list[str]:
        """
        Replaces company names in the provided dictionary with their corresponding
        locations.

        @params pair: NEROut record
        @return tagged_text_list:
            A list of updated items where company names are replaced with locations.
        """
        sentence_list, companies_list, econs_list = (
            pair.Sentence_list,
            pair.Companies_list,
            pair.Econs_list,
        )
        tagged_text_list = []
        for num in range(len(sentence_list)):
            tagged_text = sentence_list[num]
            # company_econ_list = dict(companies_list[num], **econs_list[num])
            for i, item in enumerate(companies_list[num]):
                tagged_text = tagged_text.replace(item, "locationC" + str(i + 1), 1)
            for i, item in enumerate(econs_list[num]):
                tagged_text = tagged_text.replace(item, "locationE" + str(i + 1), 1)
            tagged_text_list.append(tagged_text)
        return tagged_text_list

    def generate_sentences_info(self, pair: NEROut):
        """
        Generates information for sentences based on company and economic entities.

        @params pair: NEROut object from out_col database.
        @return SentenceInput:
            Record containing sentence IDs, the sentences themselves, and associated
            company/economic entities.
        """

        sentence_id = []
        sentence_1 = []
        sentence_2 = []
        Companies_econs = []

        tagged_text_list = self.replace_company_name(pair)
        companies_list, econs_list = pair.Companies_list, pair.Econs_list

        for i in range(len(companies_list)):
            company_econ_list = companies_list[i] + econs_list[i]
            if len(company_econ_list) == 0:
                sentence_1.append(tagged_text_list[i])
                sentence_2.append("")
                Companies_econs.append("")
                sentence_id.append(i)
            else:
                for j in range(len(companies_list[i])):
                    sentence_id.append(i)
                    sentence_1.append(tagged_text_list[i])
                    sentence_2.append("locationC" + str(j + 1))
                    Companies_econs.append(companies_list[i][j])
                for j in range(len(econs_list[i])):
                    sentence_id.append(i)
                    sentence_1.append(tagged_text_list[i])
                    sentence_2.append("locationE" + str(j + 1))
                    Companies_econs.append(econs_list[i][j])

        record_id = create_id(pair.Title, None, pair._id)

        result_dict = {
            "_id": record_id,
            "article_id": pair.article_id,
            "Title": pair.Title,
            "Date": pair.Date,
            "ner_out_id": pair._id,
            "Companies_econs": Companies_econs,
            "Sentence_id": sentence_id,
            "Sentence_1": sentence_1,
            "Sentence_2": sentence_2,
        }

        return SentenceInput(**result_dict)

    def batch_helper(self, batch: list[NEROut]) -> int:
        """
        Processes a batch of data, generates sentence information, and inserts it into
        MongoDB collection.

        @param batch: list[NEROut]
        @return results: list[SentenceInput]
        """
        sentence_col = self.out_col
        sentence_output_record: list[SentenceInput] = []

        try:
            for pair in tqdm(batch):
                sentence_output_record.append(self.generate_sentences_info(pair))
            batch_upsert(sentence_col, sentence_output_record)
            return sentence_output_record
        except Exception as e:
            self.logger.info(e)

    def run(self, ner_data: list[NEROut]):
        """
        Main method to run the NER output processor.
        It loads data from MongoDB, processes it in batches, and inserts the results
        back into the database.
        """

        self.logger.info("Ner Output processor is running...")
        start_time = time.time()
        self.logger.info("loading data from mongodb...")
        self.logger.info("the list of dict size is {}".format(len(ner_data)))
        self.logger.info(
            "load data from mongodb time: {}".format(time.time() - start_time)
        )

        # ----- RAY RUNNING BATCH --------
        try:
            start_time = time.time()
            self.logger.info("processing data...")
            parallel_results = Parallel(n_jobs=1)(
                delayed(self.batch_helper)(batch) for batch in split_iter(ner_data, 1)
            )
            self.logger.info(
                "{} sentences added to database".format(sum(parallel_results))
            )

            self.logger.info("process time: {}".format(time.time() - start_time))
            self.logger.info("all done")
            return parallel_results
        except Exception as e:
            self.logger.info(f"Exception occured at {e}")
            return []


if __name__ == "__main__":
    # set parameters
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", default=r"ner_out", type=str, help="Specify input ner collection."
    )
    parser.add_argument(
        "--output",
        default=r"sentence_split",
        type=str,
        help="Specify output sentence_split collection.",
    )

    args, unknown = parser.parse_known_args()
    in_col = args.input
    out_col = args.output

    nerout_processor = NerOutputProcessor(in_col, out_col)
    nerout_processor.run()
