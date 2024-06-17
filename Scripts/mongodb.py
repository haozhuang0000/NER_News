from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
import os
import json
from logger import Log
import uuid
import bson
from bson.binary import Binary, UuidRepresentation
import pandas as pd
_ = load_dotenv(find_dotenv())

class MongoDBHandler:

    def __init__(self):
        self.DB_URL = os.environ['LOCAL_URL']
        self.client = MongoClient(self.DB_URL)

    def get_database(self, DB='local'):
        """
        :param DB: Your mongodb database, default is local
        """
        db = self.client[DB]
        return db

    def create_id(self, data):
        """
        This is function is used to create unique id base on article `Title` & `Date`

        :param data: Input dict does not have unique ID
        :return: Output dict with unique ID `_id`
        """

        _id = uuid.uuid3(uuid.NAMESPACE_DNS, data['Title'] + data['Date'])
        _id = bson.Binary.from_uuid(_id, uuid_representation=UuidRepresentation.PYTHON_LEGACY)
        data['_id'] = _id
        return data

    def insert_raw_data(self, raw_data_path):

        """
        This function is used to insert raw text data into database `News`

        :param raw_data_path: Path that stored .json file
        :return: length of inserted article
        """
        with open(raw_data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        db = self.get_database('local')
        col = db['News']
        out_data = [self.create_id(i) for i in data]

        col.insert_many(out_data)
        return len(out_data)

    def transform_data_to_results_dict(self, find_cursor):

        results_dict = {"_id": [], "Date": [], "Title": [], "Author": [], "Content": [], "Category": []}

        for result in find_cursor:
            results_dict["_id"].append(result["_id"])
            results_dict["Date"].append(result["Date"])
            results_dict["Title"].append(' '.join(result["Title"].split()))
            results_dict["Author"].append(result["Author"])
            results_dict["Content"].append(' '.join(result["Content"].split()))
            results_dict["Category"].append(result["Category"])

        return results_dict

    def get_raw_text_from_mongo(self, collection_name, out_col):

        in_db = self.get_database('local')
        out_db = self.get_database('Text_Prerocessing')
        input_col = in_db[collection_name]
        check_col_for_ner = out_db[out_col]

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

if __name__ == '__main__':

    logger = Log("MongoDB").getlog()
    logger.info(f"Running {os.path.basename(__file__)}")
    try:
        raw_data_path = os.environ['RAW_DATA_PATH']
    except:
        raise KeyError("RAW_DATA_PATH environment variable not set.")

    mongodb_handler = MongoDBHandler()
    len_data = mongodb_handler.insert_raw_data(raw_data_path)
    logger.info(f"Inserted {len_data} raw articles!!")