from pymongo import MongoClient
import os
import json
from logger import Log
import uuid
import bson
from bson.binary import Binary, UuidRepresentation
import pandas as pd
import warnings
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

class MongoDBHandler:

    def __init__(self):
        super().__init__()
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

    def insert_raw_data(self):

        """
        This function is used to insert raw text data into database `News`

        :param raw_data_path: Path that stored .json file
        :return: length of inserted article
        """
        logger = Log("MongoDB").getlog()
        logger.info(f"Running {os.path.basename(__file__)}")

        try:
            raw_data_path = os.environ['RAW_DATA_PATH']
        except:
            raise KeyError("RAW_DATA_PATH environment variable not set.")

        with open(raw_data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        db = self.get_database('local')
        col = db['News']
        out_data = [self.create_id(i) for i in data]
        try:
            col.insert_many(out_data)
            logger.info(f"Inserted {len(out_data)} raw articles!!")
        except Exception as e:
            warnings.warn(str(e))
            pass


if __name__ == '__main__':

    mongodb_handler = MongoDBHandler()
    len_data = mongodb_handler.insert_raw_data()
