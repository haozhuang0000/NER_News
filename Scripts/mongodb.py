from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
import os
import json
from logger import Log
import uuid
import bson
from bson.binary import Binary, UuidRepresentation

_ = load_dotenv(find_dotenv())

def create_id(data):

    _id = uuid.uuid3(uuid.NAMESPACE_DNS, data['Title'] + data['Date'])
    _id = bson.Binary.from_uuid(_id, uuid_representation=UuidRepresentation.PYTHON_LEGACY)
    data['_id'] = _id
    return data

def get_database(DB='local'):
    """
    :param DB: Your mongodb database, default is local
    """
    DB_URL = os.environ['LOCAL_URL']
    client = MongoClient(DB_URL)
    db = client[DB]

    return db

def insert_raw_data(raw_data_path):

    # Read data from the JSON file
    with open(raw_data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    db = get_database()
    col = db['News']
    out_data = []
    for i in data:
        data_temp = create_id(i)
        out_data.append(data_temp)

    col.insert_many(out_data)
    logger.info(f"Inserted {len(out_data)} articles!!")

if __name__ == '__main__':

    logger = Log("MongoDB").getlog()
    logger.info(f"Running {os.path.basename(__file__)}")
    try:
        raw_data_path = os.environ['RAW_DATA_PATH']
    except:
        raise KeyError("RAW_DATA_PATH environment variable not set.")

    insert_raw_data(raw_data_path)