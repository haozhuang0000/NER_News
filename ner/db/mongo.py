from pymongo import MongoClient
import uuid
from bson.binary import UuidRepresentation
import bson
from datetime import datetime


from ner.config import MONGODB_HOST


def create_id(title: str, date: str | None, url: str):
    id_date = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),) if date is None else date
    _id = uuid.uuid3(uuid.NAMESPACE_DNS, str(title) + id_date + url)
    _id = bson.Binary.from_uuid(
        _id, uuid_representation=UuidRepresentation.PYTHON_LEGACY
    )
    return _id


def connect_db(dbs="AIDF_AlternativeData"):

    client = MongoClient(MONGODB_HOST)
    db = client[dbs]
    return db


db = connect_db()
