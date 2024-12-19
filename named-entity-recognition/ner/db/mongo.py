from pymongo import MongoClient
import uuid
from bson.binary import UuidRepresentation
import bson


from ner.config import MONGODB_HOST, MONGODB_DB


def create_id(collection: str, identifier_string: str):
    _id = uuid.uuid3(uuid.NAMESPACE_DNS, collection + identifier_string)
    _id = bson.Binary.from_uuid(
        _id, uuid_representation=UuidRepresentation.PYTHON_LEGACY
    )
    return _id


def connect_db(dbs=MONGODB_DB):

    client = MongoClient(MONGODB_HOST)
    db = client[dbs]
    return db


db = connect_db()
