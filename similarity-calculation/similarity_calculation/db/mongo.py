from pymongo import MongoClient
import uuid
from bson.binary import UuidRepresentation
import bson


from similarity_calculation.config import MONGODB_HOST


def create_id(collection: str, identifier_string: str):
    _id = uuid.uuid3(uuid.NAMESPACE_DNS, collection + identifier_string)
    _id = bson.Binary.from_uuid(
        _id, uuid_representation=UuidRepresentation.PYTHON_LEGACY
    )
    return _id


def connect_db(dbs="AIDF_AlternativeData"):

    client = MongoClient(MONGODB_HOST)
    db = client[dbs]
    return db


db = connect_db()
