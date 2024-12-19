import numpy as np
from pymongo import UpdateOne
from pymongo.collection import Collection
from typing import TypeVar, Generator
from dataclasses import asdict

from ner.db.mongo import db

T = TypeVar("T")


def upsert_record(cn: str | Collection, record: T):
    collection = db[cn] if isinstance(cn) == str else cn

    collection.update_one(
        {"_id": record._id},
        {"$set": asdict(record)},
        upsert=True,
    )

    if collection.find_one({"_id": record._id}) != asdict(record):
        raise Exception(f"Upsert record {record._id} failed.")


def split_iter(list1: list[T], batch_num: int) -> Generator[list[T], None, None]:
    split_points = np.linspace(0, len(list1), batch_num + 1, dtype="uint64")
    for i in range(batch_num):
        yield list1[split_points[i] : split_points[i + 1]]


def bulk_upsert(collection, data: list[T]):
    collection.bulk_write(
        UpdateOne(
            {"_id": record._id},
            {"$set": asdict(record)},
            upsert=True,
        )
        for record in data
    )


def batch_upsert(collection_name: str | Collection, data: list[T]):
    if len(data) == 0:
        return

    collection = (
        db[collection_name] if isinstance(collection_name) == str else collection_name
    )

    bulk_upsert(collection, data)
