from pymongo import UpdateOne
from pymongo.collection import Collection
from typing import TypeVar
from dataclasses import asdict

from ingestion.db.mongo import db


T = TypeVar("T")


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
