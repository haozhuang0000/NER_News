from ner.db.mongo import db, create_id
from ner.db.write import batch_upsert


database = db

__all__ = ["database", "create_id", "batch_upsert"]
