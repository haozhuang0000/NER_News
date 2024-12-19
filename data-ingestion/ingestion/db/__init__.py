from ingestion.db.mongo import db, create_id
from ingestion.db.write import batch_upsert


database = db

__all__ = ["database", "create_id", "batch_upsert"]
