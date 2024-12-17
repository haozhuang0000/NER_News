from similarity_calculation.db.mongo import db, create_id
from similarity_calculation.db.write import batch_upsert, upsert_record


database = db

__all__ = ["database", "create_id", "batch_upsert", "upsert_record"]
