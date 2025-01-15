import os

"""
Environment Presets
"""
env_variables = [
    "MONGODB_HOST",
    "MONGODB_DB",
    "NEWS_API_KEY",
    "VDB_HOST",
    "HUGGINGFACE_TOKEN",
    "EMBEDDING_API",
]

for env_var in env_variables:
    if env_var not in os.environ:
        raise KeyError(f"{env_var} is not defined in the environment.")


MONGODB_HOST = os.getenv("MONGODB_HOST")
MONGODB_DB = os.getenv("MONGODB_DB")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DEFAULT_RAW_DATA_COLLECTION = os.getenv("DEFAULT_NEWS_MONGO_COLLECTION", "News")

VDB_HOST = os.getenv("VDB_HOST")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
EMBEDDING_API = os.getenv("EMBEDDING_API")
EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "Server")

"""
Non environment variables
"""
# AIRFLOW PRESETS


# NEWSAPI PRESETS
NEWS_API_TIMEFRAME = 5
NEWS_API_QUERY_SIZE = 50
NEWS_API_COUNTRIES = {
    "cn": "zh",
    "id": "id",
    "jp": "jp",
    "kr": "ko",
    "my": "ms",
    "th": "th",
    "us": "en",
}
NEWS_API_KEYWORDS = ["finance", "debt"]

# Named Entity Recognition Variables
DEFAULT_TEXT_PROCESSED_COLLECTION = "ner_out"
DEFAULT_SENTENCE_SPLIT_COLLECTION = "sentence_split"
DEFAULT_SELECTED_SENTENCE_COLLECTION = "selected_sentence"
DEFAULT_NER_MAPPING_COLLECTION = "ner_mapped"

# Collections
SUFFIX_COLLECTION = "suffix"
COMPANY_NAME_COLLECTION = "company_name"
ECONOMIES_COLLECTION = "economies"
