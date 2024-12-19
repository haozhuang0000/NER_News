import os

"""
Environment Presets
"""
env_variables = [
    "MONGODB_HOST",
    "VDB_HOST",
    "RAW_DATA_PATH",
    "RULER_PATH",
    "COMPANY_PATH_NAME",
    "ECON_PATH_NAME",
    "HUGGINGFACE_TOKEN",
    "EMBEDDING_API",
    "COMPANY_PATH_NAME",
    "MONGODB_DB",
    "NEWS_API_KEY",
    "BOOTSTRAP_SERVERS",
    "DEFAULT_NEWS_MONGO_COLLECTION",
]

for env_var in env_variables:
    if env_var not in os.environ:
        raise KeyError(f"{env_var} is not defined in the environment.")


MONGODB_HOST = os.getenv("LOCAL_HOST")
VDB_HOST = os.getenv("VDB_HOST")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
RULER_PATH = os.getenv("RULER_PATH")
COMPANY_PATH_NAME = os.getenv("COMPANY_PATH_NAME")
ECON_PATH_NAME = os.getenv("ECON_PATH_NAME")
HUGGINGFACE_TOKEN = os.getenv("COMPANY_PATH_NAME")
EMBEDDING_API = os.getenv("EMBEDDING_API")
COMPANY_PATH_NAME = os.getenv("COMPANY_PATH_NAME")
MONGODB_DB = os.getenv("DB")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS")
DEFAULT_NEWS_MONGO_COLLECTION = os.getenv("DEFAULT_NEWS_MONGO_COLLECTION", "News")

"""
Non environment variables
"""
MAX_LIMIT = 10

# AIRFLOW PRESETS


# NEWSAPI PRESETS
NEWS_API_TIMEFRAME = 12
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
