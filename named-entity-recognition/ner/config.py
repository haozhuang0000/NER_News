import os


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
    "KAFKA_TOPIC",
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
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")
BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS")

""" 
Non environment variables
"""
MAX_LIMIT = 10

# Named Entity Recognition Variables
DEFAULT_RAW_DATA_COLLECTION = "News"
DEFAULT_TEXT_PROCESSED_COLLECTION = "ner_out"
DEFAULT_SENTENCE_SPLIT_COLLECTION = "sentence_split"
DEFAULT_SELECTED_SENTENCE_COLLECTION = "selected_sentence"
DEFAULT_NER_MAPPING_COLLECTION = "ner_mapped"
EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "Local")
