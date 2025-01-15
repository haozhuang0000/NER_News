# NER_News
This repository focuses on recognizing entities in news articles and mapping them to predefined entity names with corresponding IDs.

## Install environment

Python Version: Python 3.10 and above.
Libraries : As outlined in requirements.txt 

- Please follow the steps to download Mongodb: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/
- Please follow the steps to download Milvus: https://milvus.io/docs/install_standalone-docker.md
- pip3 install -r requirements.txt
- If running embedding model in your local pc, Please:
  - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- Please follow the steps to install Docker Desktop: https://www.docker.com/products/docker-desktop/
  - If `docker compose` is not installed in the environment, please check https://docs.docker.com/compose/install/linux/

## Code running instruction

Prerequisite for `generate_mapping_company.py`:

- Please create Milvus vector datebase use this code `Create_VDB.py` under `Scripts.VDB_Similiarity_Search`
- Setting for Embedding model, either `Server` or `Local`

Option 1:
- run code step by step

1. run `mongodb.py` to insert raw news data
2. run `ner.py` to extract companies for each sentence
3. run `ner_output_processor.py` to further process from ner
4. run `generate_mapping_company.py --embedding_method Local` to do similarity calculation

Option 2:

run code in one shot:

- run `main.py --embedding_method Local` for all functionality


## Docker Commands and starting server

Start docker compose as daemon process.
```
docker compose <name-of-file>.yml up -d

```
To start docker compose and view logs
```
docker compose <name-of-file>.yml up
```

To shut down docker file
```
docker compose <name-of-file>.yml down
```


## Troubleshooting and FAQ

### Environment Variable not found.

Check the config.py to configure. Anything caught within the error should be handled.

### Unable to write files or no data.

Due to resource constraints, OOM/insufficient inodes are common issues in local development. Clear the cache to reset.

### No airflow logs

Ensure a logs file is mounted in the docker-compose file. It doesn't need to exist in local, as long as a path is specified on the file.
