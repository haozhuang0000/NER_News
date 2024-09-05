# NER_News

## install environment

- Please follow the steps to download Mongodb: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/
- Please follow the steps to download Milvus: https://milvus.io/docs/install_standalone-docker.md
- pip3 install -r requirements.txt
- If running embedding model in your local pc, Please:
  - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Code running instruction

Prerequiest for `generate_mapping_company.py`:

- Please create Milvus vector datebase use this code `Create_VDB.py` under `Scripts.VDB_Similiarity_Search`
- Setting for Embedding model, either `Server` or `Local`

Option 1:
- run code step by step

1. run `mongodb.py` to insert raw news data
2. run `ner.py` to extract companies for each sentence
3. run `ner_output_processor.py` to further process from ner
4. run `generate_mapping_company.py --embedding_method Local` to do similarity calculation

Option 2:
- run code in one shot

run `run.py --embedding_method Local` for all functionality