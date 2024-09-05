import os
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
import requests
import json
from VDB_Similarity_Search.Model import NVEmbed
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

class MilvusDB():

    def __init__(self, host=os.environ['VDB_HOST'], port="19530"):
        super().__init__()
        self.host = host
        self.port = port
        self.milvus_client = MilvusClient("http://" + host + ":" + port)

    def _connect_vdb(self, col_name, **kwargs):

        connections.connect("default", host=self.host, port=self.port)
        description = kwargs['description']
        fields = kwargs['fields']
        schema = CollectionSchema(fields, description)
        col = Collection(col_name, schema, consistency_level="Strong")
        return col

    def _drop_vdb(self, col_name):

        self.milvus_client.drop_collection(col_name)

    def _add_index(self, col, nlist=4096, **kwargs):

        index_params = self.milvus_client.prepare_index_params()

        index_params.add_index(
            field_name=kwargs['pk_name']
        )
        index_params.add_index(
            field_name="embeddings",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": nlist}
        )
        self.milvus_client.create_index(
            collection_name=col,
            index_params=index_params
        )

    def vectorsearch(self, ner_company_name, embeddings=None, col_name='NER_Mapping',
                     limit=30, output_field=["u3_id", "company_name", 'Type']):

        if self.milvus_client.has_collection(col_name) is False:
            raise ValueError('Please create your vector database by running code under `Scripts.VDB_Similiarity_Search` -> `Create_VDB.py`'
                             'It is highly recommend that you run `Create_VDB.py` in a GPU Server')

        if embeddings is None:
            raise ValueError('Embeddings must be provided. Please set up your embedding model.')

        if isinstance(embeddings, NVEmbed):
            # Handle local embedding
            vector_to_search = embeddings.embed_query(ner_company_name)
        elif isinstance(embeddings, str) and (embeddings.startswith('http://') or embeddings.startswith('https://')):
            URL = embeddings
            data = {'input': ner_company_name, 'type': 'query'}
            response = requests.post(URL, json=data, verify=False)
            if response.status_code != 200:
                raise ConnectionError(
                    'Request failed with status code {}. Please contact the server admin.'.format(response.status_code))
            vector_to_search = response.json()
        else:
            raise ValueError('Invalid embeddings object. It must be either a SentenceTransformer embedding or a valid URL string.')

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 256},
        }
        result = self.milvus_client.search(
            collection_name=col_name,
            data=[vector_to_search],
            anns_field="embeddings",
            search_params=search_params,
            limit=limit,
            output_fields=output_field
        )
        return result