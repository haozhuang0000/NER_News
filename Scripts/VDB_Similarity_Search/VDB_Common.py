import os
import time
import torch
import numpy as np
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

class MilvusDB():

    def __init__(self, host=os.environ['VDB_HOST'], port="19430"):

        self.host = host
        self.port = port
        self.client = MilvusClient(host)

    def _connect_vdb(self, col_name, **kwargs):
        connections.connect("default", host=self.host, port=self.port)
        description = kwargs['description']
        fields = kwargs['fields']
        schema = CollectionSchema(fields, description)
        col = Collection(col_name, schema, consistency_level="Strong")
        return col

    def _drop_vdb(self, col_name):

        utility.drop_collection(col_name)

    def _add_index(self, col, nlist=4096, **kwargs):

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name=kwargs['pk_name']
        )
        index_params.add_index(
            field_name="embeddings",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": nlist}
        )
        self.client.create_index(
            collection_name=col,
            index_params=index_params
        )