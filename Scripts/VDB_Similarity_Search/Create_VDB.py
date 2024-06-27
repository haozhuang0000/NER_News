import numpy as np
import torch
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
import time

from VDB_Common import MilvusDB
from Model import NVEmbed
from ..Config.config import *

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
def batch_insert(col_, entities, batch_size=2000):

    total = len(entities[0])
    for batch in range(0, total, batch_size):
        batch_end = min(batch + batch_size, total)
        col_.insert([
            entities[0][batch:batch_end],
            entities[1][batch:batch_end],
            entities[2][batch:batch_end],
            entities[3][batch:batch_end],
            entities[4][batch:batch_end],
        ])
        print(f"Done {batch}")

def prepare_data(embeddings):
    df_company = pd.read_csv(r'../../_static/all_company_names_v3_cleaned.csv')

    ## Full
    comp_df = df_company[['u3_num', 'Company_name']].dropna()
    comp_df['u3id_type'] = comp_df.u3_num.apply(lambda x: str(x) + "_full")
    u3_id = list(comp_df.u3_num)
    comp_name = list(comp_df.Company_name)
    comp_id_type = list(comp_df.u3id_type)

    ## Clean
    cleancomp_df = df_company[['u3_num', 'clean_name']].dropna()
    cleancomp_df['u3id_type'] = cleancomp_df.u3_num.apply(lambda x: str(x) + "_clean")
    clean_u3_id = list(cleancomp_df.u3_num)
    clean_name = list(cleancomp_df.clean_name)
    clean_name_id_type = list(cleancomp_df.u3id_type)

    ## Clean_s
    cleancomp_s_df = df_company[['u3_num', 'clean_name_stricter_suffix']].dropna()
    cleancomp_s_df['u3id_type'] = cleancomp_s_df.u3_num.apply(lambda x: str(x) + "_clean_s")
    clean_s_u3_id = list(cleancomp_s_df.u3_num)
    clean_name_strict = list(cleancomp_s_df.clean_name_stricter_suffix)
    clean_name_s_id_type = list(cleancomp_s_df.u3id_type)

    ## Common
    common_df = df_company[['u3_num', 'Common_Name']].dropna()
    common_df['u3id_type'] = common_df.u3_num.apply(lambda x: str(x) + "_common")

    com_id = list(common_df.u3_num)
    com_name = list(common_df.Common_Name)
    com_name_id_type = list(common_df.u3id_type)

    ## tickers

    df = df_company.rename(columns={'U3_COMPANY_NUMBER': 'U3_Company_Number'}).drop(
        columns=['Unnamed: 0', 'Prime_exchange', 'U4_COMPANY_ID'])
    df_company = df_company.rename(columns={'Company_name': 'Company_Name'})
    duplicates = df.duplicated('Ticker', keep=False)
    df_cleaned = df[~duplicates]
    df_cleaned = df_cleaned[['u3_num', 'Ticker']].dropna()
    df_cleaned['u3id_type'] = df_cleaned.u3_num.apply(lambda x: str(x) + "_ticker")

    tic_id = list(df_cleaned.u3_num)
    tic_name = list(df_cleaned.Ticker)
    tic_name_id_type = list(df_cleaned.u3id_type)

    ## full
    full_name_embedding = embeddings.embed_documents(comp_name)
    ## clean
    clean_name_embedding = embeddings.embed_documents(clean_name)
    ## clean_s
    clean_s_name_embedding = embeddings.embed_documents(clean_name_strict)
    ## com
    com_name_embedding = embeddings.embed_documents(com_name)
    ## tic
    tic_name_embedding = embeddings.embed_documents(tic_name)

    full_name_entities = [
        comp_id_type,
        u3_id,
        comp_name,
        ['full' for i in range(len(u3_id))],
        full_name_embedding
    ]

    clean_name_entities = [
        clean_name_id_type,
        clean_u3_id,
        clean_name,
        ['clean' for i in range(len(u3_id))],
        clean_name_embedding
    ]

    clean_s_name_entities = [
        clean_name_s_id_type,
        clean_s_u3_id,
        clean_name_strict,
        ['clean_s' for i in range(len(u3_id))],
        clean_s_name_embedding
    ]

    com_name_entities = [
        com_name_id_type,
        com_id,
        com_name,
        ['common' for i in range(len(u3_id))],
        com_name_embedding
    ]

    tic_entities = [
        tic_name_id_type,
        tic_id,
        tic_name,
        ['ticker' for i in range(len(u3_id))],
        tic_name_embedding
    ]

    return full_name_entities, clean_name_entities, clean_s_name_entities, com_name_entities, tic_entities



def creaet_NER_Mapping(embeddings):

    fields = [
        FieldSchema(name="u3id_type", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="u3_id", dtype=DataType.DOUBLE, auto_id=False, max_length=100),
        FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="Type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=4096)
    ]
    vdb_name = "NER_Mapping"
    col_ = milvusdb._connect_vdb(vdb_name,
                                 description='NER_Mapping_Companies',
                                 fields=fields)

    full_name_entities, clean_name_entities, clean_s_name_entities, com_name_entities, tic_entities = prepare_data(embeddings=embeddings)

    batch_insert(col_, full_name_entities)
    batch_insert(col_, clean_name_entities)
    batch_insert(col_, clean_s_name_entities)
    batch_insert(col_, com_name_entities)
    batch_insert(col_, tic_entities)

    milvusdb._add_index(vdb_name, pk_name='u3id_type')


if __name__ == '__main__':

    milvusdb = MilvusDB()
    ## Model Config

    embeddings = NVEmbed(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
        #    multi_process=True,
        query_instruction=query_prefix
    )
    embeddings.client.max_seq_length = 4096
    embeddings.client.tokenizer.padding_side = "right"
    embeddings.eos_token = embeddings.client.tokenizer.eos_token
    EMBEDDING_DIMENSION = 4096

    creaet_NER_Mapping(embeddings)

    ## Querying
    vector_to_search = embeddings.embed_query('Tesla Inc')

    client = milvusdb.client
    start_time = time.time()
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 256}, ## set as 4 * sqrt(n) n is the len of nlist
    }
    result = client.search(
        collection_name="NER_Mapping",
        data=[vector_to_search],
        anns_field="embeddings",
        search_params=search_params,
        limit=30,
        output_fields=["u3_id", "company_name", 'Type']
    )
    end_time = time.time()


