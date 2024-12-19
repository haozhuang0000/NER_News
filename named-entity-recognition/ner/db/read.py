from pymongo.collection import Collection
from ner.models import Article, NEROut, SentenceSplit


def list_data(cursor: list, type: str):
    if type == "Article":
        return [Article(**result) for result in cursor]
    if type == "NEROut":
        return [NEROut(**result) for result in cursor]
    if type == "SentenceSplit":
        return [SentenceSplit(**result) for result in cursor]
    else:
        raise Exception(f"Collection {type} not found.")


def pull_mongo_data(in_col: Collection, out_col: Collection, type: str):

    source_collection = in_col
    dest_collection = out_col

    pipeline = [{"$group": {"_id": "$_id"}}]
    check_col_ner_ids = list(dest_collection.aggregate(pipeline))
    input_col_ids = list(source_collection.aggregate(pipeline))

    check_collection_list = [item["_id"] for item in check_col_ner_ids]
    input_col_ids_list = [item["_id"] for item in input_col_ids]

    clear_input_list = list(
        set(input_col_ids_list).difference(set(check_collection_list))
    )

    if len(clear_input_list) < 5e5:
        find_cursor = source_collection.find({"_id": {"$in": clear_input_list}})
        return list_data(find_cursor, type)
    else:
        data: list = []
        n_cores = 20  # number of splits
        total_size = len(clear_input_list)
        batch_size = round(total_size / n_cores + 0.5)
        skips = range(0, n_cores * batch_size, batch_size)
        for skip_n in skips:
            find_cursor = source_collection.find(
                {"_id": {"$in": clear_input_list[skip_n : skip_n + batch_size]}}
            )
            data.extend(list_data(find_cursor, type))

        return data
