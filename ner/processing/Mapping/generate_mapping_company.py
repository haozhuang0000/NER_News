"""
This script serves for similarity mapping.

for example:
    1. We have `AAPL` recognized by NER.
    2. we want to map `AAPL` to `Apple Inc`, and give a specific ID.
"""

from joblib import Parallel, delayed, parallel_backend
from logger import Log
from tqdm import tqdm
import Levenshtein
import argparse
import copy
import warnings
from huggingface_hub import login
from ner.db import database
from ner.processing.Mapping.helper_functions import *
from ner.processing.VDB_Similarity_Search.Model import NVEmbed
from ner.processing.VDB_Similarity_Search.VDB_Common import MilvusDB
import os


class SimilarityMapping(database, MilvusDB):

    def __init__(
        self,
        in_col: str,
        out_col: str,
        ner_mapped_col: str = "ner_mapped",
        inserted_threshold: int = 1000,
        embeddings: NVEmbed | str | None = None,
    ) -> None:
        """
        Initializes the NER output processor with the specified input and output collections,
        and an optional NER mapping object.

        Parameters:
        - in_col (str): Name of the input collection.
        - out_col (str): Name of the output collection.
        - ner_mapped_col (str): Name of the ner mapped collection. This collection store the historical mapped companies

        Functionality:
        - Entity Names Loading: Loads combined entity names (companies and economics) using the get_entity_names function,
                                    which reads from specified CSV files based on environment variables.

        """
        super().__init__()
        self.db = database
        self.in_col = self.db[in_col]
        self.out_col = self.db[out_col]
        self.ner_mapping_col = self.db[ner_mapped_col]
        self.embeddings = embeddings
        self.inserted_threshold = inserted_threshold
        self.logger = Log(f"{os.path.basename(__file__)}").getlog()

        self.combined_full_comp_list, self.econ_list, self.econ_id = get_entity_names(
            company_list_name=os.environ["COMPANY_PATH_NAME"],
            econ_list_name=os.environ["ECON_PATH_NAME"],
            logger=self.logger,
        )
        self.full_name_dict = {
            number: name for name, number in self.combined_full_comp_list
        }

    def transform_data_to_results_dict(self, find_cursor: list) -> dict:
        """
        Transforms MongoDB cursor output into a dictionary of results.

        Parameters:
        - find_cursor (list): A list of documents returned by MongoDB.

        Returns:
        - dict: A dictionary where each key is a unique identifier (e.g., _id)
                and the value is the corresponding document.
        """
        results_dict = {
            "_id": [],
            "Sentence_id": [],
            "Output_sentence1": [],
            "Output_sentence2": [],
            "Companies_econs": [],
            "Date": [],
            "Title": [],
        }
        for result in tqdm(find_cursor):
            results_dict["_id"].append(result["_id"])
            results_dict["Sentence_id"].append(result["Sentence_id"])
            results_dict["Output_sentence1"].append(result["Sentence_1"])
            results_dict["Output_sentence2"].append(result["Sentence_2"])
            results_dict["Companies_econs"].append(result["Companies_econs"])
            results_dict["Date"].append(result["Date"])
            results_dict["Title"].append(result["Title"])
        return results_dict

    def get_sentence_split_data(self) -> list:
        """
        Retrieves Sentence data from collection `sentence_split`.

        Returns:
        - list: A list of data
        """
        input_col = self.in_col
        check_col = self.out_col

        input_col_ids = input_col.find({})
        check_col_ids = check_col.find({})
        input_col_ids_list = []
        check_col_ids_list = []

        for item in tqdm(input_col_ids):
            input_col_ids_list.append(item["_id"])
        for item in tqdm(check_col_ids):
            check_col_ids_list.append(item["_id"])
        clear_input_list = list(
            set(input_col_ids_list).difference(set(check_col_ids_list))
        )
        results_dict = {
            "_id": [],
            "Sentence_id": [],
            "Output_sentence1": [],
            "Output_sentence2": [],
            "Companies_econs": [],
            "Date": [],
            "Title": [],
        }

        ## define the result dict
        if len(clear_input_list) < 5e5:
            find_cursor = input_col.find({"_id": {"$in": clear_input_list}})
            results_dict = self.transform_data_to_results_dict(find_cursor)
        else:
            n_cores = 20  # number of splits
            total_size = len(clear_input_list)
            batch_size = round(total_size / n_cores + 0.5)
            skips = range(0, n_cores * batch_size, batch_size)
            for skip_n in tqdm(skips):
                find_cursor = input_col.find(
                    {"_id": {"$in": clear_input_list[skip_n : skip_n + batch_size]}}
                )
                temp_dict = self.transform_data_to_results_dict(find_cursor)
                for key in list(temp_dict.keys()):
                    results_dict[key].extend(temp_dict[key])

        return list(pd.DataFrame(results_dict).to_dict("index").values())

    def calculate_similarity_helper(
        self,
        input_article: dict,
        ner_comp: str,
        ner_comp_original: str,
        lower_bound: int,
        _id: str,
        check: str,
        result_list: list,
    ) -> tuple[list, bool]:
        """
        Calculates similarity between a named entity (NER) component and mapped entities from a list provided.

        Parameters:
        - input_article (dict): The dictionary representing the article being processed, where similarities and entity details will be added.
        - ner_comp (str): The named entity component to compare against the mapped entities.
        - ner_comp_original (str): A original entity name recognized by NER for additional comparison.
        - lower_bound (int): The minimum similarity threshold for considering a match.
        - _id (str): The identifier for the current entity mapping (id in ner_mapped).
        - check (str): The type of name comparison to perform (e.g., 'Full_Name', 'Common_Name').
        - result_list (list): A list of possible matching entities retrieved from the database.

        Returns:
        - tuple[list, bool]: A tuple where the first element is the updated `input_article` list containing the similarity and entity details,
                             and the second element is a boolean flag indicating whether a match was found (False if a match was found, True otherwise).

        The function performs the following tasks:
        1. Searches for a direct mapping of the named entity in the database.
        2. If found, updates the `input_article` with relevant entity details and marks the match as found.
        3. If not found, iterates through the `result_list` to compare the `ner_comp` against each possible entity.
        4. If a sufficient similarity is found, it performs a second comparison using `ner_comp_original`.
        5. Updates the `input_article` with similarity scores and matched entity details.
        6. Inserts the mapping into the database if a new match is found.
        """

        def _compute_similarity_for_each_mode(
            ner_comp: str, check: str, name_to_map: str, name_to_map_type: str
        ) -> int | None:

            checker = {
                "Full_Name": "full",
                "Common_Name": "common",
                "Clean_Name": "clean",
                "Clean_Name_S": "clean_s",
            }
            check_type = checker[check]
            if check_type == name_to_map_type or name_to_map_type == "ticker":
                if first_word(ner_comp.lower()) != first_word(name_to_map.lower()):
                    return None
                else:
                    similarity = Levenshtein.ratio(
                        ner_comp.lower(), name_to_map.lower()
                    )
                    return similarity
            else:
                return None

        Similarity_1st = []
        Similarity_2nd = []
        cleaned_ner_entity_1st = []
        matched_cleaned_comp_1st = []
        ner_entity_2nd = []
        matched_comp_2nd = []
        flag = True
        ## Find Mapping from Database 'NER_Mapping'
        mapped_NER_Name = [i for i in self.ner_mapping_col.find({"NER_Name": ner_comp})]
        if mapped_NER_Name != []:
            mapped_NER_Name = mapped_NER_Name[0]

            _id = mapped_NER_Name["_id"]
            match_u3id = mapped_NER_Name["U3_Company_Number"]
            match_company = mapped_NER_Name["Company_Name"]
            match_similarity = mapped_NER_Name["Similarity"]
            Similarity_1st = mapped_NER_Name["Similarity_1st"]
            Similarity_2nd = mapped_NER_Name["Similarity_2nd"]
            cleaned_ner_entity_1st = mapped_NER_Name["1st_cleaned_ner_entity"]
            matched_cleaned_comp_1st = mapped_NER_Name["1st_matched_cleaned_comp"]
            ner_entity_2nd = mapped_NER_Name["2nd_ner_entity"]
            matched_comp_2nd = mapped_NER_Name["2nd_matched_comp"]

            input_article["Entity_id"].append(int(match_u3id))
            input_article["Bingo_entity"].append(match_company)
            input_article["Similarity"].append(match_similarity)
            input_article["Similarity_1st"].append(Similarity_1st)
            input_article["Similarity_2nd"].append(Similarity_2nd)
            input_article["1st_cleaned_ner_entity"].append(cleaned_ner_entity_1st)
            input_article["1st_matched_cleaned_comp"].append(matched_cleaned_comp_1st)
            input_article["2nd_ner_entity"].append(ner_entity_2nd)
            input_article["2nd_matched_comp"].append(matched_comp_2nd)
            input_article["NER_Mapping_ID"].append(_id)
            flag = False

        else:
            target_indexs = []
            for idx in range(0, len(result_list)):
                # print(comp_list[i])
                # print("here:"+comp_list[i])
                name_to_map = result_list[idx]["company_name"]
                name_to_map_type = result_list[idx]["Type"]

                similarity = _compute_similarity_for_each_mode(
                    ner_comp, check, name_to_map, name_to_map_type
                )
                if similarity is None:
                    continue

                if similarity >= lower_bound:
                    Similarity_1st.append(similarity)
                    cleaned_ner_entity_1st.append(ner_comp)
                    matched_cleaned_comp_1st.append(
                        self.full_name_dict[result_list[idx]["u3_id"]]
                    )
                    target_indexs.append(idx)
                    flag = False
            if flag == True:
                pass
            else:
                max_similarity = 0.5
                match_company = ""
                match_u3id = -1

                for idx in target_indexs:  # second comparison
                    company = self.full_name_dict[result_list[idx]["u3_id"]]
                    u3id = result_list[idx]["u3_id"]

                    sim = Levenshtein.ratio(ner_comp_original.lower(), company.lower())
                    Similarity_2nd.append(sim)
                    ner_entity_2nd.append(ner_comp_original)
                    matched_comp_2nd.append(company)
                    if sim >= max_similarity:
                        max_similarity = sim
                        match_company = company
                        match_u3id = u3id

                if match_company != "":
                    match_similarity = max_similarity
                else:
                    match_similarity = -1
                input_article["Entity_id"].append(int(match_u3id))
                input_article["Bingo_entity"].append(match_company)
                input_article["Similarity"].append(match_similarity)
                input_article["Similarity_1st"].append(Similarity_1st)
                input_article["Similarity_2nd"].append(Similarity_2nd)
                input_article["1st_cleaned_ner_entity"].append(cleaned_ner_entity_1st)
                input_article["1st_matched_cleaned_comp"].append(
                    matched_cleaned_comp_1st
                )
                input_article["2nd_ner_entity"].append(ner_entity_2nd)
                input_article["2nd_matched_comp"].append(matched_comp_2nd)
                input_article["NER_Mapping_ID"].append(_id)

                ## Insert into NER_Mapping
                if flag == False:
                    insert_mapped_dict(
                        col=self.ner_mapping_col,
                        match_u3id=match_u3id,
                        match_company=match_company,
                        ner_comp_similarity2=ner_comp_original,
                        ner_comp=ner_comp,
                        match_similarity=match_similarity,
                        Similarity_1st=Similarity_1st,
                        Similarity_2nd=Similarity_2nd,
                        cleaned_ner_entity_1st=cleaned_ner_entity_1st,
                        matched_cleaned_comp_1st=matched_cleaned_comp_1st,
                        ner_entity_2nd=ner_entity_2nd,
                        matched_comp_2nd=matched_comp_2nd,
                    )

        return input_article, flag

    def calculate_similarity(self, input_article: dict, lower_bound: int) -> dict:
        """
        Calculates similarity between company names in the input article and entities in a vector database or a list of economic terms.

        This method updates the `input_article` dictionary with detailed similarity metrics, entity mappings, and other relevant data based on several comparison checks.

        Steps:
            1. Extract the top 30 most similar vectors from the vector database for each company entity recognized in the article.
            2. Perform several stages of similarity comparisons:
               - Compare full names from the article with full names (top 30 similar term from vector database).
               - Compare common names from the article with common names (top 30 similar term from vector database).
               - Compare cleaned names from the article with cleaned names (top 30 similar term from vector database).
                 - Clean company names by removing length-1 suffixes and full suffixes, and recheck similarity.
            3. For entries identified as economic terms, use Levenshtein ratio to find the closest matches in the economic term list.
            4. Update the `input_article` with similarity results, including entity IDs, similarity scores, and mappings.


        Parameters:
        - input_article (dict): A dictionary containing details about the article.
        Expected keys include:
            - 'Companies_econs': List of company names and country names extracted from
            the article.
            - others
        - lower_bound (int): A threshold value used to filter out matches based on
        similarity score.

        Returns:
        - dict: The updated `input_article` dictionary with additional fields:
            - 'Entity_id': List of entity IDs corresponding to matched companies.
            - 'Bingo_entity': List of company names that matched.
            - 'Similarity': List of similarity scores for each company name.
            - 'Similarity_1st': List of similarity scores for the first comparison stage.
            - 'Similarity_2nd': List of similarity scores for the second comparison stage.
            - '1st_cleaned_ner_entity': List of cleaned entity names from the first stage. (clean suffix for recognized name in the article)
            - '1st_matched_cleaned_comp': List of matched cleaned company names from the first stage.
            - '2nd_ner_entity': List of cleaned entity names from the second stage.
            - '2nd_matched_comp': List of matched cleaned company names from the second stage.
            - 'NER_Mapping_ID': List of unique IDs for the NER mapping.

        Notes:
        - The method performs comparisons using full names, common names, and cleaned names.
        - It also handles special cases for economic terms and applies various cleaning methods to company names.
        - Helper functions such as `calculate_similarity_helper`, `clean_string_len1`, and `clean_string` are used for detailed processing.

        """
        input_article["Entity_id"] = []
        input_article["Bingo_entity"] = []
        input_article["Similarity"] = []
        input_article["Similarity_1st"] = []
        input_article["Similarity_2nd"] = []
        input_article["1st_cleaned_ner_entity"] = []
        input_article["1st_matched_cleaned_comp"] = []
        input_article["2nd_ner_entity"] = []
        input_article["2nd_matched_comp"] = []
        input_article["NER_Mapping_ID"] = []
        input_article["Updated"] = 0

        ner_comp_list = rm_none_entity(input_article["Companies_econs"])

        for q in range(len(input_article["Companies_econs"])):
            if len(input_article["Output_sentence2"][q]) != 0:
                if input_article["Output_sentence2"][q][8] == "C":
                    ner_comp = input_article["Companies_econs"][q]
                    ner_comp_similarity2 = copy.deepcopy(ner_comp)
                    _id = create_id(ner_comp_similarity2)

                    ## Search VDB base on non-cleaned ner_comp
                    results = self.vectorsearch(
                        ner_company_name=ner_comp, embeddings=self.embeddings
                    )
                    result_list_full = []
                    for i in results[0]:
                        if i["distance"] > 1:
                            result_list_full.append(i["entity"])

                    ########### ------------------ 1. Comparing company name in the article with full name in the list ------------------ #############
                    input_article, flag = self.calculate_similarity_helper(
                        input_article,
                        ner_comp,
                        ner_comp_similarity2,
                        lower_bound,
                        _id,
                        check="Full_Name",
                        result_list=result_list_full,
                    )
                    if flag == True:
                        ########### ------------------ 2. Comparing company name in the article with common name in the list ------------------ #############
                        input_article, flag = self.calculate_similarity_helper(
                            input_article,
                            ner_comp,
                            ner_comp_similarity2,
                            lower_bound,
                            _id,
                            check="Common_Name",
                            result_list=result_list_full,
                        )

                        if flag == True:
                            ########### ------------------ 3 Comparing company name in the article with clean name in the list ------------------ #############
                            input_article, flag = self.calculate_similarity_helper(
                                input_article,
                                ner_comp,
                                ner_comp_similarity2,
                                lower_bound,
                                _id,
                                check="Clean_Name",
                                result_list=result_list_full,
                            )
                            if flag == True:
                                ########### ------------------ 4 change comp to their full name from abbreviation/short name ------------------ #############
                                if len(ner_comp_list) > 0:
                                    for icomp in ner_comp_list:
                                        if is_shortName(ner_comp, icomp):
                                            ner_comp = icomp
                                        if is_abbrev(ner_comp, icomp):
                                            ner_comp = icomp
                                if ner_comp not in ner_comp_list:
                                    ner_comp_list.append(ner_comp)

                                ########### ------------------ 4.1 clean with length 1 suffix------------------ #############
                                cleaned_len1_ner_entity = clean_string_len1(ner_comp)

                                if len(cleaned_len1_ner_entity) == 0:
                                    input_article["Entity_id"].append(int("-1"))
                                    input_article["Similarity"].append(-1)
                                    input_article["Bingo_entity"].append("")
                                    input_article["Similarity_1st"].append([])
                                    input_article["Similarity_2nd"].append([])
                                    input_article["1st_cleaned_ner_entity"].append([])
                                    input_article["1st_matched_cleaned_comp"].append([])
                                    input_article["2nd_ner_entity"].append([])
                                    input_article["2nd_matched_comp"].append([])
                                    input_article["NER_Mapping_ID"].append(_id)

                                    insert_mapped_dict(
                                        col=self.ner_mapping_col,
                                        match_u3id=int("-1"),
                                        match_company="",
                                        ner_comp_similarity2=ner_comp,
                                        ner_comp=ner_comp,
                                        match_similarity=-1,
                                        Similarity_1st=[],
                                        Similarity_2nd=[],
                                        cleaned_ner_entity_1st=[],
                                        matched_cleaned_comp_1st=[],
                                        ner_entity_2nd=[],
                                        matched_comp_2nd=[],
                                    )
                                    continue

                                ## Search VDB base on cleaned length 1 suffix
                                results = self.vectorsearch(
                                    ner_company_name=cleaned_len1_ner_entity,
                                    embeddings=self.embeddings,
                                )
                                result_list_clean = []
                                for i in results[0]:
                                    if i["distance"] > 1:
                                        result_list_clean.append(i["entity"])
                                ########### ------------------ 4.2 Comparing company name that has been cleaned for len 1 suffix in the article with clean name in the list ------------------ #############
                                input_article, flag = self.calculate_similarity_helper(
                                    input_article,
                                    cleaned_len1_ner_entity,
                                    ner_comp_similarity2,
                                    lower_bound,
                                    _id,
                                    check="Clean_Name",
                                    result_list=result_list_clean,
                                )
                                if flag == True:
                                    ########### ------------------ 5.1 clean with full suffix ------------------ #############
                                    cleaned_ner_entity = clean_string(
                                        cleaned_len1_ner_entity
                                    )
                                    if len(cleaned_ner_entity) == 0:
                                        input_article["Entity_id"].append(int("-1"))
                                        input_article["Similarity"].append(-1)
                                        input_article["Bingo_entity"].append("")
                                        input_article["Similarity_1st"].append([])
                                        input_article["Similarity_2nd"].append([])
                                        input_article["1st_cleaned_ner_entity"].append(
                                            []
                                        )
                                        input_article[
                                            "1st_matched_cleaned_comp"
                                        ].append([])
                                        input_article["2nd_ner_entity"].append([])
                                        input_article["2nd_matched_comp"].append([])
                                        input_article["NER_Mapping_ID"].append(_id)

                                        insert_mapped_dict(
                                            col=self.ner_mapping_col,
                                            match_u3id=int("-1"),
                                            match_company="",
                                            ner_comp_similarity2=ner_comp,
                                            ner_comp=ner_comp,
                                            match_similarity=-1,
                                            Similarity_1st=[],
                                            Similarity_2nd=[],
                                            cleaned_ner_entity_1st=[],
                                            matched_cleaned_comp_1st=[],
                                            ner_entity_2nd=[],
                                            matched_comp_2nd=[],
                                        )
                                        continue
                                    ## Search VDB base on cleaned full length suffix
                                    results = self.vectorsearch(
                                        ner_company_name=cleaned_ner_entity,
                                        embeddings=self.embeddings,
                                    )
                                    result_list_clean_s = []
                                    for i in results[0]:
                                        if i["distance"] > 1:
                                            result_list_clean_s.append(i["entity"])

                                    ########### ------------------ 5.2 Comparing company name that has been cleaned for full suffix in the article with clean name in the list ------------------ #############
                                    input_article, flag = (
                                        self.calculate_similarity_helper(
                                            input_article,
                                            cleaned_ner_entity,
                                            ner_comp_similarity2,
                                            lower_bound,
                                            _id,
                                            check="Clean_Name_S",
                                            result_list=result_list_clean_s,
                                        )
                                    )

                                    if flag == True:
                                        input_article["Entity_id"].append(int("-1"))
                                        input_article["Similarity"].append(-1)
                                        input_article["Bingo_entity"].append("")
                                        input_article["Similarity_1st"].append([])
                                        input_article["Similarity_2nd"].append([])
                                        input_article["1st_cleaned_ner_entity"].append(
                                            []
                                        )
                                        input_article[
                                            "1st_matched_cleaned_comp"
                                        ].append([])
                                        input_article["2nd_ner_entity"].append([])
                                        input_article["2nd_matched_comp"].append([])
                                        input_article["NER_Mapping_ID"].append(_id)

                                        insert_mapped_dict(
                                            col=self.ner_mapping_col,
                                            match_u3id=int("-1"),
                                            match_company="",
                                            ner_comp_similarity2=ner_comp,
                                            ner_comp=ner_comp,
                                            match_similarity=-1,
                                            Similarity_1st=[],
                                            Similarity_2nd=[],
                                            cleaned_ner_entity_1st=[],
                                            matched_cleaned_comp_1st=[],
                                            ner_entity_2nd=[],
                                            matched_comp_2nd=[],
                                        )
                                        continue

                if input_article["Output_sentence2"][q][8] == "E":
                    flag = True
                    for idx in range(len(self.econ_list)):
                        similarity = Levenshtein.ratio(
                            input_article["Companies_econs"][q], self.econ_list[idx]
                        )
                        if similarity > lower_bound:
                            input_article["Entity_id"].append(
                                int(int(self.econ_id[idx]))
                            )
                            input_article["Bingo_entity"].append(self.econ_list[idx])
                            input_article["Similarity"].append(similarity)
                            input_article["Similarity_1st"].append([])
                            input_article["Similarity_2nd"].append([])
                            input_article["1st_cleaned_ner_entity"].append([])
                            input_article["1st_matched_cleaned_comp"].append([])
                            input_article["2nd_ner_entity"].append([])
                            input_article["2nd_matched_comp"].append([])
                            input_article["NER_Mapping_ID"].append("")
                            flag = False
                            break
                    if flag:
                        input_article["Entity_id"].append(int("-1"))
                        input_article["Similarity"].append(-1)
                        input_article["Bingo_entity"].append("")
                        input_article["Similarity_1st"].append([])
                        input_article["Similarity_2nd"].append([])
                        input_article["1st_cleaned_ner_entity"].append([])
                        input_article["1st_matched_cleaned_comp"].append([])
                        input_article["2nd_ner_entity"].append([])
                        input_article["2nd_matched_comp"].append([])
                        input_article["NER_Mapping_ID"].append("")
            else:
                input_article["Entity_id"].append(int("-3"))
                input_article["Similarity"].append(-1)
                input_article["Bingo_entity"].append("")
                input_article["Similarity_1st"].append([])
                input_article["Similarity_2nd"].append([])
                input_article["1st_cleaned_ner_entity"].append([])
                input_article["1st_matched_cleaned_comp"].append([])
                input_article["2nd_ner_entity"].append([])
                input_article["2nd_matched_comp"].append([])
                input_article["NER_Mapping_ID"].append("")
        return input_article

    def batch_helper(self, batch: list) -> int:
        """
        Processes a batch of data, generates sentence information, and inserts it into a MongoDB collection.

        Parameters:
        - batch (list): A list of dictionaries, where each dictionary contains data needed for sentence generation.

        Returns:
        - int: The number of documents successfully inserted into the MongoDB collection.
        """
        sentence_col = self.out_col
        sentences_divided_by = []
        inserted_num = 0
        inserted_per_round = 0

        # try:
        for pair in tqdm(batch):
            sub_result = self.calculate_similarity(pair, 0.9)
            if sub_result != None:
                sentences_divided_by.append(sub_result)
                inserted_per_round += 1
                if len(sentences_divided_by) >= self.inserted_threshold:
                    inserted_num = inserted_num + len(sentences_divided_by)
                    sentence_col.insert_many(sentences_divided_by)
                    sentences_divided_by = []
                    inserted_per_round = 0
        if inserted_per_round > 0:
            inserted_num = inserted_num + inserted_per_round
            sentence_col.insert_many(sentences_divided_by)

        return inserted_num

    def run(self):

        start_time = time.time()
        self.logger.info("loading data from mongodb...")
        list_of_dict = self.get_sentence_split_data()
        self.logger.info("the list of dict size is {}".format(len(list_of_dict)))
        self.logger.info(
            "load data from mongodb time: {}".format(time.time() - start_time)
        )

        # process
        if len(list_of_dict) == 0:
            self.logger.info("No new data")
        else:
            start_time = time.time()
            self.logger.info("processing data...")
            with parallel_backend("threading", n_jobs=3):
                parallel_results = Parallel(n_jobs=1)(
                    delayed(self.batch_helper)(batch)
                    for batch in split_iter(list_of_dict, 20)
                )
            self.logger.info(
                "{} sentences added to selected_sentences database".format(
                    sum(parallel_results)
                )
            )
            # [self.batch_helper(batch) for batch in split_iter(list_of_dict, 5)]

            self.logger.info("process time: {}".format(time.time() - start_time))
        self.logger.info("all done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_col",
        default="sentence_split",
        type=str,
        help="Specify sentence collection.",
    )
    parser.add_argument(
        "--out_col",
        default="selected_sentence",
        type=str,
        help="Specify output selected sentence collection.",
    )
    parser.add_argument(
        "--embedding_method",
        default=None,
        type=str,
        help="Specify mode of embedding model",
    )
    args, unknown = parser.parse_known_args()

    if args.embedding_method == "Local":
        from ner.processing.Model_Config.config import (
            MODEL_NAME,
            MODEL_KWARGS,
            ENCODE_KWARGS,
            QUERY_PREFIX,
        )

        warnings(
            "It is highly recommended to host your embedding model on a server. For guidance, please refer to this:  https://github.com/haozhuang0000/RESTAPI_Docker"
        )
        login(os.environ["HUGGINGFACE_TOKEN"])

        embeddings = NVEmbed(
            model_name=MODEL_NAME,
            model_kwargs=MODEL_KWARGS,
            encode_kwargs=ENCODE_KWARGS,
            show_progress=True,
            #    multi_process=True,
            query_instruction=QUERY_PREFIX,
        )
        embeddings.client.max_seq_length = 4096
        embeddings.client.tokenizer.padding_side = "right"
        embeddings.eos_token = embeddings.client.tokenizer.eos_token
        EMBEDDING_DIMENSION = 4096
    elif args.embedding_method == "Server":
        embeddings = "http://10.230.252.6:7777/api/NVEmbed"
    else:
        raise ValueError(
            "Please set argument embedding_method. It must be either a Local or Server"
        )
    similarity_map = SimilarityMapping(
        in_col=args.in_col, out_col=args.out_col, embeddings=embeddings
    )
    similarity_map.run()
