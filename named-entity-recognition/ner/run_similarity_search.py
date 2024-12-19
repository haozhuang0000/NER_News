import os
import warnings
from huggingface_hub import login
from ner.config import (
    DEFAULT_SELECTED_SENTENCE_COLLECTION,
    DEFAULT_SENTENCE_SPLIT_COLLECTION,
    EMBEDDING_METHOD,
)
from ner.processing.Mapping.generate_mapping_company import (
    SimilarityMapping,
)

if EMBEDDING_METHOD == "Local":
    from ner.processing.VDB_Similarity_Search.Model import (
        NVEmbed,
    )
    from ner.processing.Model_Config.config import (
        MODEL_NAME,
        MODEL_KWARGS,
        ENCODE_KWARGS,
        QUERY_PREFIX,
    )

    warnings.warn(
        "It is highly recommended to host your embedding model on a GPU"
        "server. For guidance, please refer to this: "
        "https://github.com/haozhuang0000/RESTAPI_Docker"
    )
    login(os.environ["HUGGINGFACE_TOKEN"])

    embeddings = NVEmbed(
        model_name=MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
        show_progress=True,
        query_instruction=QUERY_PREFIX,
    )
    embeddings.client.max_seq_length = 4096
    embeddings.client.tokenizer.padding_side = "right"
    embeddings.eos_token = embeddings.client.tokenizer.eos_token

elif EMBEDDING_METHOD == "Server":
    embeddings = os.environ["EMBEDDING_API"]
else:
    raise ValueError(f"Embedding Method {EMBEDDING_METHOD} not of Local or Server.")


similarity_map = SimilarityMapping(
    in_col=DEFAULT_SENTENCE_SPLIT_COLLECTION,
    out_col=DEFAULT_SELECTED_SENTENCE_COLLECTION,
    embeddings=embeddings,
)
similarity_map.run()
