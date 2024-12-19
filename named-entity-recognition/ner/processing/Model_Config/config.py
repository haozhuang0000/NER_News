import torch

"""
This config file describes model configurations, while the root config
which describes the universal configuration.
"""


MODEL_NAME = "nvidia/NV-Embed-v1"

BATCH_SIZE = 8
MODEL_KWARGS = {
    "device": "cuda",
    "trust_remote_code": True,
    "model_kwargs": {"torch_dtype": torch.bfloat16},
}
# need to update encode kwargs with prompt when embedding the query
ENCODE_KWARGS = {"batch_size": BATCH_SIZE, "normalize_embeddings": True}

# Each query needs to be accompanied by an corresponding instruction
# describing the task.
TASK_NAME_TO_INSTRUCT = {
    "default": "Given a question, retrieve passages that answer the question",
}
QUERY_PREFIX = "Instruct: " + TASK_NAME_TO_INSTRUCT["default"] + "\nQuery: "
