model_name = r'D:\MyGithub\NV-Embed-v1'

batch_size = 8
model_kwargs = {'device': 'cuda', "trust_remote_code": True, "model_kwargs": {"torch_dtype": torch.bfloat16}}
# need to update encode kwargs with prompt when embedding the query
encode_kwargs = {"batch_size": batch_size, 'normalize_embeddings': True}

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"default": "Given a question, retrieve passages that answer the question", }
query_prefix = "Instruct: " + task_name_to_instruct["default"] + "\nQuery: "