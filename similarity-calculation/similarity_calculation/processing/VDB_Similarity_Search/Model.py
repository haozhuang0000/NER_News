from langchain_community.embeddings import HuggingFaceEmbeddings
import sentence_transformers


class NVEmbed(HuggingFaceEmbeddings):
    eos_token: str | None = None
    """End of sentence token to use."""
    query_instruction: str | None = ""
    embed_instruction: str | None = ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Compute document embeddings using a HuggingFace transformer model.

        @param texts (list[str]): The list of texts to embed.

        @return list - A list of embeddings, one for each text.
        """
        texts = [self.embed_instruction + text + self.eos_token for text in texts]

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(
                texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Compute query embeddings using a HuggingFace transformer model.

        @param text: The text to embed.

        @return Embeddings for the text.
        """
        text = self.query_instruction + text + self.eos_token
        return self.embed_documents([text])[0]
