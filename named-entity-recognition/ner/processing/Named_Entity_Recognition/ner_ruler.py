import json
import os
import spacy


class NER_Ruler:
    """
    A class to enhance a spaCy NLP model with
    custom Named Entity Recognition (NER) rules.
    """

    def __init__(self, ruler_path: str = None):
        """
        Initializes the NER_Ruler with a specified ruler path.

        :param ruler_path: Path to the directory containing the patterns.jsonl file.
                           Defaults to environment variable 'RULER_PATH'.
        """
        self.ruler_path = ruler_path or os.getenv("RULER_PATH")
        if not self.ruler_path:
            raise ValueError(
                "RULER_PATH must be specified either as a "
                "parameter or as an environment variable."
            )

        self.nlp_model = spacy.load("en_core_web_sm")
        # self._add_ruler()

    def _load_patterns(self, file_path: str) -> list[dict]:
        """
        Loads patterns from a JSONL file.

        :param file_path: Path to the JSONL file containing patterns.
        :return: A list of patterns.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [json.loads(line) for line in file]
        except Exception as e:
            raise RuntimeError(f"Failed to load patterns from {file_path}: {e}")

    def _add_ruler(self):
        """
        Adds a custom entity ruler to the NLP model using
        patterns defined in the specified JSONL file.
        """
        ruler = self.nlp_model.add_pipe("entity_ruler", before="ner")
        patterns_path = os.path.join(self.ruler_path, "patterns.jsonl")
        patterns = self._load_patterns(patterns_path)
        ruler.add_patterns(patterns)

    def get_model(self) -> spacy.Language:
        """
        Returns the spaCy NLP model with the custom NER ruler.

        :return: The spaCy NLP model.
        """
        return self.nlp_model
