import spacy
import os


class NER_Ruler:
    """
    A class to enhance a spaCy NLP model with
    custom Named Entity Recognition (NER) rules.
    """

    def __init__(self):
        """
        Initializes the NER_Ruler with a specified ruler path.

        :param ruler_path: Path to the directory containing the patterns.jsonl file.
                           Defaults to environment variable 'RULER_PATH'.
        """
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        model_path = os.path.join(base_dir, "nlp/final_v2")
        self.nlp_model = spacy.load(model_path)

    def get_model(self) -> spacy.Language:
        """
        Returns the spaCy NLP model with the custom NER ruler.

        :return: The spaCy NLP model.
        """
        return self.nlp_model
