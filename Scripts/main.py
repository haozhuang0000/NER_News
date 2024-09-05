from NER.ner import NER_TextProcessor
from Mapping.generate_mapping_company import SimilarityMapping
from Mapping.ner_output_processor import NerOutputProcessor
import argparse
import os
import warnings
from huggingface_hub import login
from Mongodb.mongodb import MongoDBHandler

def set_arguments():

    parser = argparse.ArgumentParser()
    ## ------------------------------- 1. NER ------------------------------- ##
    parser.add_argument("--ner_input",
                        default="News",
                        type=str,
                        help="Specify input ner collection.")
    parser.add_argument('--ner_output',
                        default="ner_out",
                        type=str,
                        help="Specify output ner collection")
    ## ------------------------------- 2. Sentence Split ------------------------------- ##
    parser.add_argument("--sentence_input",
                        default="ner_out",
                        type=str,
                        help="Specify input ner collection.")
    parser.add_argument("--sentence_output",
                        default="sentence_split",
                        type=str,
                        help="Specify output sentence_split collection.")

    ## ------------------------------- 3. Similarity Calculation ------------------------------- ##

    parser.add_argument("--selected_sentence_input",
                        default="sentence_split",
                        type=str,
                        help="Specify sentence collection.")
    parser.add_argument("--selected_sentence_output",
                        default="selected_sentence",
                        type=str,
                        help="Specify output selected sentence collection.")
    parser.add_argument("--embedding_method",
                        default=None,
                        type=str,
                        help="Specify mode of embedding model")
    args, unknown = parser.parse_known_args()
    return args

def main():

    args = set_arguments()

    ## ------------------------------- 1. Raw data ------------------------------- ##

    mongodb_handler = MongoDBHandler()
    mongodb_handler.insert_raw_data()

    ## ------------------------------- 2. NER ------------------------------- ##

    ner_rawdata_processor = NER_TextProcessor(in_col=args.ner_input, out_col=args.ner_output)
    ner_rawdata_processor.run()

    ## ------------------------------- 3. Sentence Split ------------------------------- ##

    nerout_processor = NerOutputProcessor(in_col=args.sentence_input, out_col=args.sentence_output)
    nerout_processor.run()

    ## ------------------------------- 4. Similarity Calculation ------------------------------- ##

    if args.embedding_method == 'Local':
        from VDB_Similarity_Search.Model import NVEmbed
        from Config.config import model_name, model_kwargs, encode_kwargs, query_prefix
        warnings.warn('It is highly recommended to host your embedding model on a GPU server. For guidance, '
                 'please refer to this:  https://github.com/haozhuang0000/RESTAPI_Docker')
        login(os.environ['HUGGINGFACE_TOKEN'])

        embeddings = NVEmbed(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True,
            query_instruction=query_prefix
        )
        embeddings.client.max_seq_length = 4096
        embeddings.client.tokenizer.padding_side = "right"
        embeddings.eos_token = embeddings.client.tokenizer.eos_token

    elif args.embedding_method == 'Server':
        embeddings = os.environ['EMBEDDING_API']
    else:
        raise ValueError(
            'Please set argument embedding_method. It must be either a Local or Server')

    similarity_map = SimilarityMapping(in_col=args.selected_sentence_input, out_col=args.selected_sentence_output, embeddings=embeddings)
    similarity_map.run()


if __name__ == '__main__':

    main()