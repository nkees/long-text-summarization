import bertabs.run_summarization as run_summarization
from manage_files import find_all_summaries, find_unprocessed_files, read_file_contents, write_file_contents, delete_temp
from bertabs.run_summarization import parse_args
import logging
import re
import os
import nltk
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn_extra.cluster import KMedoids
from bertabs.MMR import MMR
import numpy as np

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# SETTING THE LOGGER LEVEL
logging.getLogger().setLevel(logging.INFO)

# PATH CONFIGURATION
USER_PATH = re.sub('heuristics.py', '', os.path.realpath(__file__))
USER_PATH = re.sub(r"\\", r"/", USER_PATH)
os.chdir(USER_PATH)

TOKEN_LIMIT = 512


def split_text(text, tokenizer, token_limit) -> list:
    """
    Splits the text into parts shorter than the given token limit, with retaining sentence boundaries.
    Args:
        text: (string) text to tokenize
        tokenizer: tokenizer, e.g. Bert tokenizer
        token_limit: (int) a token limit as indication of max. length of sniplets

    Returns: a list of sniplets shorter than the token limit

    """
    text_sent = nltk.sent_tokenize(text)
    token_limit = token_limit - 2*len(text_sent)
    text_sent_tokenized = [tokenizer.tokenize(elem) for elem in text_sent]
    sent_lengths = [len(elem) for elem in text_sent_tokenized]
    text_length = sum(sent_lengths)
    result = []
    if text_length < token_limit:
        result.append(text)
        return result
    if any(sent_lengths) > token_limit:
        raise IndexError(f"One of the sentences is too long. Mind the following text: "
                         f"{text}")
    size = 0
    result = []
    chunk = ""
    while text_sent:
        new_sent = text_sent.pop(0)
        new_len = sent_lengths.pop(0)
        if size + new_len < token_limit:
            chunk = chunk + new_sent
            size = size + new_len
        else:
            result.append(chunk)
            chunk = "" + new_sent
            size = 0 + new_len
    if chunk:
        result.append(chunk)
    return result


def split_and_summarize_chunks(args, text, id, narrative_storage_path, tokenizer, token_limit):
    """
    Takes the text, splits it into chunks which are shorter than the token limit, stores the chunks in temp,
    produces summaries of the chunks and returns the path where the chunks along with their summaries are stored.
    Args:
        args: arguments specified by the user
        text: the text to be split
        id: id of the summary according to the file name
        narrative_storage_path: the initial path specified by the user where the narratives are stored
        tokenizer: tokenizer for splitting
        token_limit: token limit for splitting

    Returns: path to the chunks and the summaries

    """
    text_split = split_text(text, tokenizer, token_limit)
    logging.info(f"There are {len(text_split)} chunks of text. They will be summarized.")
    temp_path = os.path.join(narrative_storage_path, "temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    counter = 0
    for i in text_split:
        write_file_contents(os.path.join(temp_path, f"narrative_{id}_{counter}_raw.txt"), i)
        counter += 1
    args.documents_dir = temp_path
    args.summaries_output_dir = args.documents_dir
    logging.info(f"New path to the narratives is: {args.documents_dir}")
    run_summarization.main(args)
    logging.info(f"New summaries are: {find_all_summaries(args.documents_dir)}")
    # now the summaries are in the temp under f"narrative_{id}_{counter}_summary.txt
    #args.documents_dir = narrative_storage_path
    return temp_path


def combine_chunks(filename_list, path):
    chunks = []
    for file_name in filename_list:
        chunk = read_file_contents(os.path.join(path, file_name))
        chunks.append(chunk)
    combined_result = " ".join(chunks)
    return combined_result


class Heuristic1:
    """
    --- SumSum ---
    Summarize the chunks of text which are shorter than the token limit and then summarize those chunks again
    to get one representative summary.
    """
    @staticmethod
    def process(args, text, id, narrative_storage_path, tokenizer, token_limit):
        # split text, save to temp, change args path, summarize from the temp
        # for each of the texts: summarize them
        path_to_summaries = split_and_summarize_chunks(args, text, id, narrative_storage_path, tokenizer, token_limit)
        summary_files = find_all_summaries(path_to_summaries)
        summaries_combined = combine_chunks(summary_files, path_to_summaries)
        # for each of the summaries: concatenate, check if less than 510 tokens, otherwise split again, summarize again
        while len(summary_files) > 1:
            path_to_summaries = split_and_summarize_chunks(args, summaries_combined, id, path_to_summaries, tokenizer, token_limit)
            summary_files = find_all_summaries(path_to_summaries)
            summaries_combined = combine_chunks(summary_files, path_to_summaries)
        # return the original path
        args.documents_dir = narrative_storage_path
        # delete the temp
        delete_temp(args.documents_dir)
        # return a summary
        return summaries_combined


def find_cluster_centres(text, num_clusters):
    corpus = nltk.sent_tokenize(text)
    corpus_embeddings = embedder.encode(corpus)
    clustering_model = KMedoids(n_clusters=num_clusters, random_state=0, metric="cosine")
    clustering_model.fit(corpus_embeddings)
    cluster_center_embeddings = clustering_model.cluster_centers_
    cluster_centers = []
    for center_embedding in cluster_center_embeddings:
        for index, sentence_embedding in enumerate(corpus_embeddings):
            if np.array_equal(sentence_embedding, center_embedding):
                if corpus[index] not in cluster_centers:
                    cluster_centers.append(corpus[index])
    return cluster_centers


class Heuristic2:
    """
    --- SumClus#4 ---
    Summarize the chunks of text which are shorter than the token limit and then cluster all the sentences into 4
    clusters and combine their Medoids to get one representative summary.
    """
    @staticmethod
    def process(args, text, id, narrative_storage_path, tokenizer, token_limit):
        # split text, save to temp, change args path, summarize from the temp
        # for each of the texts: summarize them
        path_to_summaries = split_and_summarize_chunks(args, text, id, narrative_storage_path, tokenizer, token_limit)
        summary_files = find_all_summaries(path_to_summaries)
        summaries_combined = combine_chunks(summary_files, path_to_summaries)
        # combine, cluster sentences and find cluster centres
        cluster_centers = find_cluster_centres(summaries_combined, num_clusters=4)
        # return a summary
        summary = " ".join(cluster_centers)
        # return the original path
        args.documents_dir = narrative_storage_path
        # delete the temp
        delete_temp(args.documents_dir)
        return summary


class Heuristic3:
    """
    --- SumMMR ---
    Summarize the chunks of text which are shorter than the token limit and then Select 4 key sentences with
    Maximal Marginal Relevance (MMR) algorithm, similar as key phrase extraction in EmbedRank:
    https://github.com/swisscom/ai-research-keyphrase-extraction/blob/master/swisscom_ai/research_keyphrase/model/method.py.
    """
    @staticmethod
    def process(args, text, id, narrative_storage_path, tokenizer, token_limit):
        # split text, save to temp, change args path, summarize from the temp
        # for each of the texts: summarize them
        path_to_summaries = split_and_summarize_chunks(args, text, id, narrative_storage_path, tokenizer,
                                                       token_limit)
        summary_files = find_all_summaries(path_to_summaries)
        summaries_combined = combine_chunks(summary_files, path_to_summaries)
        # combine, cluster sentences and find cluster centres
        # select the most central sentences with EmbedRank
        # apply EmbedRank with Diversity (MMR)
        candidates = nltk.sent_tokenize(summaries_combined)
        candidates_embedded = embedder.encode(candidates)
        key_sentences, _ = MMR(summaries_combined, candidates, candidates_embedded, 0.5, 4, embedder)
        summary = " ".join(key_sentences)
        # return the original path
        args.documents_dir = narrative_storage_path
        # delete the temp
        delete_temp(args.documents_dir)
        return summary


class Heuristic4:
    """
    --- Clus#4Only ---
    Cluster all the sentences into 4 clusters and combine their Medoids to get one representative summary.
    """
    @staticmethod
    def process(args, text, id, narrative_storage_path, tokenizer, token_limit):
        # cluster sentences and find cluster centres
        cluster_centers = find_cluster_centres(text, num_clusters=4)
        # return a summary
        summary = " ".join(cluster_centers)
        return summary

class Heuristic5:
    """
    --- MMROnly ---
    Select 4 most representative sentences with the MMR algorithm, as in SumMMR, combine them to get one summary.
    """
    @staticmethod
    def process(args, text, id, narrative_storage_path, tokenizer, token_limit):
        # apply EmbedRank with Diversity (MMR)
        candidates = nltk.sent_tokenize(text)
        candidates_embedded = embedder.encode(candidates)
        key_sentences, _ = MMR(text, candidates, candidates_embedded, 0.5, 4, embedder)
        summary = " ".join(key_sentences)
        return summary

class Heuristic6:
    """
    --- MMR with subsequent summarization ---
    Select 5 most representative sentences with the MMR algorithm, as in SumMMR, combine them and summarize to get one summary.
    """
    @staticmethod
    def process(args, text, id, narrative_storage_path, tokenizer, token_limit):
        # apply EmbedRank with Diversity (MMR)
        candidates = nltk.sent_tokenize(text)
        candidates_embedded = embedder.encode(candidates)
        key_sentences, _ = MMR(text, candidates, candidates_embedded, 0.5, 5, embedder)
        summary = " ".join(key_sentences)
        path_to_summaries = split_and_summarize_chunks(args, summary, id, narrative_storage_path, tokenizer, token_limit)
        summary_files = find_all_summaries(path_to_summaries)
        summaries_combined = combine_chunks(summary_files, path_to_summaries)
        # for each of the summaries: concatenate, check if less than 510 tokens, otherwise split again, summarize again
        while len(summary_files) > 1:
            path_to_summaries = split_and_summarize_chunks(args, summaries_combined, id, path_to_summaries, tokenizer,
                                                           token_limit)
            summary_files = find_all_summaries(path_to_summaries)
            summaries_combined = combine_chunks(summary_files, path_to_summaries)
        # return the original path
        args.documents_dir = narrative_storage_path
        # delete the temp
        delete_temp(args.documents_dir)
        # return a summary
        return summaries_combined


def summarization_pipeline(args):
    """
    The main function for running summarization of long texts based on a given heuristic.
    Args:
        args: arguments defined by the user, consult parse_args() in bertabs/run_summarization.py

    """
    unprocessed_ids = find_unprocessed_files(args.documents_dir) # ids
    heuristic = available_heuristics[args.heuristic]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    NARRATIVE_STORAGE_PATH = args.documents_dir
    result_folder = os.path.join(NARRATIVE_STORAGE_PATH, f"heuristic{args.heuristic}")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    unprocessed_ids = sorted(unprocessed_ids)
    for id in unprocessed_ids:
        logging.info(f"Processing file number {id}")
        narrative_long = read_file_contents(os.path.join(NARRATIVE_STORAGE_PATH, f"narrative_{id}_raw.txt"))
        summary = heuristic.process(args, narrative_long, id, NARRATIVE_STORAGE_PATH, tokenizer, TOKEN_LIMIT)
        write_file_contents(os.path.join(result_folder, f"narrative_{id}_summary.txt"), summary)
        logging.info("Moving up to the next id.")


available_heuristics = {
    1: Heuristic1,
    2: Heuristic2,
    3: Heuristic3,
    4: Heuristic4,
    5: Heuristic5,
    6: Heuristic6
}


if __name__ == "__main__":
    args = parse_args()
    summarization_pipeline(args)