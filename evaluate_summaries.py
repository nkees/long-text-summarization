import rouge
import nltk
import os
import argparse
import datetime
from manage_files import find_all_summaries, read_file_contents, find_summary_ids
from bertabs.run_summarization import format_rouge_scores

nltk.download("punkt")


def evaluate(path_to_gen_summaries, path_to_human_summaries):
    """
    Runs evaluation of a given heuristic through comparing automatically produced summaries
    with human-made reference summaries

    Saves the scores in a separate file.

    """

    rouge_evaluator = rouge.Rouge(
                metrics=["rouge-n", "rouge-l"],
                max_n=3,
                apply_avg=True,
                apply_best=False,
                alpha=0.5,  # Default F1_score
                weight_factor=1.2,
                stemming=True,
            )

    generated_summaries_with_ids = procure_and_sort_summaries(path_to_gen_summaries)
    reference_summaries_with_ids = procure_and_sort_summaries(path_to_human_summaries)

    reference_sum_ids = [int(x[1]) for x in reference_summaries_with_ids]
    generated_summaries_with_ids_filtered = [tuple_ for tuple_ in generated_summaries_with_ids if int(tuple_[1]) in reference_sum_ids]
    gen_sum_ids = [int(x[1]) for x in generated_summaries_with_ids_filtered]
    reference_summaries_with_ids_filtered = [tuple_ for tuple_ in reference_summaries_with_ids if int(tuple_[1]) in gen_sum_ids]
    assert len(generated_summaries_with_ids_filtered) == len(reference_summaries_with_ids_filtered)

    gen_summaries = [x[0] for x in generated_summaries_with_ids_filtered]
    ref_summaries = [x[0] for x in reference_summaries_with_ids_filtered]

    with open("summaries.txt", "w") as file:
        for i in ref_summaries:
            file.write(f"{i}\n")

    scores = rouge_evaluator.get_scores(gen_summaries, ref_summaries)
    len_scores = evaluate_length_factor(gen_summaries, ref_summaries)

    str_scores = format_rouge_scores(scores)
    save_rouge_scores(str_scores, len_scores)

    print(str_scores)


def evaluate_length_factor(gen_summaries, reference_summaries):
    scores = []
    for gen, ref in list(zip(gen_summaries, reference_summaries)):
        gen_tokenized = nltk.word_tokenize(gen)
        ref_tokenized = nltk.word_tokenize(ref)
        score = len(gen_tokenized)/len(ref_tokenized)
        scores.append(score)
    avg_score = sum(scores)/len(scores)
    return avg_score


def procure_and_sort_summaries(path):
    """ Guarantees the right order of the summaries and alignment between the ids in both lists. """
    summary_files = find_all_summaries(path)
    summaries = [read_file_contents(os.path.join(path, filename)) for filename in summary_files]
    summaries = [format_summary(t) for t in summaries]
    summary_ids = find_summary_ids(summary_files)
    tuple_list = list(zip(summaries, summary_ids))
    tuple_list_sorted = sorted(tuple_list, key=lambda tup: tup[1])
    return tuple_list_sorted  # tuple : (summary, summary_id)


def save_rouge_scores(str_scores, len_scores):
    """ Saves the evaluation outputs """
    now = datetime.datetime.now()
    with open(os.path.join("eval_results", f"rouge_scores_heuristic{args.heuristic}_{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}.txt"), "w") as output:
        output.write(f"{str_scores}\n\nLen scores: {len_scores}")


def format_summary(raw_summary):
    """ Transforms the output into nicely formatted summaries. """
    summary = (
        raw_summary.replace("[unused0]", "")
        .replace("[unused3]", "")
        .replace("[PAD]", "")
        .replace("[unused1]", "")
        .replace(r" +", " ")
        .replace(" [unused2] ", ". ")
        .replace("[unused2]", "")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(" ,", ",")
        .replace(" .", ".")
        .strip()
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heuristic",
        default=None,
        required=True,
        type=int,
        help="Define the heuristic you want to apply for processing long texts (larger than 512 tokens)."
             "1 stands for splitting the text, summarizing the parts and then summarizing summaries,"
             "2 stands for splitting the text, summarizing the parts, clustering sentences and select the centers in each cluster,"
             "3 stands for splitting the text, summarizing the parts, applying Maximal Marginal Relevance"
    )
    parser.add_argument(
        "--path_ref",
        default=None,
        type=str,
        help="Path to reference summaries"
    )
    parser.add_argument(
        "--path_gen",
        default=None,
        type=str,
        help="Path to automatically generated summaries"
    )

    args = parser.parse_args()
    if not args.path_ref:
        PATH_TO_HUMAN_SUMMARIES = os.path.join("experiments", "reference")
    else:
        PATH_TO_HUMAN_SUMMARIES = args.path_ref

    if not args.path_gen:
        PATH_TO_GENERATED_SUMMARIES = os.path.join("experiments", f"heuristic{args.heuristic}_complete")
    else:
        PATH_TO_GENERATED_SUMMARIES = args.path_gen

    evaluate(PATH_TO_GENERATED_SUMMARIES, PATH_TO_HUMAN_SUMMARIES)
