# encoding: utf-8
import os
from os import listdir
from os.path import isfile, join
import re
import shutil
import logging

logging.getLogger().setLevel(logging.INFO)

def preprocess_narratives(narrative):
    """
    Cleans up the narratives for better summarization
    :param narrative: raw narrative text (string)
    :return: processed narrative text (string)
    """
    narrative = re.sub(r'\.\.\.', '', narrative)
    narrative = re.sub(r'XX/XX/XXXX', '23 September 2018', narrative)
    narrative = re.sub(r'XXXX / XXXX / XXXX', '23 September 2018', narrative)
    narrative = re.sub('XX/XX/\d+', '23 September 2018', narrative)
    narrative = re.sub('X+ ?', 'A NAMED ENTITY ', narrative)
    narrative = re.sub(r"do n't", 'do not', narrative)
    narrative = re.sub(r"{", '', narrative)
    narrative = re.sub(r"}", '', narrative)
    narrative = re.sub(r' \.', '.', narrative)
    narrative = re.sub(r' \)', r')', narrative)
    narrative = re.sub(r'  ', ' ', narrative)
    narrative = re.sub('(A NAMED ENTITY )+(A NAMED ENTITY)*', 'A NAMED ENTITY ', narrative)
    narrative = re.sub('(23 September 2018 )+', '23 September 2018 ', narrative)
    narrative = re.sub(r' \.', '.', narrative)
    return narrative


def read_file_contents(file_with_narrative):
    with open(file_with_narrative, "r", encoding="utf-8") as file:
        try:
            narrative = file.read()
        except UnicodeDecodeError:
            with open(file_with_narrative, "r", encoding="windows-1252") as the_file:
                narrative = the_file.read()
    return narrative


def write_file_contents(path, contents):
    file_name = path
    with open(file_name, "w", encoding="utf-8") as document:
        document.write(contents)


def find_all_narratives(path):
    """Searches for all narrative files in the folder."""
    rawfiles = [file for file in listdir(path) if isfile(join(path, file)) and "raw" in file]
    return rawfiles # list of file names (without specific paths)


def find_all_summaries(path):
    """Searches for all summary files in the folder."""
    summaries = [file for file in listdir(path) if isfile(join(path, file)) and "summary" in file]
    return summaries # list of file names (without specific paths)


def find_raw_ids(rawfiles):
    """Identifies IDs of the narratives."""
    raw_ids = [re.search("_\d+_", file).group(0).replace("_", "") for file in rawfiles]
    return raw_ids


def find_summary_ids(summaries):
    """Identifies IDs of the summaries."""
    summary_ids = [re.search("_\d+_", file).group(0).replace("_", "") for file in summaries]
    return summary_ids


def compare_ids(pool_of_narratives_ids, pool_of_summaries_ids):
    """Finds which narrative IDs are not yet among the summary IDs and thus need to be summarized."""
    to_summarize = []
    for i in pool_of_narratives_ids:
        if i not in pool_of_summaries_ids:
            to_summarize.append(i)
    return to_summarize


def add_file_to_temp(path, file_name):
    temp_path = os.path.join(path, "temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    contents = read_file_contents(os.path.join(path, file_name))
    extension = file_name.split(".")[0]


def add_unprocessed_files_to_temp(path):
    """
    Collects the ids of the files which have not been summarized and transfer them to the folder temp.
    :param path: path to the folder where all the narratives are based. The files will be transported to path/temp
    """
    unprocessed_ids = find_unprocessed_files(path)
    temp_path = os.path.join(path, "temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    num_narratives = len(unprocessed_ids)
    logging.info("  These ids will be processed: {}".format(unprocessed_ids))
    logging.info("  There are {} new narratives on the total.".format(num_narratives))
    for i in unprocessed_ids:
        number = i
        contents = read_file_contents(os.path.join(path, "narrative_{}_raw.txt".format(number)))
        preprocessed_contents = preprocess_narratives(contents)
        if preprocessed_contents != contents:
            logging.info("  Narrative {} has been changed.".format(number))
        if len(contents) >= 230:
            # if the narrative has more than 230 characters:
            write_file_contents(os.path.join(temp_path, "narrative_{}_raw.txt".format(number)), preprocessed_contents)
        else:
            write_file_contents(os.path.join(path, "narrative_{}_summary.txt".format(number)), preprocessed_contents)
        # save the preprocessed narrative back to the file:
        write_file_contents(os.path.join(path, "narrative_{}_raw.txt".format(number)), preprocessed_contents)


def delete_temp(path):
    shutil.rmtree(os.path.join(path, "temp"))


def return_summarized_files_back(path):
    """
    Collects the summaries pushes them to the folder specified in path.
    :param path: path to save the summaries to from path/temp
    """
    summaries = find_all_summaries(os.path.join(path, "temp"))
    ids = find_summary_ids(summaries)
    for i in ids:
        number = i
        summary = read_file_contents(os.path.join(path, "temp", "narrative_{}_summary.txt".format(number)))
        write_file_contents(os.path.join(path, "narrative_{}_summary.txt".format(number)), summary)
    delete_temp(path)


def find_unprocessed_files(path):
    """
    Finds the narratives which (are new and) have not been summarized yet.
    :param path: the directory in which the narratives and the summaries lie
    :return: a list of identifiers for the new narratives (or those which have simply not been summarized yet)
    """
    rawfiles = find_all_narratives(path)
    summaries = find_all_summaries(path)
    raw_ids = find_raw_ids(rawfiles)
    summary_ids = find_summary_ids(summaries)
    to_summarize = compare_ids(raw_ids, summary_ids)
    return to_summarize # ids


def save_narratives(file_with_documents, file_with_labels, amount, from_index=0):
    """
    Save the narratives from a data frame to single files separate for each narrative.
    :param file_with_documents: path to the file where the narratives are stored
    :param file_with_labels: path to the file where the labels of those narratives are stored
    :param from_index: starting index of the narratives in the data frame
    :param amount: the amount of narratives to save
    """
    with open(file_with_documents, "r", encoding="utf-8") as file:
        narratives = file.read()
        narratives = narratives.split("\n")
        n = from_index
        path = "bertabs/data"
        logging.info("  Files will be saved to: ({})".format(path))
        for i in narratives[from_index:(from_index+amount)]:
            file_name = os.path.join(path, "narrative_{}_raw.txt".format(n))
            # print("NARRATIVE " + str(n))
            # print(i)
            # command = input()
            # if command == "":
            #    continue
            while os.path.exists(file_name):
                n += 1
                file_name = os.path.join(path, "narrative_{}_raw.txt".format(n))
            file_name = os.path.join(path, "narrative_{}_raw.txt".format(n))
            with open(file_name, "w", encoding="utf-8") as document:
                document.write(i)
            n += 1

    with open(file_with_labels, "r", encoding="utf-8") as file:
        labels = file.read()
        labels = labels.split("\n")
        n = from_index
        for i in labels[from_index:amount]:
            file_name = "bertabs/gold/label_{}.txt".format(n)
            # print("NARRATIVE " + str(n))
            # print(i)
            # command = input()
            # if command == "":
            #    continue
            # else:
            #     break
            with open(file_name, "w", encoding="utf-8") as document:
                document.write(i)
            n += 1


def count_lengths(path):
    lengths = []
    summaries = [file for file in listdir(path) if isfile(join(path, file)) and "summary" in file]
    for i in summaries:
        summary = read_file_contents(join(path, i))
        length = len(summary)
        lengths.append(length)
    mean = sum(lengths)/len(lengths)
    maximum = lengths.index(max(lengths))
    print(sorted(lengths))
    print(summaries[maximum])
    minimum = lengths.index(min(lengths))
    print(summaries[minimum])
    return mean, maximum, minimum


def plot_lengths():
    """
    Determine the optimal shortest narrative length.
    :return:
    """
    raw = find_all_narratives("bertabs/data")
    summaries = find_all_summaries("bertabs/data")
    #raw_ids = find_raw_ids(raw)
    #summary_ids = find_summary_ids(summaries)
    #to_exclude = compare_ids(raw_ids, summary_ids)

    #to_plot_ids = compare_ids(raw_ids, to_exclude)
    #to_plot = []
    #for i in raw:
    #    for id in to_plot_ids:
    #        if id in i:
    #            to_plot.append(i)

    summaries_lengths = []
    for i in summaries:
        summary = read_file_contents(join("bertabs/data", i))
        length = len(summary)
        summaries_lengths.append(length)
    print(len(summaries_lengths))

    narrative_lengths = []
    for i in raw:
        narr = read_file_contents(join("bertabs/data", i))
        length = len(narr)
        narrative_lengths.append(length)
    print(len(narrative_lengths))

    import matplotlib.pyplot as plt
    plt.plot(summaries_lengths, narrative_lengths, 'ro')
    plt.plot([1, 600], [1, 600]) # the line indicating which narratives are shorter than their summaries
    plt.plot([1, 600], [230, 230]) # the line indicating an optimal boundary for skipping the summarization task
    plt.ylabel('X: summary lengths, Y: corresponding narrative lengths')
    plt.show()
    # all the points below the blue line have their summaries longer than the narratives themselves, which cannot be
    # ToDo: threshold defined as 230: do not produce a summary if the length is below that

def evaluate_quality(path):
    scores = [file for file in listdir(path) if isfile(join(path, file)) and "score" in file]
    rates = []
    for i in scores:
        path_to_score = os.path.join(path, i)
        with open(path_to_score, "r") as score:
            rate = float(score.read())
            rates.append(rate)
    return sum(rates)/len(rates)