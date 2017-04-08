#!/usr/bin/env python

"""
NLP2 2017
University of Amsterdam

Project 1: IBM Model 1 translation
"""
__author__ = "Edwin Lima Valladares, Ioanna Sanida, Janosch Haber"
__copyright__ = "Copyright 2017"
__credits__ = ["Edwin Lima Valladares, Ioanna Sanida, Janosch Haber"]
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "janosch.haber@student.uva.nl"
__status__ = "First Draft"

import numpy as np
import time

INDEX_WORDS = True
VERBOSE = False
TRAIN_LIMIT = 1000
VAL_LIMIT = 1

def read_file(filename):
    """
    Returns a list of sentences from a specified file
    :param filename: path to file
    :return: list of lines (sentences) in the specified file
    """
    with open(filename) as f:
        lines = f.readlines()
    return lines


def wordlist2dict(words):
    """
    Returns a dictionary to index all words of a given list through integers
    :param words: set of words to be converted to integer indexes
    :return: a dictionary that links each string word to an integer index representation
    """
    w_dict = {}
    for index, word in enumerate(words):
        w_dict[word] = index
    return w_dict


def inverse_dict(dictionary):
    """
    Inverts the indexing dictionary to retrieve words from indexes
    :param dictionary: the dictionary to invert
    :return: the inverted dictionary
    """
    inv_dict = {}
    for key, value in dictionary.iteritems():
        inv_dict[value] = key
    return inv_dict


def get_trans_prob(e_word, f_word, trans_probs, nr_e_words):
    """
    Returns the expected translation probability of a given pair of aligned words
    :param e_word: english word
    :param f_word: french word
    :param trans_probs: dictionary[f_word][e_word] with all recorded conditional probabilities p(e_word|f_word)
    :param nr_e_words: total number of encountered english words
    :return: conditional probability p(e_word|f_word)
    """
    if trans_probs.has_key(f_word):
        if trans_probs[f_word].has_key(e_word):
            # print("Found: Probability of {} given {} is {}".format(e_word, f_word, trans_probs[f_word][e_word]))
            return trans_probs[f_word][e_word]
        else:
            f_nr_entries = len(trans_probs[f_word])
            f_missing = nr_e_words - f_nr_entries
            f_prob_mass = 0
            for entry in trans_probs[f_word].values():
                f_prob_mass += entry
            # print("Not Found: Probability of {} given {} is {} with a remaining probability mass of {} and {} words not assigned yet ".format(e_word, f_word, float(1 - f_prob_mass) / f_missing, float(1 - f_prob_mass), f_missing))
            return float(1 - f_prob_mass) / f_missing
    else:
        # print("No Key: Probability of {} given {} is {}".format(e_word, f_word, float(1) / nr_e_words))
        return float(1) / nr_e_words


def get_perplexity(e_pred_sent, f_sent, trans_probs, nr_e_words):
    """
    Returns a sentence's translation's perplexity given the current state of the model
    :param e_pred_sent: the predicted translation
    :param f_sent: the original sentence
    :param trans_probs: the current model (translation probabilities p(e_word|f_word)) = trans_probs[f_word][e_word] = probability
    :param nr_e_words: total number of english words encountered in training
    :return: the sentence's translation's perplexity given the current state of the model
    """
    perplexity = 0
    for index, e_pred in enumerate(e_pred_sent):
        # TODO FIX: One of the input formats is wrong!
        print get_trans_prob(e_pred, f_sent[index], trans_probs, nr_e_words)
        perplexity += np.log2(get_trans_prob(e_pred, f_sent[index], trans_probs, nr_e_words))
    return perplexity


def align_sentences(f_sents, f_dict, e_sents, e_dict, trans_probs):
    """
    Translates a given set of sentences using the specified model
    :param f_sents: the set of french sentences
    :param f_dict: string to integer index dictionary for french
    :param e_dict: string to integer index dictionary for english
    :param trans_probs: the current model (translation probabilities p(e_word|f_word)) = trans_probs[f_word][e_word] = probability
    :return: a list of translations
    """
    nr_e_words = len(e_dict)
    translations = [None] * len(e_sents)

    aligned = zip(e_sents, f_sents)
    for pair_id,  pair in enumerate(aligned):
        e_sent = [word.lower() for word in pair[0].split()]
        f_sent = [word.lower() for word in pair[1].split()]
        sent_alignment = [None] * len(f_sent)

        f_index = 0
        for f_word in f_sent:
            if INDEX_WORDS:
                if f_dict.has_key(f_word):
                    f_word = f_dict[f_word]
                else:
                    f_word = -1

            e_probs = np.zeros(len(e_sent))
            e_index = 0
            for e_word in e_sent:
                if INDEX_WORDS:
                    if e_dict.has_key(e_word):
                        e_word = e_dict[e_word]
                    else:
                        e_word = -1

                e_probs[e_index] = get_trans_prob(e_word, f_word, trans_probs, nr_e_words)
                e_index += 1
            sent_alignment[f_index] = e_sent[np.argmax(e_probs)]
            f_index += 1

        translations[pair_id] = sent_alignment
    return translations


def evaluate_model(f_sents, f_dict, e_sents, e_dict, trans_probs):
    """
    Returns the current model performance in terms of translation perplexity
    :param f_sents: the set of french sentences
    :param f_dict: string to integer index dictionary for french
    :param e_dict: string to integer index dictionary for english
    :param trans_probs: the current model (translation probabilities p(e_word|f_word)) = trans_probs[f_word][e_word] = probability
    :return: translation perplexity
    """
    nr_e_words = len(e_dict)

    sent_perplexities = np.zeros(len(f_sents))
    translations = align_sentences(f_sents, f_dict, e_sents, e_dict, trans_probs)
    for index, e_pred_sent in enumerate(translations):
        print e_pred_sent
        e_pred_sent = index_sentence(e_pred_sent, e_dict)
        print e_pred_sent
        sent_perplexities[index] = get_perplexity(e_pred_sent, f_sents[index], trans_probs, nr_e_words)
        print sent_perplexities[index]
    return -np.sum(sent_perplexities)


def index_sentence(sentence, dictionary):
    """
    Returns the indexed version of the input sentence
    :param sentence: a string list sentence
    :return: the integer index list sentence encoding
    """
    encoded = np.zeros(len(sentence))
    for index, word in enumerate(sentence):
        if INDEX_WORDS:
            if dictionary.has_key(word):
                encoded[index] = dictionary[word]
            else:
                encoded[index] = -1
    return encoded


def produce_output(f_sents, f_dict, e_sents, e_dict, trans_probs, filename='results.dat'):
    """
    Produces a translation output for the specified set of sentences
    :param f_sents: the set of french sentences
    :param f_dict: string to integer index dictionary for french
    :param e_sents: gold standard translations
    :param e_dict: string to integer index dictionary for english
    :param trans_probs: the current model (translation probabilities p(e_word|f_word)) = trans_probs[f_word][e_word] = probability
    :param filename:
    :return:
    """
    nr_e_words = len(e_dict)

    sent_perplexities = np.zeros(len(f_sents))
    translations = align_sentences(f_sents, f_dict, e_sents, e_dict, trans_probs)
    for index, e_pred_sent in enumerate(translations):
        e_pred_sent = index_sentence(e_pred_sent, e_dict)
        sent_perplexities[index] = get_perplexity(e_pred_sent, f_sents[index], trans_probs, nr_e_words)

    perplexity = -np.sum(sent_perplexities)
    result = zip(f_sents, translations, e_sents)

    f = open(filename, 'w')
    f.write("Model perplexity: {}\n\n".format(perplexity))
    for trio in result:
        f.write("{}was translated as \n{} \ngold standard translation: \n{}\n\n".format(trio[0], " ".join(trio[1]), trio[2]))
    f.close()


def main():
    start = time.time()
    e_lines = read_file('training/hansards.36.2.e')
    f_lines = read_file('training/hansards.36.2.f')
    if len(f_lines) != len(e_lines):
        print("ERROR: Training data not aligned properly!")
        quit()

    train_size = len(f_lines)
    print ("Train set size: {} sentences".format(train_size))
    print ("Limited to {} sentences for debugging".format(TRAIN_LIMIT))

    v_f_lines = read_file('validation/dev.f')
    v_e_lines = read_file('validation/dev.e')
    v_f_lines = v_f_lines[0:VAL_LIMIT]
    v_e_lines = v_e_lines[0:VAL_LIMIT]
    if len(f_lines) != len(e_lines):
        print("ERROR: Validation data not aligned properly!")
        quit()
    print ("Validation set size: {} sentences".format(len(v_f_lines)))

    e_words = set(word.lower() for line in e_lines for word in line.split())
    e_dict = wordlist2dict(e_words.union("NULL"))
    e_inv_dict = inverse_dict(e_dict)
    f_words = set(word.lower() for line in f_lines for word in line.split())
    f_dict = wordlist2dict(f_words)
    nr_e_words = len(e_words)
    nr_f_words = len(f_words)
    if VERBOSE: print "The english dictionary contains {} words, the french one {}".format(nr_e_words, nr_f_words)

    aligned = zip(e_lines, f_lines)

    trans_probs = {}
    e_f_counts = {}
    total = {}

    last_perplexity = None
    while True:

        counter = 0
        for pair in aligned:
            e_sent = [word.lower() for word in pair[0].split()]
            f_sent = [word.lower() for word in pair[1].split()]
            if VERBOSE: print e_sent, f_sent

            # TODO: Add the zero word to the English sentence

            if VERBOSE: print("Compute Normalization")
            s_total = np.zeros(len(e_sent))
            e_index = 0
            for e_word in e_sent:
                if INDEX_WORDS: e_word = e_dict[e_word]
                for f_word in f_sent:
                    if INDEX_WORDS: f_word = f_dict[f_word]
                    s_total[e_index] = get_trans_prob(e_word, f_word, trans_probs, nr_e_words)
                e_index += 1
            if VERBOSE: print s_total

            if VERBOSE: print("Collect Counts")
            e_index = 0
            for e_word in e_sent:
                if INDEX_WORDS: e_word = e_dict[e_word]
                for f_word in f_sent:
                    if INDEX_WORDS: f_word = f_dict[f_word]

                    # Update Counts
                    if e_f_counts.has_key(f_word):
                        if e_f_counts[f_word].has_key(e_word):
                            current_count = e_f_counts[f_word][e_word]
                            e_f_counts[f_word][e_word] = current_count + get_trans_prob(e_word, f_word, trans_probs, nr_e_words) / s_total[e_index]
                        else:
                            e_f_counts[f_word][e_word] = get_trans_prob(e_word, f_word, trans_probs, nr_e_words) / s_total[e_index]
                    else:
                        e_f_counts[f_word] = {e_word : get_trans_prob(e_word, f_word, trans_probs, nr_e_words) / s_total[e_index]}
                        # if VERBOSE: print e_f_counts[f_word][e_word]
                    # Update F Total
                    if total.has_key(f_word):
                        current_count = total[f_word]
                        total[f_word] = current_count + get_trans_prob(e_word, f_word, trans_probs, nr_e_words) / s_total[e_index]
                    else:
                        total[f_word] = get_trans_prob(e_word, f_word, trans_probs, nr_e_words) / s_total[e_index]
                e_index += 1

            counter += 1
            if counter % 1000 == 0 : print("{}/{}".format(counter, TRAIN_LIMIT))
            if counter == TRAIN_LIMIT : break

        if VERBOSE: print("Estimate Probabilities")
        for f_word in total.keys():
            for e_word in e_f_counts[f_word].keys():
                if trans_probs.has_key(f_word):
                    trans_probs[f_word][e_word] = e_f_counts[f_word][e_word] / total[f_word]
                    # if VERBOSE: print e_f_counts[f_word][e_word]
                else:
                    trans_probs[f_word] = {e_word : e_f_counts[f_word][e_word] / total[f_word]}
                    # if VERBOSE: print e_f_counts[f_word][e_word]

        perplexity = evaluate_model(v_f_lines, f_dict, v_e_lines, e_dict, trans_probs)
        print ("Perplexity: {}".format(perplexity))
        if last_perplexity is None: last_perplexity = perplexity
        elif perplexity / last_perplexity > 0.5: break
        last_perplexity = perplexity

    produce_output(v_f_lines, f_dict, v_e_lines, e_dict, trans_probs)
    end = time.time()
    print("Execution time: {}".format(end - start))


if __name__ == "__main__":
    main()
