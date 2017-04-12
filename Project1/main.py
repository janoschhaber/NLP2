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
from nltk.translate import Alignment
from nltk.translate import metrics

INDEX_WORDS = True
VERBOSE = False
TRAIN_LIMIT = 1000
ITERATIONS = 3
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
    if f_word in trans_probs:
        if e_word in trans_probs[f_word]:
            # print("Found: Probability of {} given {} is {}".format(e_word, f_word, trans_probs[f_word][e_word]))
            return trans_probs[f_word][e_word]
        else:
            f_nr_entries = len(trans_probs[f_word])
            f_missing = nr_e_words - f_nr_entries
            f_prob_mass = 0
            for entry in trans_probs[f_word].values():
                f_prob_mass += entry
            if float(1 - f_prob_mass) / f_missing == 0:
                print("Not Found: Probability of {} given {} is {} with a remaining probability mass of {} and {} words not assigned yet ".format(e_word, f_word, float(1 - f_prob_mass) / f_missing, float(1 - f_prob_mass), f_missing))
                print trans_probs[f_word].values()
                return float(1) / nr_e_words
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
        print ("Probability for {} given {} is {}".format(e_pred, f_sent[index],
        get_trans_prob(e_pred, f_sent[index], trans_probs, nr_e_words)))
        perplexity += np.log2(get_trans_prob(e_pred, f_sent[index], trans_probs, nr_e_words))
    return perplexity


def get_likelihood(e_pred_sent, f_sent, trans_probs, nr_e_words):
    """
    Returns a predicted translation's likelihood given the current model
    :param e_pred_sent: the predicted translation
    :param f_sent: the original sentence
    :param trans_probs: the current model (translation probabilities p(e_word|f_word)) = trans_probs[f_word][e_word] = probability
    :param nr_e_words: total number of english words encountered in training
    :return: a predicted translation's likelihood given the current model
    """
    # TODO: Check for underflow
    likelihood = 1
    for index, e_pred in enumerate(e_pred_sent):
        likelihood *= get_trans_prob(e_pred, f_sent[index], trans_probs, nr_e_words)
    return likelihood


def align_sentences(e_sents, f_sents, e_dict, trans_probs):
    """
    Translates a given set of sentences using the specified model
    :param v_aligned: zipped list of aligned validation set sentences
    :param e_dict: string to integer index dictionary for english
    :param f_dict: string to integer index dictionary for french
    :param trans_probs: the current model (translation probabilities p(e_word|f_word)) = trans_probs[f_word][e_word] = probability
    :return: a zipped list of translations and alignments
    """
    nr_e_words = len(e_dict)
    translations = [None] * len(f_sents)
    alignments = [None] * len(f_sents)

    aligned = zip(e_sents, f_sents)
    for pair_id, pair in enumerate(aligned):
        e_sent = pair[0]
        print e_sent
        f_sent = pair[1]
        print f_sent
        sent_pred_trans = [None] * len(f_sent)
        sent_alignment = [None] * len(f_sent)

        for f_index, f_word in enumerate(f_sent):
            print("Translating {}".format(f_word))
            e_probs = np.zeros(len(e_sent))
            for e_index, e_word in enumerate(e_sent):
                e_probs[e_index] = get_trans_prob(e_word, f_word, trans_probs, nr_e_words)
                print("Probability for {} is {}".format(e_word, e_probs[e_index]))
            sent_pred_trans[f_index] = e_sent[np.argmax(e_probs)]
            sent_alignment[f_index] = (f_index+1, np.argmax(e_probs)+1)

        translations[pair_id] = sent_pred_trans
        alignments[pair_id] = sent_alignment
    return zip(translations, alignments)


def evaluate_model(e_sents, f_sents, e_sents_orig, e_dict, e_dict_inv, trans_probs, gold_alignments):
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
    sent_likelihoods = np.zeros(len(f_sents))
    sent_AERs = np.zeros(len(f_sents))

    model_output = align_sentences(e_sents, f_sents, e_dict, trans_probs)
    for index, pair in enumerate(model_output):
        e_pred_sent = pair[0]
        f_sent = f_sents[index]
        alignment = pair[1]

        print ("Predicted translation: {}".format(decode_sentence(e_pred_sent, e_dict_inv)))
        print ("Actual translation: {}".format(e_sents_orig[index]))
        print ("Alignment: {}".format(alignment))
        print ("Gold standard alignment: {}".format(gold_alignments[index]))

        sent_perplexities[index] = get_perplexity(e_pred_sent, f_sent, trans_probs, nr_e_words)
        sent_likelihoods[index] = get_likelihood(e_pred_sent, f_sent, trans_probs, nr_e_words)

        # TODO: Fix AER
        # ref = Alignment([(0, 0), (1, 1), (2, 2)])
        # test = Alignment([(0, 0), (1, 2), (2, 1)])
        # print metrics.alignment_error_rate(ref, test)

        sent_AERs[index] = 1 # metrics.alignment_error_rate(gold_alignments[index], alignment)

    return [-np.sum(sent_perplexities), sum(sent_likelihoods)/len(sent_likelihoods), sum(sent_AERs)/len(sent_AERs)]


def encode_sentence(sentence, dictionary):
    """
    Returns the indexed version of the input sentence
    :param sentence: a string list sentence
    :return: the integer index list sentence encoding
    """
    encoded = np.zeros(len(sentence))
    for index, word in enumerate(sentence):
        if word in dictionary:
            encoded[index] = dictionary[word]
        else:
            encoded[index] = -1
    return encoded


def decode_sentence(sentence, dictionary):
    """
    Returns the decoded version of the input sentence
    :param sentence: an indexed list sentence
    :return: the decoded list of strings sentence
    """
    decoded = [None] * len(sentence)
    for index, word in enumerate(sentence):
        if word in dictionary:
            decoded[index] = dictionary[word]
        else:
            decoded[index] = 'NULL'
    return decoded


def produce_output(v_e_sents, v_f_sents, e_dict, f_dict, e_inv_dict, trans_probs, gold_alignments, performances, filename='results.dat'):
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
    if INDEX_WORDS:
        e_sents = [encode_sentence(sent, e_dict) for sent in v_e_sents]
        f_sents = [encode_sentence(sent, f_dict) for sent in v_f_sents]
    else:
        e_sents = v_e_sents
        f_sents = v_f_sents

    model_output = align_sentences(e_sents, f_sents, e_dict, trans_probs)
    f = open(filename, 'w')
    f.write("Model perplexity: {}\n".format(performances[0]))
    f.write("Average sentence likelihood: {}\n".format(performances[1]))
    f.write("Average sentence AER: {}\n\n".format(performances[2]))
    for i, output in enumerate(model_output):
        translation = output[0]
        alignment = output[1]
        f.write("{}\nwas translated as \n{} \ngold standard translation: \n{}\n Predicted alignment: \n{}\n True alignment: \n{}\n\n"
                .format(v_f_sents[i], decode_sentence(translation, e_inv_dict), v_e_sents[i], alignment, gold_alignments[i]))
    f.close()


def get_gold_alignments(filename = 'validation/dev.wa.nonullalign'):
    """
    Returns a NLTK tuple representation of the gold standard alignments as specified in an EGYPT formatted file
    :param filename: path to the file with the alignment annotations
    :return: a list of tuples of gold standard alignments
    """
    lines = read_file(filename)
    alignments = []
    sent_align = []
    index = 1
    for line in lines:
        annotation = [int(word) for word in line.split()[0:3]]
        if annotation[0] == index:
            sent_align.append((annotation[1], annotation[2]))
        else:
            index += 1
            alignments.append(sent_align)
            sent_align = [(annotation[1], annotation[2])]
    return alignments


def main():
    start = time.time()

    # Read training data
    e_lines = read_file('training/hansards.36.2.e')
    f_lines = read_file('training/hansards.36.2.f')
    if len(f_lines) != len(e_lines):
        print("ERROR: Training data not aligned properly!")
        quit()

    train_size = len(f_lines)
    print ("Train set size: {} sentences".format(train_size))
    if TRAIN_LIMIT > 0:
        e_lines = e_lines[0:TRAIN_LIMIT]
        f_lines = f_lines[0:TRAIN_LIMIT]
    print ("Limited to {} sentences for debugging".format(TRAIN_LIMIT))

    e_words = set(word.lower() for line in e_lines for word in line.split())
    nr_e_words = len(e_words)
    e_dict = wordlist2dict(e_words) #.union("NULL"))
    e_dict_inv = inverse_dict(e_dict)
    f_words = set(word.lower() for line in f_lines for word in line.split())
    nr_f_words = len(f_words)
    f_dict = wordlist2dict(f_words)
    f_dict_inv = inverse_dict(f_dict)
    if VERBOSE: print "The english dictionary contains {} words, the french one {}".format(nr_e_words, nr_f_words)

    e_sents = [[word.lower() for word in line.split()] for line in e_lines]
    f_sents = [[word.lower() for word in line.split()] for line in f_lines]
    if INDEX_WORDS:
        e_sents = [encode_sentence(sent, e_dict) for sent in e_sents]
        f_sents = [encode_sentence(sent, f_dict) for sent in f_sents]

    aligned = zip(e_sents, f_sents)

    # Read validation data
    v_e_lines = read_file('validation/dev.e')
    v_f_lines = read_file('validation/dev.f')
    if VAL_LIMIT > 0:
        v_e_lines = v_e_lines[0:VAL_LIMIT]
        v_f_lines = v_f_lines[0:VAL_LIMIT]
    if len(e_lines) != len(f_lines):
        print("ERROR: Validation data not aligned properly!")
        quit()
    print ("Validation set size: {} sentence(s)".format(len(v_f_lines)))

    v_e_sents = [[word.lower() for word in line.split()] for line in v_e_lines]
    v_f_sents = [[word.lower() for word in line.split()] for line in v_f_lines]

    if INDEX_WORDS:
        v_e_sents_encoded = [encode_sentence(sent, e_dict) for sent in v_e_sents]
        v_f_sents_encoded = [encode_sentence(sent, f_dict) for sent in v_f_sents]
    else:
        v_e_sents_encoded = v_e_sents
        v_f_sents_encoded = v_f_sents

    gold_alignments = get_gold_alignments()

    # TODO: Fix AER
    # ref = Alignment([(0, 0), (1, 1), (2, 2)])
    # test = Alignment([(0, 0), (1, 2), (2, 1)])
    # print metrics.alignment_error_rate(ref, test)

    end = time.time()
    print("Execution time: {}".format(end - start))

    print f_dict_inv[754]


if __name__ == "__main__":
    main()
