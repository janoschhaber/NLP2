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
__version__ = "1.0.9"
__email__ = "janosch.haber@student.uva.nl"
__status__ = "IBM1 Debugging Version"

import numpy as np
import pickle
import time
from nltk.translate import Alignment
from nltk.translate import metrics

INDEX_WORDS = False
VERBOSE = False
TRAIN_LIMIT = 0
ITERATIONS = 5
VAL_LIMIT = 0


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


def get_trans_prob(f_word, e_word, trans_probs, nr_f_words):
    """
    Returns the expected translation probability of a given pair of aligned words
    :param f_word: french word
    :param e_word: english word
    :param trans_probs: dictionary[e_word][f_word] with all recorded conditional probabilities p(f_word|e_word)
    :param nr_f_words: total number of french corpus word types
    :return: conditional probability p(f_word|e_word)
    """
    if e_word in trans_probs:
        if f_word in trans_probs[e_word]:
            # print("Found: Probability of {} given {} is {}".format(f_word, e_word, trans_probs[e_word][f_word]))
            return trans_probs[e_word][f_word]
        else:
            e_nr_entries = len(trans_probs[e_word])
            e_missing = nr_f_words - e_nr_entries
            if e_missing == 0:
                # TODO: Check what is happening here
                # print ("No more words left, {} of {} indexed, {} isn't one of them.".format(e_nr_entries, nr_f_words, f_word))
                return 0
            e_prob_mass = 0
            for entry in trans_probs[e_word].values():
                e_prob_mass += entry
            return float(1 - e_prob_mass) / e_missing
    else:
        # print("No Key: Probability of {} given {} is {}".format(f_word, e_word, float(1) / nr_f_words))
        return float(1) / nr_f_words


def get_perplexity(e_pred_sent, f_sent, trans_probs, nr_f_words):
    """
    Returns a sentence's translation's perplexity given the current state of the model
    :param e_pred_sent: the predicted translation
    :param f_sent: the french sentence
    :param trans_probs: the current model (translation probabilities p(f_word|e_word)) = trans_probs[e_word][f_word] = probability
    :param nr_f_words: total number of french corpus word types
    :return: the sentence's translation's perplexity given the current state of the model
    """
    perplexity = 0
    for index, e_pred in enumerate(e_pred_sent):
        perplexity += np.log2(get_trans_prob(f_sent[index], e_pred, trans_probs, nr_f_words))
    return perplexity


def get_likelihood(e_pred_sent, f_sent, trans_probs, nr_f_words):
    """
    Returns a predicted sentence's likelihood given the current model
    :param e_pred_sent: the predicted translation
    :param f_sent: the french sentence
    :param trans_probs: the current model (translation probabilities p(f_word|e_word)) = trans_probs[e_word][f_word] = probability
    :param nr_f_words: total number of french corpus word types
    :return: a predicted sentence's likelihood
    """
    # TODO: Check for underflow
    likelihood = 1
    for index, e_pred in enumerate(e_pred_sent):
        likelihood *= get_trans_prob(f_sent[index], e_pred, trans_probs, nr_f_words)
    return likelihood


def align_sentences(e_sents, f_sents, trans_probs, nr_f_words):
    """
    Translates a given set of sentences using the specified model
    :param e_sents: english sentences
    :param f_sents: french sentences
    :param trans_probs: the current model (translation probabilities p(e_word|f_word)) = trans_probs[f_word][e_word] = probability
    :param nr_f_words: total number of french corpus word types
    :return: a zipped list of translations and alignments
    """
    translations = [None] * len(f_sents)
    alignments = [None] * len(f_sents)

    aligned = zip(e_sents, f_sents)
    for pair_id, pair in enumerate(aligned):
        e_sent = pair[0]
        f_sent = pair[1]
        sent_pred_trans = [None] * len(f_sent)
        sent_alignment = [None] * len(f_sent)

        for f_index, f_word in enumerate(f_sent):
            # print("Aligning {}".format(f_word))
            e_probs = np.zeros(len(e_sent))
            for e_index, e_word in enumerate(e_sent):
                e_probs[e_index] = get_trans_prob(f_word, e_word, trans_probs, nr_f_words)
                # print("Probability for {} is {}".format(e_word, e_probs[e_index]))
            sent_pred_trans[f_index] = e_sent[np.argmax(e_probs)]
            # print("Most likely alignment is {}".format(e_sent[np.argmax(e_probs)]))
            sent_alignment[f_index] = (np.argmax(e_probs), f_index+1)

        translations[pair_id] = sent_pred_trans
        # Removing NULL productions
        sent_alignment = [pair for pair in sent_alignment if pair[0] != 0]
        alignments[pair_id] = sent_alignment
    return zip(translations, alignments)


def evaluate_model(f_sents, e_sents_orig, nr_f_words, e_sents, e_dict_inv, trans_probs, gold_alignments):
    """
    Returns the current model performance in terms of translation perplexity
    :param f_sents: the set of french sentences
    :param e_sents_orig:
    :param nr_f_words:
    :param e_sents:
    :param e_dict_inv:
    :param trans_probs:
    :param gold_alignments:
    :return:
    """
    sent_perplexities = np.zeros(len(f_sents))
    sent_likelihoods = np.zeros(len(f_sents))
    sent_aers = np.zeros(len(f_sents))

    model_output = align_sentences(e_sents, f_sents, trans_probs, nr_f_words)
    for index, pair in enumerate(model_output):
        e_pred_sent = pair[0]
        f_sent = f_sents[index]
        alignment = pair[1]

        if VERBOSE: print ("Sentence: {}".format(f_sent))
        if VERBOSE:
            if INDEX_WORDS: print ("Predicted translation: {}".format(decode_sentence(e_pred_sent, e_dict_inv)))
            else: print ("Predicted translation: {}".format(e_pred_sent))
        if VERBOSE: print ("Actual translation: {}".format(e_sents_orig[index]))
        if VERBOSE: print ("Alignment: {}".format(alignment))
        if VERBOSE: print ("Gold standard alignment: {}".format(gold_alignments[index]))

        sent_perplexities[index] = get_perplexity(e_pred_sent, f_sent, trans_probs, nr_f_words)
        sent_likelihoods[index] = get_likelihood(e_pred_sent, f_sent, trans_probs, nr_f_words)
        sent_aers[index] = metrics.alignment_error_rate(Alignment(gold_alignments[index]), Alignment(alignment))

    return [-np.sum(sent_perplexities), sum(sent_likelihoods)/len(sent_likelihoods), sum(sent_aers)/len(sent_aers)]


def encode_sentence(sentence, dictionary):
    """
    Returns the indexed version of the input sentence
    :param sentence: a string list sentence
    :param dictionary: the dictionary to encode the sentence with
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
    :param dictionary: the dictionary to decode the sentence with
    :return: the decoded list of strings sentence
    """
    decoded = [None] * len(sentence)
    for index, word in enumerate(sentence):
        if word in dictionary:
            decoded[index] = dictionary[word]
        else:
            decoded[index] = 'null'
    return decoded


def produce_output(f_sents, e_sents_orig, nr_f_words, e_sents, e_dict_inv, trans_probs, gold_alignments, performances, filename='results.dat'):
    """
    Produces a translation output for the specified set of sentences
    :param f_sents: the set of french validation sentences
    :param e_sents_orig: not encoded set of english validation sentences
    :param nr_f_words: total number of french corpus word types
    :param e_sents: the set of english validation sentences
    :param e_dict_inv: inverted index-english dictionary to decode alignment
    :param trans_probs: current model
    :param gold_alignments: annotated gold standard alignments
    :param performances: model performances as measured during the last iteration of ME
    :param filename: name of the file to save the result in
    """
    pickle.dump( performances, open("performances.dat", "wb"))

    performances = performances[-1]

    model_output = align_sentences(e_sents, f_sents, trans_probs, nr_f_words)
    f = open(filename, 'w')
    f.write("Model perplexity: {}\n".format(performances[0]))
    f.write("Average sentence likelihood: {}\n".format(performances[1]))
    f.write("Average sentence AER: {}\n\n".format(performances[2]))
    for i, output in enumerate(model_output):
        translation = output[0]
        alignment = output[1]
        f.write("{}\nwas translated as \n{} \ngold standard translation: \n{}\n Predicted alignment: \n{}\n True alignment: \n{}\n\n"
                .format(f_sents[i], translation, e_sents[i], alignment, gold_alignments[i]))
    f.close()


def get_gold_alignments(filename='validation/dev.wa.nonullalign'):
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
    alignments.append(sent_align)
    return alignments


def main():
    start = time.time()

    # Read training data
    e_lines = read_file('training/hansards.36.2.e')
    f_lines = read_file('training/hansards.36.2.f')
    if len(f_lines) != len(e_lines):
        print("ERROR: Training data not aligned properly!")
        quit()

    if TRAIN_LIMIT > 0:
        e_lines = e_lines[0:TRAIN_LIMIT]
        f_lines = f_lines[0:TRAIN_LIMIT]

    train_size = len(f_lines)
    print ("Train set size: {} sentences".format(train_size))

    print ("Limited to {} sentences for debugging".format(train_size))

    e_words = set(word.lower() for line in e_lines for word in line.split())
    e_words = e_words.union(['null'])
    nr_e_words = len(e_words)
    e_dict = wordlist2dict(e_words)
    e_dict_inv = inverse_dict(e_dict)
    f_words = set(word.lower() for line in f_lines for word in line.split())
    nr_f_words = len(f_words)
    f_dict = wordlist2dict(f_words)
    print "The english dictionary contains {} words, the french one {}".format(nr_e_words, nr_f_words)

    e_sents = [[word.lower() for word in np.append('NULL', line.split())] for line in e_lines]
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

    v_e_sents = [[word.lower() for word in np.append('NULL', line.split())] for line in v_e_lines]
    v_f_sents = [[word.lower() for word in line.split()] for line in v_f_lines]

    if INDEX_WORDS:
        v_e_sents_encoded = [encode_sentence(sent, e_dict) for sent in v_e_sents]
        v_f_sents_encoded = [encode_sentence(sent, f_dict) for sent in v_f_sents]
    else:
        v_e_sents_encoded = v_e_sents
        v_f_sents_encoded = v_f_sents

    gold_alignments = get_gold_alignments()

    # Train a IBM 1 model through Expectation Maximization
    trans_probs = {}

    performances = []
    for i in range(0, ITERATIONS):
        f_e_counts = {}

        # E-Step
        for p_index, pair in enumerate(aligned):
            e_sent = pair[0]
            f_sent = pair[1]

            if VERBOSE: print("Compute Normalization")
            s_total = np.zeros(len(f_sent))
            for f_index, f_word in enumerate(f_sent):
                e_sum = 0
                for e_word in e_sent:
                    e_sum += get_trans_prob(f_word, e_word, trans_probs, nr_f_words)
                s_total[f_index] = e_sum

            if VERBOSE: print("Collect Counts")
            for f_index, f_word in enumerate(f_sent):
                for e_word in e_sent:
                    # Update Counts
                    if e_word in f_e_counts:
                        if f_word in f_e_counts[e_word]:
                            current_count = f_e_counts[e_word][f_word]
                            f_e_counts[e_word][f_word] = current_count + get_trans_prob(f_word, e_word, trans_probs, nr_f_words) / s_total[f_index]
                        else:
                            f_e_counts[e_word][f_word] = get_trans_prob(f_word, e_word, trans_probs, nr_f_words) / s_total[f_index]
                    else:
                        f_e_counts[e_word] = {f_word: get_trans_prob(f_word, e_word, trans_probs, nr_f_words) / s_total[f_index]}

            print '\rIteration {}/{} - {:.0%}'.format(i+1, ITERATIONS, float(p_index) / train_size),
            if p_index == train_size: break

        # M-Step
        if VERBOSE: print("Estimate Probabilities")
        for e_word in f_e_counts.keys():
            e_total = sum(f_e_counts[e_word].values())
            for f_word in f_e_counts[e_word].keys():
                if e_word in trans_probs:
                    # if f_e_counts[e_word][f_word] / total[e_word] < 0: print ("Below 0 probability!")
                    trans_probs[e_word][f_word] = f_e_counts[e_word][f_word] / e_total
                    # if VERBOSE: print f_e_counts[e_word][f_word]
                else:
                    # if f_e_counts[e_word][f_word] / total[e_word] < 0: print ("Below 0 probability!")
                    trans_probs[e_word] = {f_word: f_e_counts[e_word][f_word] / e_total}
                    # if VERBOSE: print f_e_counts[e_word][f_word]

        if VERBOSE: print("Sanity check")
        for e_word in trans_probs.keys():
            e_total = sum(f_e_counts[e_word].values())
            if e_total < 0.9: print ("Probability mass for {} is {}".format(e_word, e_total))

        if VERBOSE: print("Evaluate Model")
        [perplexity, avg_likelihood, avg_aer] = evaluate_model(v_f_sents_encoded, v_e_sents, nr_f_words, v_e_sents_encoded, e_dict_inv, trans_probs, gold_alignments)
        performances.append((perplexity, avg_likelihood, avg_aer))
        print ("\nPerplexity: {}".format(perplexity))
        print ("Average likelihood: {}".format(avg_likelihood))
        print ("Average AER: {}\n".format(avg_aer))

    produce_output(v_f_sents_encoded, v_e_sents, nr_f_words, v_e_sents_encoded, e_dict_inv, trans_probs, gold_alignments, performances)
    end = time.time()
    print("Execution time: {}".format(end - start))


if __name__ == "__main__":
    main()
