import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_performances():
    performances = pickle.load( open("performances.dat", "rb"))
    x = np.arange(1, len(performances)+1, 1)
    perplexity = [tuple[0] for tuple in performances]
    likelihood = [tuple[1] for tuple in performances]
    aer = [tuple[2] for tuple in performances]



    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(x, perplexity)
    axarr[0].set_title('Translation Perplexity')
    axarr[1].plot(x, likelihood)
    axarr[1].set_title('Translation Likelihood')
    axarr[2].plot(x, aer)
    axarr[2].set_title('Alignment Error Rate')
    plt.xticks(x)
    plt.show()

if __name__ == "__main__":
    plot_performances()
