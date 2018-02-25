import warnings
from asl_data import SinglesData
from math import inf
import operator

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Setup necessary variables.
    probabilities = []
    guesses = []

    # Iterate over given words.
    challenges = sorted(test_set.get_all_Xlengths().keys())

    for challenge in challenges:

        # Get the challenge data.
        X, lengths = test_set.get_item_Xlengths(challenge)

        # Calculate the probability.
        def calculate(model):
            try:
                return model.score(X, lengths)
            # Sometimes we get mysterious exceptions.
            except:
                return -inf

        p = {word: calculate(model) for word, model in models.items()}
        probabilities.append(p)

        best_guess = max(p.items(), key=operator.itemgetter(1))
        guesses.append(best_guess[0])

#    print("Errors encountered for models of: {}".format(error_words))

    return probabilities, guesses
