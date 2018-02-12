import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Setup necessary variables.

        best_BIC = None
        best_n_components = None
        logN = math.log(len(self.sequences))
        f = self.X.shape[1]

        try:

            # Consider possible choices for number of model states.
            for n_components in range(self.min_n_components, self.max_n_components+1):

                # Compute the number of free parameters coming from
                # transmat + means + covar matrices + initial distribution
                # https://discussions.udacity.com/t/understanding-better-model-selection/232987/3
                p = n_components * n_components + 2 * f * n_components - 1

                try:
                    # Train the model.
                    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                    # Compute model's BIC score.
                    logL = model.score(self.X, self.lengths)
                    BIC = - 2 * logL + p * logN

                    # print(self.this_word, n_components, -2 * logL, p * logN, BIC)

                    # Remember the best (smallest) results.
                    if best_BIC is None or BIC < best_BIC:
                        best_BIC = BIC
                        best_n_components = n_components

                except:
                    continue

            # Return the best choice of model.
            return self.base_model(best_n_components)

        except:
            # Unless we couldn't find the best choice.
            return None



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Setup necessary variables.

        best_DIC = None
        best_n_components = None
        factor = 1.0 / (len(self.words) - 1.0)

        try:

            # Consider possible choices for number of model states.
            for n_components in range(self.min_n_components, self.max_n_components+1):

                try:
                    # Train the model.
                    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                    # Compute model's score for this word log(P(X(i)).
                    DIC = model.score(self.X, self.lengths)

                    # Adjust by values for all other words - 1/(M-1)SUM(log(P(X(all but i)).
                    for word in self.words:
                        if word != self.this_word:
                            other_X, other_lengths = self.hwords[word]
                            DIC -= factor * model.score(other_X, other_lengths)

                    # Remember the best (largest) results.
                    if best_DIC is None or DIC > best_DIC:
                        best_DIC = DIC
                        best_n_components = n_components

                except:
                    continue

            # Return the best choice of model.
            return self.base_model(best_n_components)

        except:
            # Unless we couldn't find the best choice.
            return None


class TrivialSplitMethod:
    ''' don't actually split – return all indices

    '''
    def split(self, sequences):
        indices = range(0, len(sequences))
        return indices, indices


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Setup necessary variables.
        best_logL = None
        best_n_components = None

        # Create the split method, cannot use more splits than data.
        try:
            split_method = KFold(n_splits=min(3, len(self.sequences)))
        except:
            split_method = TrivialSplitMethod()

        try:
            # Consider possible choices for number of model states.
            for n_components in range(self.min_n_components, self.max_n_components+1):

                try:
                    sum_logL = 0
                    count_logL = 0

                    # Consider different splits, produced by split_method.
                    for train_idx, test_idx in split_method.split(self.sequences):

                        try:

                            # Combine the sequences back.
                            trainX, trainlengths = combine_sequences(train_idx, self.sequences)
                            testX, testlengths = combine_sequences(test_idx, self.sequences)

                            # Use our selected set for training.
                            model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(trainX, trainlengths)

                            # Use sequences left out from training for testing.
                            sum_logL += model.score(testX, testlengths)
                            count_logL += 1

                        except:
                            continue


                    # Compute the average.
                    logL = sum_logL / count_logL

                    # Remember the best (largest) results.
                    if best_logL is not None and logL <= best_logL:
                        continue
                    best_logL = logL
                    best_n_components = n_components

                except:
                    continue

            # Return the best choice of model.
            return self.base_model(best_n_components)

        except:
            # Unless we couldn't find the best choice.
            return None
