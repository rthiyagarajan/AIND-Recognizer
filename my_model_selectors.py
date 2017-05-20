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
        # @RT this finds the sequences and XLength for this_word - training.get_all_sequences()[this_word]
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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_score = float('+inf') 
        best_model = None
        try:
            for n in range(self.min_n_components, self.max_n_components+1):
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                LogL = model.score(self.X, self.lengths)
                # N datapoints = observations in all frames for the word
                N = len(self.X)
                # no of params = Initial state occupation probabilities + Transition probabilities + Emission probabilities
                # initial prob = n - 1
                # transmat prob = n * (n-1)
                # emission prob = no of means + no of covars for diag = n * d + n * d
                # d = features in each observation
                params = (n * n) + (2 * n * len(self.X[0])) - 1
                BIC_score = ((-2) * LogL) + (params * math.log(N))
                
                if BIC_score < best_score:
                    best_score = BIC_score
                    best_model = model
        except:
            return best_model
        
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

#
#        # TODO implement model selection based on DIC scores
#        
#        # compare each word with all other words for each number of components
#        # n order to get an optimal model for any word, we need to run the model on all other words so that we can calculate the formula
#        # DIC = log(P(original word)) - average(log(P(otherwords)))
#
        # initialise
        max_score = float("-inf")
        max_model = None

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                sum_score = 0.0
                wc = 0.0
                # training model for this_word                
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                # score the other words using this model and sum LogL scores
                for word in self.hwords:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        sum_score += model.score(X, lengths)
                        wc +=1

                DIC_score = model.score(self.X, self.lengths) - (sum_score/wc)
                
                if DIC_score > max_score:
                    max_score = DIC_score
                    max_model = model
            except:
                pass

        return max_model        
        

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # initialise
        best_score = float("-inf") 
        best_model = None
        logs_array = []
        # number of folds cannot be less than number of sequences, default is 3
        try:
            split_method = KFold(min(3,len(self.sequences)))
            for n in range(self.min_n_components, self.max_n_components+1):
                # split CV folds 
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # break loop if issue with GaussianHMM() using this n 
                    try:
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        # get new X and lengths for combined test folds
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        # calculated log likelihood based on test data
                        LogL = model.score(X_test, lengths_test)
                        logs_array.append(LogL)                
                    except:
                        pass
                mean_score = np.mean(logs_array)
                # update best score and train model with all word sequences
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        except:
            return best_model
        
        return best_model


            
        
