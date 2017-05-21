import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key is a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Iterate through word index sequences in the test_set        
    for word_id in range(0,len(test_set.get_all_Xlengths())):
        # Initialise dict for probabilities list entry
        pd={}

        current_sequence, current_lengths = test_set.get_item_Xlengths(word_id)
        for word, model in models.items():
            try:
                # Calculate LogL score for each model -> assume score for associated word in models
                score = model.score(current_sequence, current_lengths)
                pd[word] = score
            except:
                # Some sequences/lengths cannot be scored so placeholder kept
                pd[word] = float('-inf')
                continue
        # Append dict of scores for each sequence to probabilities[]
        probabilities.append(pd)
        # Select word with max score in each dict as best guess and add to guesses
        guess = max(pd, key=pd.get)
        guesses.append(guess)
    return probabilities, guesses