
import csv, string, re, math
import numpy as np
import neural_network as nn
import activations as avs
from sklearn.metrics import roc_auc_score
from prediction_utils import trn_val_tst

def get_words(text):
    """Removes all punctuation from text, and collapses all whitespace
    characters to a single space"""
    return WS_COLLAPSE_RE.sub(' ', text.lower().translate(PUNCT_REMOVES))

def read_texts_stars(csv_path, maxrows = math.inf):
    """return the 'text' and 'stars' entries (in two lists) from the 
    first maxrows records in csv_path"""
    texts = []
    stars = []
    nrow = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter = ',')
        for row in reader:
            nrow += 1
            if nrow > maxrows: break
            texts.append(row['text'])
            stars.append(row['stars'])
    return texts, stars

## Stores a mapping of words to index positions
class WordVec:
    words_ix = dict()
    def __init__(self, word_list):
        self.word_vec = np.zeros(len(WordVec.words_ix))
        for word in word_list:
            self.word_vec[WordVec.words_ix[word]] += 1

    @classmethod
    def set_word_universe(cls, word_list):
        cls.words_ix = dict(zip(word_list, range(len(word_list))))


DATA_DIR = "../../data"
PUNCT_REMOVES = str.maketrans('', '', string.punctuation)
WS_COLLAPSE_RE = re.compile("\W+")

texts, stars = read_texts_stars(f'{DATA_DIR}/yelp_review.csv', 500)
word_lists = [get_words(text).split() for text in texts]
unique_words = list(set().union(*word_lists))

WordVec.set_word_universe(unique_words)
word_vecs = [WordVec(word_list) for word_list in word_lists]
## word_mat is a matrix with one column per data record and one
## row per feature. Features are counts of how many times each word
## appears.
word_mat = np.array([wv.word_vec for wv in word_vecs])
high_score = np.array([1 if int(rating) >= 4 else 0 for rating in stars])

X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(word_mat, high_score, 
                                                       8/10,1/10, 1/10)

net_shape = [word_mat.shape[1], 20, 7, 5, 1]
activations = [avs.relu, avs.relu, avs.relu, avs.relu, avs.sigmoid]

net = nn.Net(net_shape, activations)
net.train(X = X_trn.T, y = y_trn, 
          iterations = 500, learning_rate = 0.01,
          debug = True)

yhat_trn = net.predict(X_trn.T)
yyhat_trn = np.vstack((y_trn, yhat_trn)).T
auc_trn = roc_auc_score(y_trn, yhat_trn)

yhat_val = net.predict(X_val.T)
yyhat_val = np.vstack((y_val, yhat_val)).T
yyhat_val = yyhat_val[yyhat_val[:,1].argsort()[::-1]]
auc_val = roc_auc_score(y_val, yhat_val)





