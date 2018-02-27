
import csv, string, re, math, numpy as np

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

texts, stars = read_texts_stars(f'{DATA_DIR}/yelp_review.csv', 100)
word_lists = [get_words(text).split() for text in texts]
unique_words = list(set().union(*word_lists))

WordVec.set_word_universe(unique_words)
word_vecs = [WordVec(word_list) for word_list in word_lists]
## word_mat is a matrix with one column per data record and one
## row per feature. Features are counts of how many times each word
## appears.
word_mat = np.array([wv.word_vec for wv in word_vecs]).T
