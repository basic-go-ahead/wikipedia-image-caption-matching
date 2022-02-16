from transliterate import translit
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS as general_stopwords


german_stopwords = stopwords.words('german')

def transliterate(s: str):
    try:
        return translit(s, reversed=True)
    except:
        return ''


from .WordBasicPreprocessor import WordBasicPreprocessor
from .ShortWordPreprocessor import ShortWordPreprocessor