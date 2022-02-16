import pandas as pd

from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

import cloudpickle


class WordContainer:
    """
    Представляет контейнер слов из названий элементов.
    """
    def save(self, destination):
        """
        Сохраняет контейнер слов из названий элементов.
        """
        with open(destination, 'wb') as file:
            cloudpickle.dump([self._element2words, self._word2elements], file)
            
            
    def load(self, source):
        """
        Загружает контейнер из файла с предсохраненными словами.
        """
        with open(source, 'rb') as file:
            self._element2words, self._word2elements = cloudpickle.load(file)


    def by(self, element_id):
        """
        Возвращает слова, содержащиеся в названии элемента с идентификатором `element_id`.
        """
        return self._element2words[element_id]


    @property
    def element_ids(self):
        """
        Возвращает коллекцию всех идентификаторов элементов, содержащихся в контейнере.
        """
        return self._element2words.keys()

    
    def elements(self, word):
        """
        Возвращает идентификаторы элементов, содержащие указанное слово.
        """
        return self._word2elements[word]
    
    
    def total(self):
        """
        Возвращает общее число слов в контейнере.
        """
        return sum(map(len, self._element2words.values()))