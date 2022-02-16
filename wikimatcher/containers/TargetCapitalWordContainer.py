import pandas as pd
import numpy as np

from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

from collections import defaultdict, Counter

import cloudpickle


class TargetCapitalWordContainer:
    """
    Представляет контейнер слов таргетов, состоящих из прописных букв.
    """
    def process(self, source):
        """
        Обрабатывает входной `pd.DataFrame`.
        """
        if isinstance(source, pd.DataFrame):
            all_words = source['target'].str.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b').map(list)
            
            self.target2words = defaultdict(set)
            self.word2targets = defaultdict(set)
            counter = Counter()
            
            german_stopwords = stopwords.words('german')

            for index, target_id in enumerate(source['target_id']):
                target_words = all_words[index]
                for w in target_words:
                    for o in w.split(' '):
                        o_lowered = o.lower()
                        if len(o) > 2 and o_lowered not in STOPWORDS and o_lowered not in german_stopwords and \
                            len(set(o) - set('IXV')) and o != 'SEP':
                            self.target2words[target_id].add(o)
                            sorted_o = ''.join(sorted(o))
                            self.word2targets[o].add(target_id)
                            counter.update([o])
                            
            self.frequencies = { }
            self.weights = { }
            total = sum(counter.values())
            
            shift = np.e - 1
            
            for n in counter:
                frequency = counter[n] / total
                weight = np.log(shift + 1./frequency)
                self.frequencies[n] = frequency, weight 
                self.weights[n] = weight
        else:
            raise NotImplementedError()
            
            
    def save(self, destination):
        """
        Сохраняет контейнер.
        """
        with open(destination, 'wb') as file:
            data = [self.word2targets, self.target2words, self.frequencies, self.weights]
            cloudpickle.dump(data, file)
            
            
    def load(self, destination):
        """
        Загружает контейнер из файла.
        """
        with open(destination, 'rb') as file:
            self.word2targets, self.target2words, self.frequencies, self.weights = cloudpickle.load(file)
             
        
    @property
    def target_ids(self):
        """
        Возвращает идентификаторы таргетов, для которых в контейнере есть наборы чисел.
        """
        return self.target2words.keys()
    
    
    def frequency(self, word):
        """
        Возвращает частоту встречаемости слова в контейнере.
        """
        return self.frequencies[word]
    
    
    def weight(self, word):
        """
        Возвращает частоту встречаемости слова в контейнере.
        """
        return self.weights[word]
                
                
    def inverse(self, words):
        """
        Возвращает идентификаторы таргетов, отвечающие указанному параметру `words`,
        который может быть как числом, так и коллекцией чисел.
        """
        if isinstance(words, list) or isinstance(words, set):
            return set.union(*map(self.word2targets.__getitem__, words))
        elif isinstance(words, str):
            return self.word2targets[words]
        else:
            raise NotImplementedError()
                
                
    def by(self, target_id):
        """
        Возвращает слова, отвечающие таргету с идентификатором `target_id`.
        """
        return self.target2words[target_id]