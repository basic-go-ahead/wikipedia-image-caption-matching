import pandas as pd

from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

from collections import defaultdict

import cloudpickle


class ImageCapitalWordContainer:
    """
    Представляет контейнер слов изображений, состоящих из прописных букв.
    """
    def process(self, source: pd.DataFrame):
        """
        Обрабатывает входной `pd.DataFrame`.
        """
        if isinstance(source, pd.DataFrame):
            image_words = source['spaced_undigit_filename'].str.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b').map(list)
            self.image2words = defaultdict(set)
            
            german_stopwords = stopwords.words('german')

            for index, image_id in enumerate(source['id']):
                image_words_at = image_words[index]
                for w in image_words_at:
                    for o in w.split(' '):
                        o_lowered = o.lower()
                        if len(o) > 2 and o_lowered not in STOPWORDS and o_lowered not in german_stopwords and \
                            len(set(o) - set('IXV')):
                            self.image2words[image_id].add(o)
        else:
            raise NotImplementedError()
            
            
    @property
    def image_ids(self):
        """
        Возвращает идентификаторы изображений, для которых в контейнере есть элементы.
        """
        return self.image2words.keys()
            
            
    def save(self, destination):
        """
        Сохраняет контейнер.
        """
        with open(destination, 'wb') as file:
            cloudpickle.dump(self.image2words, file)
            
            
    def load(self, destination):
        """
        Загружает контейнер.
        """
        with open(destination, 'rb') as file:
            self.image2words = cloudpickle.load(file)
            
            
    def all(self, image_id):
        """
        Возвращает все слова (в том числе восстановленные), соответствующие изображению с идентификатором `image_id`.
        """
        return self.image2words[image_id]
    

    def by(self, image_id):
        """
        Возвращает слова, содержащиеся в названии файла изображения с идентификатором `image_id`.
        """
        return self.image2words[image_id]
    
    
    def __len__(self):
        """
        Возвращает количество изображений, для которых есть слова в контейнере.
        """
        return len(self.image2words)
    
    
    def total(self):
        """
        Возвращает общее число слов в контейнере.
        """
        return sum(map(len, self.image2words.values()))