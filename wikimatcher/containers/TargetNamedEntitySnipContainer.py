from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from nltk.corpus import stopwords
import string

import cloudpickle

from ..preprocessing import transliterate


class TargetNamedEntitySnipContainer:
    """
    Представляет контейнер частичек-слов именнованных сущностей таргетов. 
    """
    def process(self, sources):
        """
        Обрабатывает входные файлы, содержащие информацию об именнованных сущностях.
        """
        self.snip2targets = { }
        self.target2snips = { }
        
        german_stopwords = stopwords.words('german')

        for source in sources:
            with open(source, 'rb') as file:
                source_entities = cloudpickle.load(file)
#region Очистка именнованных сущностей и формирование частичек                
            for index in source_entities:
                d = source_entities[index]
                d = d['misc'] + d['per'] + d['org'] + d['loc']

                for e in d:
                    for x in e.split(' '):
                        x = x.strip().strip(string.punctuation)

                        x_lowered = x.lower()

                        for apostrof in ['`', '\'']:
                            if apostrof in x:
                                x = x.split(apostrof)[1]

                        if x_lowered != 'i':
                            if len(x) <= 1 or x[0].islower() or x[0].isdigit() or \
                                x_lowered in STOPWORDS or x_lowered in german_stopwords or x == 'CLS':
                                continue

                        if x in self.snip2targets:
                            self.snip2targets[x].append(index)
                        else:
                            self.snip2targets[x] = [index]

                        if index in self.target2snips:
                            count_dict = self.target2snips[index]

                            if x in count_dict:
                                count_dict[x] += 1
                            else:
                                count_dict[x] = 1
                        else:
                            self.target2snips[index] = { x: 1 }                            
#endregion
        self.snip2targets = { k: set(v) for k, v in self.snip2targets.items() }
        self.target2translits = { k: self._transliterates_snips(v) for k, v in self.target2snips.items()}        
        self.target2snips = { k: list(v) for k, v in self.target2snips.items() }
        
        self.snip_index2targets = []
            
        for snip in self.snip2targets:
            self.snip_index2targets.append(self.snip2targets[snip])
            
        self.snip_index2snip = list(self.snip2targets.keys())
        
        
    def _transliterates_snips(self, snips):
        """
        Выполняет транслитерацию указанных частичек.
        """
        transliterated = set(transliterate(o) for o in snips)
                
        if '' in transliterated:
            transliterated.remove('')

        return list(transliterated) if transliterated else []
    
    
    def translits(self, target_id):
        """
        Возвращает транслитерацию частичек по идентфикатору таргета.
        """
        return self.target2translits[target_id]


    @property
    def target_ids(self):
        return self.target2snips.keys()
            

    def targets(self, nothing=None, snip=None, snip_index=None):
        """
        Возвращает идентификаторы таргетов по частичке именованной сущности или её индексу.
        """
        if snip_index is not None:
            return self.snip_index2targets[snip_index]
        elif snip is not None:
            return self.snip2targets[snip]
        else:
            raise ValueError('Необходимо указывать явно параметр.')
            
            
    def snip_at(self, snip_index):
        """
        Возвращает частичку по её индексу.
        """
        return self.snip_index2snip[snip_index]
            
            
    def snips(self, target_id=None):
        """
        Возвращает частички по идентификатору таргета.
        """
        if target_id is not None:
            return self.target2snips[target_id]
        else:
            return self.snip_index2snip
        
            
    def contains(self, snip):
        """
        Возвращает значение, указывающее, что частичка содержится в коллекции.
        """
        return snip in self.snip2targets
    
    
    def __len__(self):
        """
        Возвращает общее число частичек в коллекции.
        """
        return len(self.snip2targets)
    
    
    def save(self, destination):
        """
        Сохраняет контейнер.
        """
        with open(destination, 'wb') as file:
            data = [self.snip2targets, self.target2snips, self.target2translits, self.snip_index2targets, self.snip_index2snip]
            cloudpickle.dump(data, file)
            
            
    def load(self, source):
        """
        Загружает контейнер из файла с предсохраненными данными.
        """
        with open(source, 'rb') as file:
            self.snip2targets, self.target2snips, self.target2translits, \
            self.snip_index2targets, self.snip_index2snip = cloudpickle.load(file)