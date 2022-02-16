from collections import defaultdict

import string

import cloudpickle

from ..preprocessing import german_stopwords, general_stopwords


class AcronymContainer:
    """
    Представляет контейнер из грубо составленных акронимов по первым прописным (или просто) буквам слов предложений.
    """
    def process(self, df, columns2process, id_column, admissible_acronym_len=3):
        """
        Обрабатывает стобцы `columns2process` датафрейма `df` с идентифицирующим столбцом `id_column`.
        """
        extracted = []
        for c in columns2process:
            e = df[c].map(AcronymContainer._extract_acronym).tolist()
            extracted.append(e)
            
        combined = list(zip(*extracted))
        
        self.acronym2elements = defaultdict(set)
        self.element2acronyms = defaultdict(list)
        
        for index, element_id in enumerate(df[id_column]):
            for acronym in combined[index]:
                if len(acronym) >= admissible_acronym_len:
                    self.acronym2elements[acronym].add(element_id)
                    self.element2acronyms[element_id].append(acronym)

        return self


    def process_crudely(self, df, columns2process, id_column, admissible_acronym_len=3):
        """
        Обрабатывает стобцы `columns2process` датафрейма `df` с идентифицирующим столбцом `id_column`.
        """
        extracted = []
        for c in columns2process:
            e = df[c].map(AcronymContainer._extract_acronym_crudely).tolist()
            extracted.append(e)
            
        combined = list(zip(*extracted))
        
        self.acronym2elements = defaultdict(set)
        self.element2acronyms = defaultdict(list)
        
        for index, element_id in enumerate(df[id_column]):
            for acronym in combined[index]:
                if len(acronym) >= admissible_acronym_len:
                    self.acronym2elements[acronym].add(element_id)
                    self.element2acronyms[element_id].append(acronym)

        return self
                    
                    
    def elements(self, acronym):
        """
        Возвращает идентификаторы элементов, соответствующие указанному акрониму.
        """
        return self.acronym2elements[acronym]
    
    
    def by(self, element_id):
        return self.element2acronyms[element_id]

                    
    @property
    def values(self):
        """
        Возвращает коллекцию всех акронимов, содержащихся в контейнере.
        """
        return self.acronym2elements.keys()


    def _extract_acronym_crudely(text):
        """
        Извлекает акроним из предоставленного текста.
        """
        r = []
        for w in text.split(' '):
            w = w.strip(string.punctuation)
            w_lowered = w.lower()
            if len(w) and w[0].isalpha() and w_lowered not in general_stopwords and w_lowered not in german_stopwords:
                r.append(w[0].upper())
        return ''.join(r)
    
    
    def _extract_acronym(text):
        """
        Извлекает акроним из предоставленного текста.
        """
        r = []
        for w in text.split(' '):
            w = w.strip(string.punctuation)
            w_lowered = w.lower()
            if len(w) and w[0].isupper() and w_lowered not in general_stopwords and w_lowered not in german_stopwords:
                r.append(w[0])
        return ''.join(r)
    
    
    def save(self, destination):
        """
        Сохраняет контейнер.
        """
        with open(destination, 'wb') as file:
            data = [self.acronym2elements, self.element2acronyms]
            cloudpickle.dump(data, file)
            
            
    def load(self, source):
        """
        Загружает контейнер из файла с предсохраненными данными.
        """
        with open(source, 'rb') as file:
            self.acronym2elements, self.element2acronyms = cloudpickle.load(file)