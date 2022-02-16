import torch

import cloudpickle

import gc

from ..containers import TargetCapitalWordContainer, AcronymContainer, \
    TargetNamedEntitySnipContainer, TargetNumberContainer


class TargetStore:
    """
    Представляет хранилище предподсчитанных данных для таргетов.
    """
    def __init__(self, df, snips_path=None, numbers_path=None, embeddings_paths=None, capitals_path=None,
        capital_acronyms_path=None, crude_acronyms_path=None, embeddings512_paths=None):
        """
        Инициализирует хранилище путями к предсохраненным данным.
        """
        self.df = df

        if embeddings512_paths is not None:
            self._embeddings512 = {}
            
            for embedding_type, path in embeddings512_paths:
                with open(path, 'rb') as file:
                    embeddings_data = torch.from_numpy(cloudpickle.load(file).astype('float32'))
                    self._embeddings512[embedding_type] = embeddings_data

        gc.collect()

        if embeddings_paths is not None:
            self._embeddings = { }
            
            for embedding_type, path in embeddings_paths:
                with open(path, 'rb') as file:
                    self._embeddings[embedding_type] = torch.from_numpy(cloudpickle.load(file))


        if snips_path is not None:
            self._snips = TargetNamedEntitySnipContainer()
            self._snips.load(snips_path)
            
        if capital_acronyms_path is not None:
            self._capital_acronyms = AcronymContainer()
            self._capital_acronyms.load(capital_acronyms_path)
            
        if crude_acronyms_path is not None:
            self._crude_acronyms = AcronymContainer()
            self._crude_acronyms.load(crude_acronyms_path)

        if numbers_path is not None:
            self._numbers = TargetNumberContainer(numbers_path)
            
        if capitals_path is not None:
            self._capitals = TargetCapitalWordContainer()
            self._capitals.load(capitals_path)

             
    @property
    def embeddings512(self):
        """
        Возвращает словарь коротких эмбеддингов (предназначенных для сравнения с эмбеддингами изображений).
        """
        return self._embeddings512

            
    @property
    def embeddings(self):
        """
        Возвращает коллекцию эмбеддингов для titles и captions.
        """
        return self._embeddings
    
    
    @property
    def crude_acronyms(self):
        """
        Возвращает контейнер акронимов, полученных из прописных букв.
        """
        return self._crude_acronyms
    
    
    @property
    def capital_acronyms(self):
        """
        Возвращает контейнер акронимов, полученных из прописных букв.
        """
        return self._capital_acronyms
    
    
    @property
    def capitals(self):
        """
        Возвращает контейнер слов, состоящих из прописных букв.
        """
        return self._capitals


    @property
    def snips(self):
        """
        Возвращает контейнер слов именнованных сущностей.
        """
        return self._snips


    @property
    def numbers(self):
        """
        Возвращает контейнер чисел.
        """
        return self._numbers