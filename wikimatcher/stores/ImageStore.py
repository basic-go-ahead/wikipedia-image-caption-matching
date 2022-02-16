import torch

import cloudpickle

from ..containers import WordContainer, ImageNumberContainer, ImageCapitalWordContainer

import gc


class ImageStore:
    """
    Представляет хранилище предподсчитанных данных для изображений.
    """
    def __init__(self,
        df,
        words_path=None,
        numbers_path=None,
        embeddings_paths=None,
        capitals_path=None,
        short_words_path=None,
        embeddings512_path=None
    ):
        """
        Инициализирует хранилище путями к предсохраненным данным.
        """
        self.df = df

        if embeddings512_path is not None:
            with open(embeddings512_path, 'rb') as file:
                self._embeddings512 = torch.from_numpy(cloudpickle.load(file).astype('float32'))

        gc.collect()
        
        if words_path is not None:
            self._words = WordContainer()
            self._words.load(words_path)
            
        if short_words_path is not None:
            self._short_words = WordContainer()
            self._short_words.load(short_words_path)

        if numbers_path is not None:
            self._numbers = ImageNumberContainer(numbers_path)
            
        if capitals_path is not None:
            self._capitals = ImageCapitalWordContainer()
            self._capitals.load(capitals_path)
            
        if embeddings_paths is not None:
            self._embeddings = []
            
            for path in embeddings_paths:
                with open(path, 'rb') as file:
                    embeddings_data = torch.from_numpy(cloudpickle.load(file))
                    self._embeddings.append(embeddings_data)
                    
                    
    @property
    def embeddings(self):
        """
        Возвращает коллекцию эмбеддингов различных названий файлой изображений.
        """
        return self._embeddings


    @property
    def image_embeddings512(self):
        """
        Возвращает коллекцию эмбеддингов изображений.
        """
        return self._embeddings512
    
    
    @property
    def capitals(self):
        """
        Возвращает контейнер слов, состоящих из прописных букв.
        """
        return self._capitals
    
    
    @property
    def short_words(self):
        """
        Возвращает контейнер коротких слов.
        """
        return self._short_words


    @property
    def words(self):
        """
        Возвращает контейнер слов.
        """
        return self._words


    @property
    def numbers(self):
        """
        Возвращает контейнер чисел.
        """
        return self._numbers