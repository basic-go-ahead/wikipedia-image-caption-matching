import numpy as np


from .BaseRanker import BaseRanker
from ..metrics import wjaccard

import warnings


class CrossElementRanker(BaseRanker):
    """
    Представляет ранкер, основанный на встречаемости элементов в названиях изорбражений и таргетах.
    """
    def __init__(self, rank_name, image_elements, target_elements):
        """
        Инициализирует ранкер.

        `rank_name` — название ранжирующего значения, которое генерирует ранкер.
        """
        super().__init__(ranks=[rank_name])
        self.image_elements = image_elements
        self.target_elements = target_elements
        self.rank_name = rank_name


    def _on_applying(self, image_id, items, df, to=None, data=None):
        """
        Выполняет вычислительную работу ранкера.
        """
        if to is not None:
            warnings.warn('Параметр `to` не обрабатывается CrossElementRanker.')

        image_elements = self.image_elements.by(image_id=image_id)
        
        ranks = [None] * len(items)
        
        for i, target_id in enumerate(items):
            rank = float('-inf')
            target_elements = self.target_elements.by(target_id=target_id)
            for elements in image_elements:
                v = wjaccard(elements, target_elements, self.target_elements.weights)
                if v > rank: rank = v
            ranks[i] = rank
            
        df[self.rank_name] = np.float32(ranks)