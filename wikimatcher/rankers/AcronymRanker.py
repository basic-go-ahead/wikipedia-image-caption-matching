import math
import numpy as np

from . import BaseRanker
from ..containers import WordContainer, AcronymContainer


class AcronymRanker(BaseRanker):
    """
    Представляет ранкер, основанный на вхождении акронимов.
    """
    def __init__(self, rank_name2produce, words: WordContainer, acronyms: AcronymContainer):
        super().__init__(ranks=[rank_name2produce])
        
        self.words = words
        self.acronyms = acronyms
        self.rank_name = rank_name2produce
        
        
    def filter(self, image_id):
        """
        Выполняет фильтрацию и ранжирование.
        """
        words = self.words.by(element_id=image_id)
        
        target_ids = set()
        
        for w in words:
            for a in self.acronyms.values:
                if w in a:
                    target_ids.update(self.acronyms.elements(a))
                    
        return self.apply(image_id, target_ids)


    def _on_applying(self, image_id, items, df, to=None, data=None):
        """
        Выполняет вычислительную работу ранкера.
        """
        words = self.words.by(element_id=image_id)
        ranks = []
        
        for target_id in items:
            r = 0.
            for w in words:
                # Разные веса для page_title и caption
                for a in self.acronyms.by(target_id):
                    i = a.find(w)
                    if i >= 0:
                        r += math.log(len(w), 3) / len(a)

            ranks.append(r)
        
        df[self.rank_name] = np.float32(ranks)