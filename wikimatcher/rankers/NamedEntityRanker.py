import math
import numpy as np

from collections import defaultdict

from rapidfuzz import process, fuzz

from . import BaseRanker
from ..containers import WordContainer, TargetNamedEntitySnipContainer
from ..metrics import jaccard


class NamedEntityRanker(BaseRanker):
    """
    Представляет ранкер, основанный на матчинге именованных сущностей.
    """
    def __init__(self, iwc: WordContainer, tnesc: TargetNamedEntitySnipContainer, prefix='', score_cutoff=60):
        self.main_rank = prefix + 'ENTITY_RANK'
        self.secondary_rank = prefix + 'ENTITY_RANK2'
        self.score_cutoff = score_cutoff
        
        super().__init__(ranks=[self.main_rank, self.secondary_rank])
        
        self.iwc = iwc
        self.tnesc = tnesc


    def _on_applying(self, image_id, items, df, to=None, data=None):
        """
        Выполняет вычислительную работу ранкера.
        """
        image_filename_words = self.iwc.by(element_id=image_id)
        
        word_total_scores = defaultdict(float)
        word_target_scores = { }
        
        if not len(image_filename_words):
            df[self.main_rank] = df[self.secondary_rank] = 0.
            return
        
        for k in range(len(image_filename_words)):
            word_total_scores[k] = 0

        for target_id in items:
            target_entities = self.tnesc.snips(target_id=target_id)
            target_entities_translited = self.tnesc.translits(target_id=target_id)

            word_target_scores[target_id] = word_scores = defaultdict(float)
            
            for k in range(len(image_filename_words)):
                word_scores[k] = 0
            
#region
            containers = [target_entities] if not target_entities_translited else [target_entities, target_entities_translited]
#endregion
            for entity_word_container in containers:
        
                m = process.cdist(image_filename_words, entity_word_container, scorer=fuzz.ratio, score_cutoff=self.score_cutoff)

                for i in range(len(image_filename_words)):
                    argmax = np.argmax(m)
                    row_max, col_max = np.unravel_index(argmax, m.shape)                    
                    max_value = m.item(row_max, col_max)
                    
                    if max_value == 0:
                        break

                    image_match_word = image_filename_words[row_max]
                    target_match_word = target_entities[col_max]

                    t = max_value * jaccard(image_match_word, target_match_word)**2
                    current_score = t * math.log(len(image_match_word), 4)

                    if current_score > word_scores[row_max]:
                        word_scores[row_max] = current_score
                        word_total_scores[row_max] += t

                    m[row_max, :] = m[:, col_max] = 0

                if 'm' in locals():
                    del m
            
            del containers
            pass        
        
        mm = max(word_total_scores.values())
                 
        if mm == 0:
            mm = 1
        
        rarity = np.asarray(list(word_total_scores.values())) / mm
        rarity[rarity == 0] = 1e-2
        rarity = np.log(1. / rarity + np.exp(1) - 1)
        
        def calc_scores(target_id):
            word_scores = word_target_scores[target_id].values()
            return sum(np.fromiter(word_scores, dtype='float32') * rarity), \
                sum(word_scores)

        
        pairs = list(map(calc_scores, items))
        
        a, b = zip(*pairs)
            
        df[self.main_rank], df[self.secondary_rank] = np.float32(a), np.float32(b)