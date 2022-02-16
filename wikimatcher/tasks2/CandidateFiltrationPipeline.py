import torch
import numpy as np
import pandas as pd

from sentence_transformers import util

from rapidfuzz import process, fuzz

import gc

from ..rankers import CrossElementRanker, SenseRanker2 as SenseRanker, NamedEntityRanker, AcronymRanker
from ..rankers import VisualSenseRanker


def filter_method(image_series, items, top, scorer=fuzz.ratio, return_scores=False):
    d = { }
    
    matrix = process.cdist(image_series.values, items.values, scorer=scorer)
    best_indices = np.take(np.argpartition(matrix, -top, axis=1), np.arange(-top, 0), axis=1)
    
    
    if return_scores:
        for row_index, image_index in enumerate(image_series.index):
            sorted_best_indices = np.argsort(-matrix[row_index, best_indices[row_index]])
            d[image_index] = list(zip(best_indices[row_index][sorted_best_indices], matrix[row_index, best_indices[row_index]][sorted_best_indices]))
            del sorted_best_indices
    else:
        for row_index, image_index in enumerate(image_series.index):
            sorted_best_indices = np.argsort(-matrix[row_index, best_indices[row_index]])
            d[image_index] = best_indices[row_index][sorted_best_indices]
            del sorted_best_indices
        
    del matrix
    gc.collect()
    
    return d


class CandidateFiltrationPipeline:
    """
    Представляет класс, осуществляющий подбор кандидатов на матч.
    """
    def __init__(self, image_ids, image_store, target_store, rank_groups=None):
        """
        Инициализирует пайплайн отбора кандидатов.
        
        `image_ids` — идентификаторы изображений.
        """
        self.images = image_store.df
        self.image_store = image_store
        self.target_store = target_store
        
        self.image_ids = image_ids
        self.image_numbers = image_store.numbers
        self.target_numbers = target_store.numbers
        self.numeric_ranker = CrossElementRanker('NUMERIC_RANK', self.image_numbers, self.target_numbers)
        self.capital_ranker = CrossElementRanker('CAPITAL_RANK', image_store.capitals, target_store.capitals)
        self.sense_ranker = SenseRanker(image_store.embeddings, target_store.embeddings)
        self.entity_ranker = NamedEntityRanker(image_store.words, target_store.snips)
        self.weak_entity_ranker = NamedEntityRanker(image_store.words, target_store.snips, prefix='WEAK_', score_cutoff=35)
        
        self.visual_sense_ranker = VisualSenseRanker()
        
        self.capital_acronym_ranker = AcronymRanker('CAPITAL_ACRONYM_RANK', image_store.short_words, target_store.capital_acronyms)
        self.crude_acronym_ranker = AcronymRanker('CRUDE_ACRONYM_RANK', image_store.short_words, target_store.crude_acronyms)
        
        self.rank_groups = rank_groups
        self.marking = self.rank_groups is not None
        
        self.n_top_entities = 10000
        
    
    def __len__(self):
        """
        Возвращает количество изображений, для которых подбираются кандидаты на матч.
        """
        return len(self.image_ids)
    
    
    def _mark_topk(self, df):
        """
        Помечает TOP-K таргетов относительно ранжирующих полей `rank_fields`.
        """
        df['FLTR_ANY'] = np.uint8(0)
        
        for topk, group in self.rank_groups:
            if 'SENSE' in group[0]:
                filter_field = 'FLTR_' + group[0].strip('_01234')
                df[filter_field] = np.uint8(0)
                
                for rank in group:
                    indices = df[rank].nlargest(topk, keep='first').index
                    df.loc[indices, filter_field] += np.uint8(1)                    
                    df.loc[indices, 'FLTR_ANY'] = np.uint8(1)
            else:
                for rank in group:
                    filter_field = 'FLTR_' + rank
                    df[filter_field] = np.uint8(0)

                    indices = df[rank].nlargest(topk, keep='first').index
                    df.loc[indices, [filter_field, 'FLTR_ANY']] = np.uint8(1)
                
                
    def _join_acronym_ranker(self, ranker, df, image_id):
        df[ranker.rank_name] = np.float32(np.nan)
        
        specific_df = ranker.filter(image_id)
        
        commons = np.intersect1d(specific_df.index.values, df.index.values)
        if len(commons):
            df.loc[commons, ranker.rank_name] = specific_df.loc[commons, ranker.rank_name]

        least = np.setdiff1d(specific_df.index, df.index)
        if len(least):
            df = pd.concat([df, specific_df.loc[least]])
            
        return df
            
            
    def _join_specific_ranker(self, ranker, df, image_elements, target_elements, image_id):
        df[ranker.rank_name] = np.float32(np.nan)

        if image_id in image_elements.image_ids:
            candidates = target_elements.inverse(image_elements.all(image_id))
            specific_df = ranker.apply(image_id, candidates)
            
            commons = np.intersect1d(specific_df.index.values, df.index.values)
            if len(commons):
                df.loc[commons, ranker.rank_name] = specific_df.loc[commons, ranker.rank_name]

            least = np.setdiff1d(specific_df.index, df.index)
            if len(least):
                df = pd.concat([df, specific_df.loc[least]])
                
        return df
    
    
    def _calc_fuzzy_scores(self, image_id, df, prefix, target_ids, pairs2match):
        fuzzy_scores = np.zeros((len(pairs2match), len(target_ids)))
            
        for k, (image_text_field, target_text_field) in enumerate(pairs2match):
            image_text = self.images.loc[image_id, image_text_field]

            if image_text != '':
                matrix = process.cdist([image_text], self.target_store.df.loc[target_ids, target_text_field], scorer=fuzz.token_set_ratio, score_cutoff=0)
                fuzzy_scores[k, :] = matrix

        np.max(fuzzy_scores, axis=0, out=fuzzy_scores[0])
        df.loc[target_ids, prefix + '_CROSS_FUZZY_RANK'] = np.float32(fuzzy_scores[0])
        pass
    
    
            
    def __iter__(self):
        """
        Перечисляет кандидатов из таргетов на матч.
        """
        
        score_cutoff = 0
        
        entity_names = ' ' + pd.Series(self.target_store.snips.snips())
        top_named_entities_indices = filter_method(' ' + self.images.loc[self.image_ids, 'final_filename'], entity_names, top=self.n_top_entities, scorer=fuzz.token_ratio)
        
        
        for image_id in self.image_ids:
            word2snip_candidates = set.union(*map(lambda index: self.target_store.snips.targets(snip_index=index), top_named_entities_indices[image_id]))
            
            df = self.entity_ranker.apply(image_id, word2snip_candidates)
#region Действие специфичных ранкеров
            df = self._join_specific_ranker(self.numeric_ranker, df, self.image_numbers, self.target_numbers, image_id)
            df = self._join_specific_ranker(self.capital_ranker, df, self.image_store.capitals, self.target_store.capitals, image_id)
            df = self._join_acronym_ranker(self.capital_acronym_ranker, df, image_id)
            df = self._join_acronym_ranker(self.crude_acronym_ranker, df, image_id)
#endregion
            matrix = process.cdist(
                [self.images.loc[image_id, 'final_filename']],
                self.target_store.df.loc[df.index, 'final_caption'],
                scorer=fuzz.token_set_ratio,
                score_cutoff=0
            )
            df['FUZZY_CAPTION_RANK'] = np.float32(np.log1p(matrix.ravel()))
        
            matrix = process.cdist(
                [self.images.loc[image_id, 'final_filename']],
                self.target_store.df.loc[df.index, 'final_page_title'],
                scorer=fuzz.token_set_ratio,
                score_cutoff=0
            )
            df['FUZZY_TITLE_RANK'] = np.float32(np.log1p(matrix.ravel()))
#region TOP-эмбеддинги
            target_ids = set()
#             cosines = []
        
            data2sense_ranker = []
    
            for filename_embeddings_container in self.image_store.embeddings:
                image_embeddings = filename_embeddings_container[image_id]
                d = { }
                data2sense_ranker.append(d)
                for embedding_type, embeddins in self.target_store.embeddings.items():
                    c = util.cos_sim(image_embeddings, embeddins)
                    d[embedding_type] = c
#                     cosines.append(c)
                    values, indices = torch.topk(c, k=1250)
                    target_ids.update(indices.numpy().ravel())
                    
            target_ids = np.array(list(target_ids), dtype='int32')
            
            least = np.setdiff1d(target_ids, df.index)
            if len(least):
                df = pd.concat([df, pd.DataFrame(index=least)])
#region WEAK_ENTITY_RANK                
#                 _target_ids = np.array(list(self.target_store.snips.target_ids))
#                 _commons = np.intersect1d(least, _target_ids)
#                 least_df = self.weak_entity_ranker.apply(image_id, _commons)
#                 df.loc[_commons, self.weak_entity_ranker.ranks] = least_df.loc[_commons, self.weak_entity_ranker.ranks]
#endregion
#region
            target_ids512 = set()
            image2text_cosines = { }
    
            image_embeddings = self.image_store.image_embeddings512[image_id]
            for embeddings_type, embeddings in self.target_store.embeddings512.items():
                c = util.cos_sim(image_embeddings, embeddings)
                image2text_cosines[embeddings_type] = c
                values, indices = torch.topk(c, k=1250)
                target_ids512.update(indices.numpy().ravel())

            target_ids512 = np.array(list(target_ids512), dtype='int32')
            least512 = np.setdiff1d(target_ids512, df.index)
            if len(least512):
                df = pd.concat([df, pd.DataFrame(index=least512)])
#endregion
            all_target_ids = np.union1d(target_ids, target_ids512)
            self._calc_fuzzy_scores(image_id, df, 'CAPTION', all_target_ids, CandidateFiltrationPipeline.caption_pairs2match)
            self._calc_fuzzy_scores(image_id, df, 'TITLE', all_target_ids, CandidateFiltrationPipeline.title_pairs2match)
            
            all_least = np.union1d(least, least512)
            df[self.weak_entity_ranker.ranks] = np.float32(np.nan)
                
            if len(all_least):
#region WEAK_ENTITY_RANK                
                _target_ids = np.array(list(self.target_store.snips.target_ids))
                _commons = np.intersect1d(all_least, _target_ids)
                least_df = self.weak_entity_ranker.apply(image_id, _commons)
                df.loc[_commons, self.weak_entity_ranker.ranks] = least_df.loc[_commons, self.weak_entity_ranker.ranks]
#endregion
#endregion
            # error code line
            # df.drop(df[df.isin([0, np.nan]).all(axis=1)].index, inplace=True)
#region Действие комплексного ранкера
            self.sense_ranker.apply(image_id, df, to=CandidateFiltrationPipeline.ranks2sense, data=data2sense_ranker)
            self.visual_sense_ranker.apply(image_id, df, to=CandidateFiltrationPipeline.ranks2visual_sense, data=image2text_cosines)
#endregion
            if self.marking and df.shape[0] > 0:
                self._mark_topk(df)
            
            
            yield image_id, df
            
            
    caption_pairs2match = [
            ('final_filename', 'final_caption'),
            ('undigit_filename', 'undigit_caption'),
            ('undigit_filename_translation', 'caption_translit'),
            ('undigit_filename', 'caption_translit'),
            ('spaced_filename_translit', 'undigit_caption'),
            ('undigit_filename_translation', 'undigit_caption'),
            ('undigit_filename', 'caption_translation'),
            ('undigit_filename_translation', 'caption_translation'),
            ('spaced_filename_translit', 'caption_translation')
        ]
    
    title_pairs2match = [
            ('final_filename', 'final_page_title'),
            ('undigit_filename', 'undigit_page_title'),
            ('spaced_filename_translit', 'undigit_page_title'),
            ('undigit_filename_translation', 'undigit_page_title'),
            ('undigit_filename_translation', 'page_title_translit'),
            ('undigit_filename', 'page_title_translit'),
        ]    
    
    
    ranks2sense = ['WEAK_ENTITY_RANK', 'ENTITY_RANK', 'NUMERIC_RANK', 'CAPITAL_RANK', 'FUZZY_CAPTION_RANK', 'FUZZY_TITLE_RANK', 'CAPTION_CROSS_FUZZY_RANK', 'TITLE_CROSS_FUZZY_RANK']
    ranks2visual_sense = ['WEAK_ENTITY_RANK', 'ENTITY_RANK', 'NUMERIC_RANK', 'FUZZY_CAPTION_RANK', 'FUZZY_TITLE_RANK', 'CAPTION_CROSS_FUZZY_RANK', 'TITLE_CROSS_FUZZY_RANK', 'CAPITAL_ACRONYM_RANK', 'CRUDE_ACRONYM_RANK']