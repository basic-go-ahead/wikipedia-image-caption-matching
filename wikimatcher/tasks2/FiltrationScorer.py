import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

import gc

from .CandidateFiltrationPipeline import CandidateFiltrationPipeline


class FiltrationScorer:
    """
    Представляет класс оценки качества фильтрации.
    """
    
    base_ranks = ['WEAK_ENTITY_RANK', 'ENTITY_RANK', 'ENTITY_RANK2', 'NUMERIC_RANK', 'CAPITAL_RANK',
       'CAPITAL_ACRONYM_RANK', 'CRUDE_ACRONYM_RANK', 'FUZZY_CAPTION_RANK',
       'FUZZY_TITLE_RANK', 'CAPTION_CROSS_FUZZY_RANK', 'TITLE_CROSS_FUZZY_RANK']
    
#     sense_ranks = ['SENSE_ENTITY_RANK_3', 'SENSE_NUMERIC_RANK_3', 'SENSE_CAPITAL_RANK_3',
#        'SENSE_FUZZY_CAPTION_RANK_3', 'SENSE_FUZZY_TITLE_RANK_3', 'SENSE_CAPTION_CROSS_FUZZY_RANK_3', 'SENSE_TITLE_CROSS_FUZZY_RANK_3']

    sense_ranks = ['CAPTION_SENSE_WEAK_ENTITY_RANK_3', 'CAPTION_SENSE_ENTITY_RANK_3', 'CAPTION_SENSE_NUMERIC_RANK_3', 'CAPTION_SENSE_CAPITAL_RANK_3',
       'CAPTION_SENSE_FUZZY_CAPTION_RANK_3', 'CAPTION_SENSE_FUZZY_TITLE_RANK_3', 'CAPTION_SENSE_CAPTION_CROSS_FUZZY_RANK_3', 'CAPTION_SENSE_TITLE_CROSS_FUZZY_RANK_3'] + [
        'TITLE_SENSE_WEAK_ENTITY_RANK_3', 'TITLE_SENSE_ENTITY_RANK_3', 'TITLE_SENSE_NUMERIC_RANK_3', 'TITLE_SENSE_CAPITAL_RANK_3', 'TITLE_SENSE_FUZZY_CAPTION_RANK_3', 'TITLE_SENSE_FUZZY_TITLE_RANK_3',
        'TITLE_SENSE_CAPTION_CROSS_FUZZY_RANK_3', 'TITLE_SENSE_TITLE_CROSS_FUZZY_RANK_3'] + ['CAPTION_VISUAL_WEAK_ENTITY_RANK_3',
        'TITLE_VISUAL_WEAK_ENTITY_RANK_3',
        'CAPTION_VISUAL_ENTITY_RANK_3',
        'TITLE_VISUAL_ENTITY_RANK_3',
        'CAPTION_VISUAL_NUMERIC_RANK_3',
        'TITLE_VISUAL_NUMERIC_RANK_3',
        'CAPTION_VISUAL_FUZZY_CAPTION_RANK_3',
        'TITLE_VISUAL_FUZZY_CAPTION_RANK_3',
        'CAPTION_VISUAL_FUZZY_TITLE_RANK_3',
        'TITLE_VISUAL_FUZZY_TITLE_RANK_3',
        'CAPTION_VISUAL_CAPTION_CROSS_FUZZY_RANK_3',
        'TITLE_VISUAL_CAPTION_CROSS_FUZZY_RANK_3',
        'CAPTION_VISUAL_TITLE_CROSS_FUZZY_RANK_3',
        'TITLE_VISUAL_TITLE_CROSS_FUZZY_RANK_3',
        'CAPTION_VISUAL_CAPITAL_ACRONYM_RANK_3',
        'TITLE_VISUAL_CAPITAL_ACRONYM_RANK_3',
        'CAPTION_VISUAL_CRUDE_ACRONYM_RANK_3',
        'TITLE_VISUAL_CRUDE_ACRONYM_RANK_3']

    
    def __init__(self, images, matchings, train=True):
        """
        Инициализирует оценщик качества фильтрации.
        """
        self.train = train

        if train:
            self.image2targets = matchings.groupby('image_url')['target_id'].apply(set).values

        self.frequency = Counter()
        
        
    def process(self, image_id, real_target_ids, df):

        for c in FiltrationScorer.base_ranks:
            _min = df[c].min()
            df[c] = (df[c] - _min) / (df[c].max() - _min + 1e-2)
        
        for c in FiltrationScorer.sense_ranks:
            _min, _mean, _std = df[c].min(), df[c].mean(), df[c].std(ddof=0) + 1e-2

            df[c + '_mean-std'] = (df[c] - _mean) / _std
#             df[c + '_var'] = _mean / _std
            df[c] = (df[c] - _min) / (df[c].max() - _min + 1e-2)
        
        df.drop(columns=self.columns2drop, inplace=True)
        
        df['image_id'] = np.int32(image_id)
        df['MATCH'] = np.uint8(0)

        if self.train:
            df.loc[df.index.intersection(real_target_ids), 'MATCH'] = 1
        
        gc.collect()
        
        self.dataframes.append(df)
        self.group_sizes.append(df.shape[0])
    

    def score(self, selections, track_empties=False, verbose=True, measurable=None):
        self.frequency.clear()
        self.dataframes = []
        self.group_sizes = []
        
#         self.columns2drop = ['SENSE_' + rank + '_' + str(k) for rank in CandidateFiltrationPipeline.ranks2sense for k in [1, 2, 4]] + \
#             ['FLTR_SIM_AGGR', 'FLTR_ANY'] + ['FLTR_SIM' + str(k) for k in range(8)] + \
#             ['FLTR_CAPITAL_ACRONYM_RANK', 'FLTR_CRUDE_ACRONYM_RANK']
        
        self.columns2drop = []

        if track_empties:
            self.empties = []
        
        if measurable is None:
            measurable = []

        filter_score = defaultdict(float)
        
        capacity, any_capacity = 0., 0.
        
        n_images = len(selections)
        lengths = np.empty(n_images, dtype='int32')
        
        for k, (image_id, image_selections) in enumerate(tqdm(selections, disable=(not verbose))):
            df = image_selections[image_selections['FLTR_ANY'] > 0].copy() if 'FLTR_ANY' in image_selections.columns else image_selections
            target_ids = set(df.index)
            
            self.frequency.update(target_ids)
            
            real_target_ids = self.image2targets[image_id] if self.train else set()
            # ох уж эти костыли)
            real_length = len(real_target_ids) if self.train else 1

            commons = len(real_target_ids & target_ids)
            lengths.itemset(k, len(target_ids))
            
            if commons or not self.train:
                any_capacity += 1
                capacity += commons / real_length
                
                self.process(image_id, real_target_ids, df)
                
                for f in measurable:
                    filter_score[f] += len(image_selections.index[image_selections[f] > 0].intersection(real_target_ids)) / real_length
            elif track_empties:
                self.empties.append(image_id)
                
        any_capacity /= n_images
        
        for f in measurable:
            filter_score[f] /= n_images

        return { **{
            'capacity': capacity / n_images,
            'any-capacity': any_capacity ,
            'zero-capacity': 1 - any_capacity,
            'max-selection-len': lengths.max(),
            'mean-selection-len': lengths.mean(),
            'q25-selection-len': np.quantile(lengths, 0.25),
            'median-selection-len': np.median(lengths),
            'q75-selection-len': np.quantile(lengths, 0.75),
            'q90-selection-len': np.quantile(lengths, 0.9),
            'n_images': n_images
        }, **dict(filter_score) }



    def just_score(self, selections, verbose=True):

        self.empties = []
        
        filter_score = defaultdict(float)
        
        capacity, any_capacity = 0., 0.
        
        n_images = len(selections)
        lengths = np.empty(n_images, dtype='int32')
        
        for k, (image_id, image_selections) in enumerate(tqdm(selections, disable=(not verbose))):
            df = image_selections[image_selections['FLTR_ANY'] > 0].copy() if 'FLTR_ANY' in image_selections.columns else image_selections
            target_ids = set(df.index)
            
            real_target_ids = self.image2targets[image_id]
            real_length = len(real_target_ids)

            commons = len(real_target_ids & target_ids)
            lengths.itemset(k, len(target_ids))
            
            if commons:
                any_capacity += 1
                capacity += commons / real_length
            else:
                self.empties.append(image_id)
                
        any_capacity /= n_images
        
        return { **{
            'capacity': capacity / n_images,
            'any-capacity': any_capacity ,
            'zero-capacity': 1 - any_capacity,
            'max-selection-len': lengths.max(),
            'mean-selection-len': lengths.mean(),
            'q25-selection-len': np.quantile(lengths, 0.25),
            'median-selection-len': np.median(lengths),
            'q75-selection-len': np.quantile(lengths, 0.75),
            'q90-selection-len': np.quantile(lengths, 0.9),
            'n_images': n_images
        }, **dict(filter_score) }