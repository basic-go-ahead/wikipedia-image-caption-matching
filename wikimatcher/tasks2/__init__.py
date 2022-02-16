from .CandidateFiltrationPipeline import CandidateFiltrationPipeline
from .FiltrationScorer import FiltrationScorer

import pandas as pd
import cloudpickle
import gc


def make_caption_sense_ranks(rank_name):
    return ['CAPTION_SENSE_{0}_{1}'.format(rank_name, k) for k in range(1, 5)]

def make_title_sense_ranks(rank_name):
    return ['TITLE_SENSE_{0}_{1}'.format(rank_name, k) for k in range(1, 5)]


def perform_filtering(image_ids, n_part, image_store, target_store, verbose=True, train=True):
#region Формирование пайплайна фильтрации 
    rank_groups = []
    rank_groups.append((200, ['ENTITY_RANK']))
    rank_groups.append((200, ['ENTITY_RANK2']))
    rank_groups.append((200, make_caption_sense_ranks('ENTITY_RANK')))
    rank_groups.append((200, make_title_sense_ranks('ENTITY_RANK')))
    rank_groups.append((200, ['CAPTION_SENSE_AGGR']))
    rank_groups.append((200, ['TITLE_SENSE_AGGR']))
    rank_groups.append((200, ['NUMERIC_RANK']))
    rank_groups.append((200, make_caption_sense_ranks('NUMERIC_RANK')))
    rank_groups.append((200, make_title_sense_ranks('NUMERIC_RANK')))
#     rank_groups.append((200, ['SIM' + str(k) for k in range(8)]))
    rank_groups.append((200, ['SENSE_UNDIGIT_FILENAME_UNDIGIT_CAPTION', 'SENSE_UNDIGIT_FILENAME_FINAL_CAPTION', 'SENSE_UNDIGIT_FILENAME_UNDIGIT_TITLE', 'SENSE_UNDIGIT_FILENAME_FINAL_TITLE',
                             'SENSE_FINAL_FILENAME_UNDIGIT_CAPTION', 'SENSE_FINAL_FILENAME_FINAL_CAPTION', 'SENSE_FINAL_FILENAME_UNDIGIT_TITLE', 'SENSE_FINAL_FILENAME_FINAL_TITLE']))
    rank_groups.append((200, ['CAPITAL_RANK']))
    rank_groups.append((200, make_caption_sense_ranks('CAPITAL_RANK')))
    rank_groups.append((200, make_title_sense_ranks('CAPITAL_RANK')))
    rank_groups.append((200, ['FUZZY_CAPTION_RANK']))
    rank_groups.append((200, ['FUZZY_TITLE_RANK']))
    rank_groups.append((200, make_caption_sense_ranks('FUZZY_TITLE_RANK')))
    rank_groups.append((200, make_title_sense_ranks('FUZZY_TITLE_RANK')))
    rank_groups.append((200, make_caption_sense_ranks('FUZZY_CAPTION_RANK')))
    rank_groups.append((200, make_title_sense_ranks('FUZZY_CAPTION_RANK')))
    rank_groups.append((100, ['CAPITAL_ACRONYM_RANK']))
    rank_groups.append((100, ['CRUDE_ACRONYM_RANK']))
    rank_groups.append((200, make_caption_sense_ranks('CAPTION_CROSS_FUZZY_RANK')))
    rank_groups.append((200, make_title_sense_ranks('CAPTION_CROSS_FUZZY_RANK')))
    rank_groups.append((200, make_caption_sense_ranks('TITLE_CROSS_FUZZY_RANK')))
    rank_groups.append((200, make_title_sense_ranks('TITLE_CROSS_FUZZY_RANK')))
    
    
    rank_groups.append((200, ['WEAK_ENTITY_RANK']))
    rank_groups.append((200, make_caption_sense_ranks('WEAK_ENTITY_RANK')))
    rank_groups.append((200, make_title_sense_ranks('WEAK_ENTITY_RANK')))
    rank_groups.append((200, ['CAPTION_VISUAL_WEAK_ENTITY_RANK_3', 'TITLE_VISUAL_WEAK_ENTITY_RANK_3']))
    
    
    rank_groups.append((200, ['VISUAL_UNDIGIT_CAPTION',
        'VISUAL_UNDIGIT_TITLE',
        'VISUAL_FINAL_CAPTION',
        'VISUAL_FINAL_TITLE',
        'CAPTION_VISUAL_AGGR',
        'TITLE_VISUAL_AGGR',
        'CAPTION_VISUAL_ENTITY_RANK_3',
        'TITLE_VISUAL_ENTITY_RANK_3',
        'CAPTION_VISUAL_NUMERIC_RANK_3',
        'TITLE_VISUAL_NUMERIC_RANK_3',
        'CAPTION_VISUAL_FUZZY_CAPTION_RANK_3',
        'TITLE_VISUAL_FUZZY_CAPTION_RANK_3',
        'CAPTION_VISUAL_FUZZY_TITLE_RANK_3',
        'TITLE_VISUAL_FUZZY_TITLE_RANK_3',
        'CAPTION_VISUAL_CAPTION_CROSS_FUZZY_RANK_3',
        'TITLE_VISUAL_TITLE_CROSS_FUZZY_RANK_3']))
    
    rank_groups.append((100, ['CAPTION_VISUAL_CAPITAL_ACRONYM_RANK_3',
        'TITLE_VISUAL_CAPITAL_ACRONYM_RANK_3',
        'CAPTION_VISUAL_CRUDE_ACRONYM_RANK_3',
        'TITLE_VISUAL_CRUDE_ACRONYM_RANK_3']))

    pipeline = CandidateFiltrationPipeline(image_ids, image_store, target_store, rank_groups=rank_groups)
#endregion

#region
    measurable = ['FLTR_ENTITY_RANK', 'FLTR_ENTITY_RANK2', 'FLTR_CAPTION_SENSE_ENTITY_RANK', 'FLTR_TITLE_SENSE_ENTITY_RANK', 'FLTR_CAPTION_SENSE_AGGR', 'FLTR_TITLE_SENSE_AGGR', \
      'FLTR_NUMERIC_RANK', 'FLTR_CAPTION_SENSE_NUMERIC_RANK', 'FLTR_TITLE_SENSE_NUMERIC_RANK', \
      'FLTR_CAPITAL_RANK', 'FLTR_CAPTION_SENSE_CAPITAL_RANK', 'FLTR_TITLE_SENSE_CAPITAL_RANK', \
      'FLTR_FUZZY_CAPTION_RANK', 'FLTR_CAPTION_SENSE_FUZZY_CAPTION_RANK', 'FLTR_TITLE_SENSE_FUZZY_CAPTION_RANK', \
      'FLTR_FUZZY_TITLE_RANK', 'FLTR_CAPTION_SENSE_FUZZY_TITLE_RANK', 'FLTR_TITLE_SENSE_FUZZY_TITLE_RANK', \
      'FLTR_CAPITAL_ACRONYM_RANK', 'FLTR_CRUDE_ACRONYM_RANK'] + ['FLTR_VISUAL_UNDIGIT_CAPTION',
        'FLTR_VISUAL_UNDIGIT_TITLE',
        'FLTR_VISUAL_FINAL_CAPTION',
        'FLTR_VISUAL_FINAL_TITLE',
        'FLTR_CAPTION_VISUAL_AGGR',
        'FLTR_TITLE_VISUAL_AGGR',
        'FLTR_CAPTION_VISUAL_ENTITY_RANK_3',
        'FLTR_TITLE_VISUAL_ENTITY_RANK_3',
        'FLTR_CAPTION_VISUAL_NUMERIC_RANK_3',
        'FLTR_TITLE_VISUAL_NUMERIC_RANK_3',
        'FLTR_CAPTION_VISUAL_FUZZY_CAPTION_RANK_3',
        'FLTR_TITLE_VISUAL_FUZZY_CAPTION_RANK_3',
        'FLTR_CAPTION_VISUAL_FUZZY_TITLE_RANK_3',
        'FLTR_TITLE_VISUAL_FUZZY_TITLE_RANK_3',
        'FLTR_CAPTION_VISUAL_CAPTION_CROSS_FUZZY_RANK_3',
        'FLTR_TITLE_VISUAL_TITLE_CROSS_FUZZY_RANK_3'] + ['FLTR_CAPTION_VISUAL_CAPITAL_ACRONYM_RANK_3',
        'FLTR_TITLE_VISUAL_CAPITAL_ACRONYM_RANK_3',
        'FLTR_CAPTION_VISUAL_CRUDE_ACRONYM_RANK_3',
        'FLTR_TITLE_VISUAL_CRUDE_ACRONYM_RANK_3'] + ['FLTR_WEAK_ENTITY_RANK', 'FLTR_CAPTION_VISUAL_WEAK_ENTITY_RANK_3', 'FLTR_TITLE_VISUAL_WEAK_ENTITY_RANK_3', 'FLTR_CAPTION_SENSE_WEAK_ENTITY_RANK', 'FLTR_TITLE_SENSE_WEAK_ENTITY_RANK']

    monitor = FiltrationScorer(image_store.df, target_store.df, train=train)
    scores = monitor.score(pipeline, measurable=measurable, verbose=verbose)
    
    concated = pd.concat(monitor.dataframes)
 
    to_drop = ['WEAK_ENTITY_RANK2', 'FLTR_ENTITY_RANK2', 'FLTR_ANY', 'FLTR_SENSE_UNDIGIT_FILENAME_UNDIGIT_CAPTION', 'FLTR_CAPITAL_RANK', 'FLTR_NUMERIC_RANK', 'FLTR_FUZZY_CAPTION_RANK', 'FLTR_FUZZY_TITLE_RANK', 'FLTR_WEAK_ENTITY_RANK']
    
    for c in concated.columns:
        if ('SENSE' in c and (c.endswith('1') or c.endswith('2') or c.endswith('4'))) or (c.startswith('FL') and c.endswith('AGGR')):
            to_drop.append(c)
    
    concated.drop(columns=to_drop, inplace=True)
    
    
    concated.to_parquet('part-{0:02d}.parquet'.format(n_part))
       
    with open('group-sizes-{0:02d}.pickle'.format(n_part), 'wb') as file:
        cloudpickle.dump(monitor.group_sizes, file)
        
    with open('frequency-{0:02d}.pickle'.format(n_part), 'wb') as file:
        cloudpickle.dump(monitor.frequency, file)
        
    if hasattr(monitor, 'empties'):
        with open('empties-{0:02d}.pickle'.format(n_part), 'wb') as file:
            cloudpickle.dump(monitor.empties, file)
        del monitor.empties
        
    del monitor.dataframes
    del monitor.group_sizes
    
    gc.collect()

#endregion
    return scores