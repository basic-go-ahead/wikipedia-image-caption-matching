from .DataPreparator import DataPreparator

from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import numpy as np

from sentence_transformers import util


def basic_preprocessing_function(df, calc_tf_idf, title_sentence_embeddings, caption_sentence_embeddings):
    image2target = df.groupby('image_id')['target_id'].apply(list)
    top = 5
    
    if calc_tf_idf:
#         start = 0
        params = [('CAP2CAP', (caption_sentence_embeddings, caption_sentence_embeddings), ('SENSE_FINAL_FILENAME_FINAL_CAPTION', 'CAPTION_VISUAL_AGGR')), \
            ('CAP2TIT', (caption_sentence_embeddings, title_sentence_embeddings), ('SENSE_FINAL_FILENAME_FINAL_CAPTION', 'TITLE_VISUAL_AGGR')), \
            ('TIT2TIT', (title_sentence_embeddings, title_sentence_embeddings), ('SENSE_FINAL_FILENAME_FINAL_TITLE', 'TITLE_VISUAL_AGGR'))]
        
        for image_id, target_ids in tqdm(image2target.iteritems()):
            assert pd.Series(target_ids).is_monotonic_increasing
            query_indices = df[(df['image_id'] == image_id)].index
            
#             end = start + len(target_ids)
        
#             query_indices = df.iloc[start:end].index
            
            for prefix, (left_embeddings, right_embeddings), (sense_column, visual_column) in params:
                similarity_matrix = util.cos_sim(left_embeddings[target_ids], right_embeddings[target_ids])
                similarity_matrix = similarity_matrix.numpy()

                indices = np.take(np.argpartition(similarity_matrix, -top, axis=1), np.arange(-top, 0), axis=1)
                mask = np.zeros_like(similarity_matrix)

                for k in range(len(indices)):
                    mask[k, indices[k]] = 1

                joint_similarity = (df.loc[query_indices, sense_column].values * df.loc[query_indices, visual_column].values)**3

                df.loc[query_indices, prefix + '_normed_TF_IDF_sharing'] = (mask * similarity_matrix)@(df.loc[query_indices, 'normed_TF_IDF'].values * joint_similarity)

#             start = end

        df['MAX_normed_TF_IDF_sharing'] = df[['CAP2CAP_normed_TF_IDF_sharing', 'CAP2TIT_normed_TF_IDF_sharing', 'TIT2TIT_normed_TF_IDF_sharing']].max(axis=1)
        df['CAP2CAP_sum_CAP2TIT'] = df['CAP2CAP_normed_TF_IDF_sharing'] + df['CAP2TIT_normed_TF_IDF_sharing']
        df['CAP2CAP_sum_TIT2TIT'] = df['CAP2CAP_normed_TF_IDF_sharing'] + df['TIT2TIT_normed_TF_IDF_sharing']
        df['CAP2TIT_sum_TIT2TIT'] = df['CAP2TIT_normed_TF_IDF_sharing'] + df['TIT2TIT_normed_TF_IDF_sharing']
        df['CAP2CAP_mul_CAP2TIT'] = df['CAP2CAP_normed_TF_IDF_sharing'] * df['CAP2TIT_normed_TF_IDF_sharing']
        df['CAP2CAP_mul_TIT2TIT'] = df['CAP2CAP_normed_TF_IDF_sharing'] * df['TIT2TIT_normed_TF_IDF_sharing']
        df['CAP2TIT_mul_TIT2TIT'] = df['CAP2TIT_normed_TF_IDF_sharing'] * df['TIT2TIT_normed_TF_IDF_sharing']
        
        df['XVISUAL_normed_TF_IDF'] = df['normed_TF_IDF'] * df['CAPTION_VISUAL_AGGR']**3
        df['XSENSE_normed_TF_IDF'] = df['normed_TF_IDF'] * df['SENSE_FINAL_FILENAME_FINAL_CAPTION']**3
        df['XJOINT_normed_TF_IDF'] = df['normed_TF_IDF'] * df['SENSE_FINAL_FILENAME_FINAL_CAPTION']**3 * df['CAPTION_VISUAL_AGGR']**3
        df['XDIFF_normed_TF_IDF'] = df['CAP2CAP_normed_TF_IDF_sharing'] - df['XJOINT_normed_TF_IDF']
        
        
    base_ranks = ['ENTITY_RANK', 'ENTITY_RANK2', 'NUMERIC_RANK', 'CAPITAL_ACRONYM_RANK', 'CRUDE_ACRONYM_RANK', \
        'FUZZY_CAPTION_RANK', 'FUZZY_TITLE_RANK', 'CAPTION_CROSS_FUZZY_RANK', 'TITLE_CROSS_FUZZY_RANK', 'WEAK_ENTITY_RANK']
    
    for base_rank in base_ranks:
        df['XCAPTION_' + base_rank] = df[base_rank] * df['CAPTION_VISUAL_AGGR']**3 * df['SENSE_FINAL_FILENAME_FINAL_CAPTION']**3
        df['XTITLE_' + base_rank] = df[base_rank] * df['SENSE_FINAL_FILENAME_FINAL_TITLE']**3 * df['TITLE_VISUAL_AGGR']**3
           
    return df