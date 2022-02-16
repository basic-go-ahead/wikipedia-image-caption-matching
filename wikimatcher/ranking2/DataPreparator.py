import pandas as pd
import numpy as np
from collections import Counter

import warnings

import gc

import cloudpickle


class DataPreparator:
    
    def __init__(self, images, matchings, part_amount, samples_per_part,
        frequency_format_string,
        part_format_string,
        group_size_format_string=None,
        filename_distances_path=None,
        caption_distances_path=None,
        title_distances_path=None,
        digits_path=None,
        prefinal_title_sentence_embeddings_path=None,
        prefinal_caption_sentence_embeddings_path=None,
        calc_tf_idf=False
    ):
        self.calc_tf_idf = calc_tf_idf
        self.images = images
        self.matchings = matchings
        self.part_amount = part_amount
        self.samples_per_part = samples_per_part
        self.part_format_string = part_format_string
        self.group_size_format_string = group_size_format_string
        
#region Loading Sentence Embeddings
        if prefinal_title_sentence_embeddings_path is not None:
            with open(prefinal_title_sentence_embeddings_path, 'rb') as file:
                self.prefinal_title_sentence_embeddings = cloudpickle.load(file)
        else:
            self.prefinal_title_sentence_embeddings = None
            
        if prefinal_caption_sentence_embeddings_path is not None:
            with open(prefinal_caption_sentence_embeddings_path, 'rb') as file:
                self.prefinal_caption_sentence_embeddings = cloudpickle.load(file)
        else:
            self.prefinal_caption_sentence_embeddings = None       
#endregion
        
        
#region Загрузка расстояний до центров кластеров
        if filename_distances_path is not None:
            with open(filename_distances_path, 'rb') as file:
                self.filename_distances = pd.DataFrame(cloudpickle.load(file), columns=['DFILENAME_' + str(k) for k in range(31)])
                self.filename_distances['image_id'] = self.filename_distances.index
        else:
            self.filename_distances = None
    
    
        if caption_distances_path is not None:
            with open(caption_distances_path, 'rb') as file:
                self.caption_distances = pd.DataFrame(cloudpickle.load(file), columns=['DCAPTION_' + str(k) for k in range(31)])
                self.caption_distances['target_id'] = self.caption_distances.index
        else:
            self.caption_distances = None

            
        if title_distances_path is not None:
            with open(title_distances_path, 'rb') as file:
                self.title_distances = pd.DataFrame(cloudpickle.load(file),columns=['DTITLE_' + str(k) for k in range(31)])
                self.title_distances['target_id'] = self.title_distances.index
        else:
            self.title_distances = None
#endregion
        
#region
        if calc_tf_idf:
            D = images.shape[0]

            frequency = Counter()

            for part_index in range(part_amount):
                counter_file = frequency_format_string.format(part_index // self.samples_per_part + 1, part_index)

                with open(counter_file, 'rb') as file:
                    frequency.update(cloudpickle.load(file))

            self.idf = { }

            for k in frequency:
                self.idf[k] = np.log(D / frequency[k])
#endregion


    def load_parts(self, indices, ranks2use=None, preprocessing_function=None):
        parts = []
        groups = []
#region Загрузка частей
        for part_index in indices:
            part_file = self.part_format_string.format(part_index // self.samples_per_part + 1, part_index)
            df = pd.read_parquet(part_file)
#             df.drop(columns=['CAPITAL_RANK'], inplace=True)       
            df.drop(columns=['FLTR_CAPTION_SENSE_CAPITAL_RANK', 'FLTR_TITLE_SENSE_CAPITAL_RANK'], inplace=True)

            parts.append(df)
            pass
        
            df = pd.concat(parts)
#endregion
            df['target_id'] = df.index

            temp = self.images[['image_id', 'filename_lang', 'filename_lang_p', 'filename_en']]
            df = pd.merge(temp, df, on='image_id')
            temp = self.matchings[['target_id', 'page_title_lang', 'page_title_lang_p', 'page_title_en', 'caption_lang', 'caption_lang_p', 'caption_en']]
            df = pd.merge(temp, df, on='target_id')
            
            df['FN_CAP_LANG'] = np.float32(np.uint8(df['filename_lang'] == df['caption_lang']) * df['filename_lang_p'] * df['caption_lang_p'])
            df['FN_TIT_LANG'] = np.float32(np.uint8(df['filename_lang'] == df['page_title_lang']) * df['filename_lang_p'] * df['page_title_lang_p'])
            df['filename_en'] = df['filename_en'].astype(np.uint8)
            df['page_title_en'] = df['page_title_en'].astype(np.uint8)
            df['caption_en'] = df['caption_en'].astype(np.uint8)
            
#region TF-IDF
        if self.calc_tf_idf:
            filter_columns = [c for c in df.columns if c.startswith('FLTR_')]
            
            for filter_column in filter_columns:
                df.loc[df[filter_column] > 0, filter_column] = np.uint8(1)

            df['FLTR_ANY'] = np.uint8(1)

            for c in filter_columns:
                df['FLTR_ANY'] += df[c]

            image_sizes = df.groupby('image_id').agg(FLTR_ANY_SUM=('FLTR_ANY', 'sum')).reset_index()
            df = pd.merge(df, image_sizes, on='image_id')
            df['TF_IDF'] = df['target_id'].map(self.idf) * df['FLTR_ANY'] / df['FLTR_ANY_SUM']
            
            tf_idf_sums = df.groupby('image_id').agg(TF_IDF_SUM=('TF_IDF', 'sum')).reset_index()
            df = pd.merge(tf_idf_sums, df, on='image_id')
            df['normed_TF_IDF'] = df['TF_IDF'] / df['TF_IDF_SUM']
        else:
            tf_idf_sums = None
#endregion
        if ranks2use is not None:
            df = pd.merge(ranks2use, df, on=['image_id', 'target_id'])

        if self.filename_distances is not None:
            df = pd.merge(df, self.filename_distances, on='image_id')

        if self.caption_distances is not None:
            df = pd.merge(df, self.caption_distances, on='target_id')
            
        if self.title_distances is not None:
            df = pd.merge(df, self.title_distances, on='target_id')

        df.sort_values(by=['image_id', 'target_id'], inplace=True, kind='mergesort')

        if preprocessing_function is not None:
            df = preprocessing_function(df, self.calc_tf_idf, self.prefinal_title_sentence_embeddings, self.prefinal_caption_sentence_embeddings)

        df.sort_values(by=['image_id', 'target_id'], kind='mergesort', inplace=True)
        groups = df.groupby('image_id').size().values.tolist()

        y = df.pop('MATCH')
        image_ids = df.pop('image_id')
        target_ids = df.pop('target_id')

        for c in df.columns:
            if '_id' in c:
                warnings.warn('{} contains `id`.'.format(c))

        if self.calc_tf_idf:
            df.drop(columns=['FLTR_ANY_SUM', 'TF_IDF_SUM', 'TF_IDF'], inplace=True)
            
        df.drop(columns=['filename_lang', 'filename_lang_p', 'page_title_lang', 'page_title_lang_p', 'caption_lang', 'caption_lang_p'], inplace=True)
        
        for column, its_dtype in df.dtypes.iteritems():
            if its_dtype == np.float64:
                df[column] = df[column].astype(np.float32)

        gc.collect()

        return df, y, groups, image_ids, target_ids, tf_idf_sums