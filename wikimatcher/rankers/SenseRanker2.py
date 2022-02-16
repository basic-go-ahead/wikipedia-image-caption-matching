import torch

import numpy as np

from . import BaseRanker

class SenseRanker2(BaseRanker):
    """
    Представляет ранкер, основанный на косинусной близости эмбеддингов.
    """
    def __init__(self, image_embeddings, target_embeddings):
        super().__init__(ranks=[None])
        self.image_embeddings = image_embeddings
        self.target_embeddings = target_embeddings
        
        
    def _handle(self, df, prefix, data, items, caption_columns, title_columns):
        prefix = 'SENSE_' + prefix + '_'
        for embedding_type, cosines in data.items():
            column_name = prefix + embedding_type

            if column_name.endswith('CAPTION'):
                caption_columns.append(column_name)
            elif column_name.endswith('TITLE'):
                title_columns.append(column_name)
            else:
                raise RuntimeError('Non-standard embedding_type `{}`'.format(embedding_type))

            df[column_name] = torch.index_select(cosines[0], 0, items).numpy()


    def _on_applying(self, image_id, items, df, to=None, data=None):
        """
        Выполняет вычислительную работу ранкера.
        """
        similarity_columns = []
        
        if not len(items):
            return

        if data is not None:
            items = torch.from_numpy(items)
            
        title_columns = []
        caption_columns = []

        if data is not None:
            undigit_fn_data, prefinal_fn_data = data
            self._handle(df, 'UNDIGIT_FILENAME', undigit_fn_data, items, caption_columns, title_columns)
            self._handle(df, 'FINAL_FILENAME', prefinal_fn_data, items, caption_columns, title_columns)
        else:
            raise RuntimeError('No data!')

        df['CAPTION_SENSE_AGGR'] = df[caption_columns].mean(axis=1)
        df['TITLE_SENSE_AGGR'] = df[title_columns].mean(axis=1)

        if to is not None:
            for basic_rank in to:
                for k in range(1, 5):
                    column_name = 'SENSE_{0}_{1}'.format(basic_rank, k)
                    df['CAPTION_' + column_name] = np.float32(df[basic_rank] * df['CAPTION_SENSE_AGGR']**k)
                    df['TITLE_' + column_name] = np.float32(df[basic_rank] * df['TITLE_SENSE_AGGR']**k)