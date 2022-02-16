import torch

import numpy as np

from . import BaseRanker


class VisualSenseRanker(BaseRanker):
    """
    Представляет ранкер, основанный на косинусной близости совместных эмбеддингов.
    """
    def __init__(self, image_embeddings=None, target_embeddings=None):
#         super().__init__(ranks=['VISUAL_AGGR'])
        super().__init__(ranks=None)
        self.image_embeddings = image_embeddings
        self.target_embeddings = target_embeddings


    def _on_applying(self, image_id, items, df, to=None, data=None):
        """
        Выполняет вычислительную работу ранкера.
        """

        if not len(items):
            return

        if data is not None:
            items = torch.from_numpy(items)

        title_columns = []
        caption_columns = []

        if data is not None:
            for embedding_type, cosines in data.items():
                column_name = 'VISUAL_' + embedding_type
                
                if column_name.endswith('CAPTION'):
                    caption_columns.append(column_name)
                elif column_name.endswith('TITLE'):
                    title_columns.append(column_name)
                else:
                    raise RuntimeError('Non-standard embedding_type `{}`'.format(embedding_type))
                
                df[column_name] = torch.index_select(cosines[0], 0, items).numpy()
        else:
            raise RuntimeError('No data!')

                    
        df['CAPTION_VISUAL_AGGR'] = df[caption_columns].max(axis=1)
        df['TITLE_VISUAL_AGGR'] = df[title_columns].max(axis=1)

        power = 3

        if to is not None:
            for basic_rank in to:
                column_name = 'VISUAL_{0}_{1}'.format(basic_rank, power)
                df['CAPTION_' + column_name] = np.float32(df[basic_rank] * df['CAPTION_VISUAL_AGGR']**power)
                df['TITLE_' + column_name] = np.float32(df[basic_rank] * df['TITLE_VISUAL_AGGR']**power)