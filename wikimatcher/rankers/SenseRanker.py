import torch
from sentence_transformers import util

import numpy as np

from . import BaseRanker


class SenseRanker(BaseRanker):
    """
    Представляет ранкер, основанный на косинусной близости эмбеддингов.
    """
    def __init__(self, image_embeddings, target_embeddings, similarity_aggregation='mean'):
        super().__init__(ranks=['SIM_AGGR'])
        self.image_embeddings = image_embeddings
        self.target_embeddings = target_embeddings
        self.similarity_aggregation = similarity_aggregation


    def _on_applying(self, image_id, items, df, to=None, data=None):
        """
        Выполняет вычислительную работу ранкера.
        """
        similarity_columns = []
        
        # if isinstance(items, set):
        #     items = np.array(list(items))

        if not len(items):
            return

        if data is not None:
            items = torch.from_numpy(items)

        k = 0
        
        for image_filename_embeddings in self.image_embeddings:
            for target_embeddings in self.target_embeddings:
                column_name = 'SIM' + str(k)
                if data is not None:
                    df[column_name] = torch.index_select(data[k][0], 0, items).numpy()
                else:
                    df[column_name] = util.cos_sim(image_filename_embeddings[image_id], target_embeddings[items]).numpy().ravel()
                similarity_columns.append(column_name)
                k += 1
    
        if self.similarity_aggregation == 'mean':
            df['SIM_AGGR'] = df[similarity_columns].mean(axis=1)
        elif self.similarity_aggregation == 'median':
            df['SIM_AGGR'] = df[similarity_columns].median(axis=1)
        elif self.similarity_aggregation == 'trimean':
            df['SIM_AGGR'] = 0.5 * df[similarity_columns].median(axis=1) + \
                0.25 * (df[similarity_columns].quantile(0.25, axis=1) + df[similarity_columns].quantile(0.75, axis=1))
        else:
            raise NotImplementedError()

        if to is not None:
            for basic_rank in to:
                for k in range(1, 5):
                    column_name = 'SENSE_{0}_{1}'.format(basic_rank, k)
                    df[column_name] = np.float32(df[basic_rank] * df['SIM_AGGR']**k)