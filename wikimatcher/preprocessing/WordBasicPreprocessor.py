from collections import defaultdict

import pandas as pd

from ..containers import WordContainer
from . import german_stopwords, general_stopwords


class WordBasicPreprocessor:
    """
    Представляет базовый препроцессор извлечения допустимых слов.
    """
    def process(self, df, columns2process, id_column):
        """
        Обрабатывает указанный `pd.DataFrame` для извлечения слов.
        
        `columns2process` — столбцы, который используется в качестве источника для извлечения слов;
        `id_column` — столбец, хранящий идентификаторы элементов.
        """        
        if isinstance(df, pd.DataFrame):
            if isinstance(columns2process, list):
                if len(columns2process) != 1:
                    raise ValueError('BasicWordPreprocessor обрабатывает только один столбец.')
                else:
                    columns2process = columns2process[0]

            container = WordContainer()
            container._element2words = { }
            container._word2elements = defaultdict(set)

            for element_id in df[id_column]:
                words = df.loc[element_id, columns2process].split(' ')

                container._element2words[element_id] = []

                for w in words:
                    w_lowered = w.lower()

                    if w_lowered != 'i':
                        if len(w) <= 1 or w_lowered in general_stopwords or w_lowered in german_stopwords or w[0].isdigit():
                            continue

                    container._element2words[element_id].append(w)
                    container._word2elements[w].add(element_id)

            return container
        else:
            raise NotImplementedError()