from collections import defaultdict

import pandas as pd

from ..containers import WordContainer


class ShortWordPreprocessor:
    """
    Представляет базовый препроцессор извлечения коротких слов.
    """
    def process(self, df, columns2process, id_column):
        """
        Обрабатывает указанный `pd.DataFrame` для извлечения коротких слов.
        
        `columns2process` — столбцы, который используется в качестве источника для извлечения слов;
        `id_column` — столбец, хранящий идентификаторы элементов.
        """        
        if isinstance(df, pd.DataFrame):
            if isinstance(columns2process, list):
                if len(columns2process) != 1:
                    raise ValueError('ShortWordPreprocessor обрабатывает только один столбец.')
                else:
                    columns2process = columns2process[0]

            container = WordContainer()
            container._element2words = { }
            container._word2elements = defaultdict(set)

            for element_id in df[id_column]:
                words = df.loc[element_id, columns2process].split(' ')

                container._element2words[element_id] = []

                for w in words:
                    if 3 <= len(w) <= 6:
                        w = w.upper()
                        container._element2words[element_id].append(w)
                        container._word2elements[w].add(element_id)

            return container
        else:
            raise NotImplementedError()