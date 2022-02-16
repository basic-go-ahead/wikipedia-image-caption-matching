import pandas as pd

from collections.abc import Iterable
import abc


class BaseRanker(abc.ABC):
    """
    Представляет интерфейс базового ранкера.
    """
    def __init__(self, ranks):
        """
        Инициализирует ранкер коллекцией имён вычисляемых ранков.
        """
        self._ranks = ranks


    @abc.abstractmethod
    def _on_applying(self, image_id, items, df, to=None, data=None):
        """
        Выполняет вычислительную работу ранкера.

        `to` — список колонок `df`, на основе которых генерируются дополнительные ранкеры.
        """
        return


    @property
    def ranks(self):
        """
        Возвращает наименования вычисляемых ранков.
        """
        return self._ranks
    
        
    def apply(self, image_id, targets, to=None, data=None, reject_zero_rank_candidates=False):
        """
        Применяет ранкер к изображению с идентификатором `image_id` и указанным таргетам.
        """
        if isinstance(targets, pd.DataFrame):
            # items = targets.index
            df = targets
            if reject_zero_rank_candidates:
                raise ValueError('Параметр reject_zero_rank_candidates=True не сочетается с входом типа pd.DataFrame.')
        elif isinstance(targets, Iterable):
            # items = set(targets)
            # df = pd.DataFrame(index=items)
            df = pd.DataFrame(index=set(targets))
        else:
            raise NotImplementedError()

        items = df.index.values

        self._on_applying(image_id, items, df, to, data)
        
        if not reject_zero_rank_candidates:
            return df
        else:
            raise NotImplementedError()