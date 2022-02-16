import pandas as pd
import numpy as np

from collections import defaultdict, Counter

import cloudpickle


class TargetNumberContainer:
    """
    Представляет контейнер чисел таргетов.
    """
    def __init__(self, source):
        """
        Инициализирует контейнер чисел таргетов из указанного `pd.DataFrame` или путя с предсохраненными числами.
        """
        if isinstance(source, pd.DataFrame):
            target_numbers_series = source['target'].str.findall(r'\d+').map(set)
            self.number2targets = defaultdict(set)
            self.target2numbers = { }
            counter = Counter()

            for index, target_id in enumerate(source['target_id']):
                number_container = set(map(int, target_numbers_series[index]))

                if number_container:
                    for n in number_container:
                        self.number2targets[n].add(target_id)
                        self.target2numbers[target_id] = number_container

                    counter.update(number_container)
                    
            self.frequencies = { }
            self.weights = { }
            total = sum(counter.values())
            
            shift = np.e - 1
            
            for n in counter:
                frequency = counter[n] / total
                weight = np.log(shift + 1./frequency)
                self.frequencies[n] = frequency, weight 
                self.weights[n] = weight
        elif isinstance(source, str):
            with open(source, 'rb') as file:
                data = cloudpickle.load(file)
                self.number2targets, self.target2numbers, self.frequencies, self.weights = data
        else:
            raise NotImplementedError()
            
            
    def save(self, destination):
        """
        Сохраняет контейнер чисел таргетов.
        """
        with open(destination, 'wb') as file:
            data = [self.number2targets, self.target2numbers, self.frequencies, self.weights]
            cloudpickle.dump(data, file)
             
        
    @property
    def target_ids(self):
        """
        Возвращает идентификаторы таргетов, для которых в контейнере есть наборы чисел.
        """
        return self.target2numbers.keys()
    
    
    def frequency(self, number):
        """
        Возвращает частоту встречаемости числа в контейнере.
        """
        return self.frequencies[number]
    
    
    def weight(self, number):
        """
        Возвращает частоту встречаемости числа в контейнере.
        """
        return self.weights[number]
                
                
    def inverse(self, numbers):
        """
        Возвращает идентификаторы таргетов, отвечающие указанному параметру `numbers`,
        который может быть как числом, так и коллекцией чисел.
        """
        if isinstance(numbers, list) or isinstance(numbers, set):
            return set.union(*map(self.number2targets.__getitem__, numbers))
        elif isinstance(numbers, int):
            return self.number2targets[numbers]
        else:
            raise NotImplementedError()
                
                
    def by(self, target_id):
        """
        Возвращает числа, отвечающие таргету с идентификатором `target_id`.
        """
        return self.target2numbers[target_id]