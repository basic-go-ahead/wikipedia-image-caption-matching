import pandas as pd

import cloudpickle


class ImageNumberContainer:
    """
    Представляет контейнер чисел изображений.
    """
    def __init__(self, source):
        """
        Инициализирует контейнер чисел изображений из указанного `pd.DataFrame` или путя с предсохраненными числами.
        """
        if isinstance(source, pd.DataFrame):
            image_numbers = source['pured_filename'].str.findall(r'\d+').map(list)
            self.image2numbers = { }
            self.image2all = { }

            for index, image_id in enumerate(source['id']):
                image_number_strings = image_numbers[index]
                number_container = set(map(int, image_number_strings))
                if number_container:
                    variant = self.get_variant(image_number_strings)
                    self.image2numbers[image_id] = (number_container, ) if variant is None else (number_container, variant)
                    self.image2all[image_id] = set.union(*self.image2numbers[image_id])
        elif isinstance(source, str):
            with open(source, 'rb') as file:
                self.image2numbers, self.image2all = cloudpickle.load(file)
        else:
            raise NotImplementedError()
            
            
    def get_variant(self, number_container: list):
        """
        Возвращает варианты набора чисел, если гипотетическое восстановление возможно.
        """
        is_year = lambda n: len(n) == 4
        
        a = number_container[0]
        variant = set([int(a)])
        repaired = False
        
        for i in range(len(number_container) - 1):
            b = number_container[i + 1]
            
            if is_year(a) and len(b) == 2 and int(a[2:]) < int(b):
                variant.add(int(a[:2] + b))
                repaired = True
            else:
                variant.add(int(b))
                
            a = b
            
        return variant if repaired else None
            
    
    @property
    def image_ids(self):
        """
        Возвращает идентификаторы изображений, для которых в контейнере есть наборы чисел.
        """
        return self.image2numbers.keys()
            
            
    def save(self, destination):
        """
        Сохраняет контейнер чисел изображений.
        """
        with open(destination, 'wb') as file:
            cloudpickle.dump([self.image2numbers, self.image2all], file)
            
            
    def all(self, image_id):
        """
        Возвращает все числа, соответствующие изображению с идентификатором `image_id`,
        в том числе, полученные с помощью восстановления.
        """
        return self.image2all[image_id]


    def by(self, image_id):
        """
        Возвращает числа, содержащиеся в названии файла изображения с идентификатором `image_id`.
        """
        return self.image2numbers[image_id]