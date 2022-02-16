def jaccard(a, b):
    """
    Вычисляет меру Жаккара.
    """
    a, b = set(a), set(b)
    c = len(a & b)
    a, b = len(a), len(b)
    
    return c / (a + b - c)


def wjaccard(a, b, weights):
    """
    Вычисляет взвешенную меру Жаккара.
    """
    a, b = set(a), set(b)
        
    commons = a & b
    total_weight = 0.
    
    for e in commons:
        total_weight += weights[e]
    
    a, b, c = len(a), len(b), len(commons)
    
    return total_weight / (a + b - c)