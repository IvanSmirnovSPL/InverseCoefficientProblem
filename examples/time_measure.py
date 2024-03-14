from typing import Callable


def time_measure(func: Callable, *args, **kwargs):
    import time
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        rez = func(*args, **kwargs)
        print(f'Elapsed time: {time.perf_counter() - start}')
        return rez
    return wrapper