import tracemalloc
from functools import wraps
from time import perf_counter


def profiled(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - start
        bar = "█" * min(int(duration * 80), 80)

        print(f"{func.__name__:<16} | {bar} {duration:.4f}s")
        return result

    return wrapper


def traceback_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 1e6:.2f} MB | Peak: {peak / 1e6:.2f} MB")
        return result

    return wrapper
