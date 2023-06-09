# Performance utilities
import time
from decorator import decorator
import logging

logger = logging.getLogger(__name__)


@decorator
def measure_performance(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    logger.debug("Time taken by", func.__name__, ":", end - start, "seconds")
    return result
