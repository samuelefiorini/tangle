"""External utilities module.

Some of the function is this module are imported from adenine:
https://github.com/slipguru/adenine/blob/master/adenine/utils/extra.py
"""
import time


def sec_to_time(seconds):
    """Transform seconds into a formatted time string.
    Parameters
    -----------
    seconds : int
        Seconds to be transformed.
    Returns
    -----------
    time : string
        A well formatted time string.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def timed(function):
    """Decorator that measures wall time of the decored function."""
    def timed_function(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        print("[{}] - Elapsed time : {} s"
              .format(function.__name__, sec_to_time(time.time() - t0)))
        return result
    return timed_function
