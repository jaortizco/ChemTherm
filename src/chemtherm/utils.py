"""
Utilities module.

"""

import cProfile
import pstats
import time
from pstats import SortKey

import numpy as np


def strip_species_name(species_name: str) -> str:
    """
    Remove a species phase identifiers like gas (g), liquid (l)
    and solid (s).

    """
    return species_name.replace("(g)", "").replace("(l)",
                                                   "").replace("(s)", "")


def remove_duplicates(mylist: list) -> list:
    """
    Remove duplicates in a list.

    """
    # A dictionary approach is used
    return list(dict.fromkeys(mylist))


def list2dict(lst):
    """
    Convert a list into a dictionary.

    The first element of the list will be the first key of the dictonary and
    the second element of the list will be the first value of the dictionary.

    Parameters
    ----------
    lst : list
        List to be converted into a dictionary.

    Returns
    -------
    dict
        Dictionary.

    """
    mydict = {}
    for index, item in enumerate(lst):
        if index % 2 == 0:
            mydict[item] = lst[index + 1]

    return mydict


def print_runtime(run_time):
    """
    Print the time that a script took to run.

    """
    if run_time < 60:
        print(f"\nScript took {round(run_time):.0f} second(s) to run.")

    elif 60 <= run_time < 3600:
        minutes = int(run_time/60)
        seconds = round(np.fmod(run_time, 60))
        print(
            f"\nScript took {minutes:.0f} minute(s) "
            + f"and {seconds:.0f} second(s) to run."
        )

    else:
        hours = int(run_time/3600)
        minutes = int(np.fmod(run_time, 3600)/60)
        seconds = round(np.fmod(np.fmod(run_time, 3600), 60))
        print(
            f"\nScript took {hours:.0f} hours(s), "
            + f"{minutes:.0f} minute(s) "
            + f"and {seconds:.0f} second(s) to run."
        )


def timer(function):

    def wrapper():
        start_time = time.perf_counter()

        function()

        run_time = time.perf_counter() - start_time
        print_runtime(run_time)

    return wrapper


def profiler(sort_key: str = "tottime", amount: int = 30):
    """
    Run and profile function.

    To be used as decorator.

    """
    SORT_TYPES = {
        "ncalls": SortKey.CALLS,
        "tottime": SortKey.TIME,
        "cumtime": SortKey.CUMULATIVE,
        "percall": SortKey.PCALLS,
        "filename": SortKey.FILENAME,
        "name": SortKey.NAME,
    }

    def decorator_wrapper(function):

        def function_wrapper(*args, **kwargs):

            def function_profiler():
                profiler = cProfile.Profile()

                profiler.enable()
                function(*args, **kwargs)
                profiler.disable()

                stats = pstats.Stats(profiler)
                stats.strip_dirs()
                stats.sort_stats(SORT_TYPES[sort_key]).print_stats(amount)

            return function_profiler()

        return function_wrapper

    return decorator_wrapper
