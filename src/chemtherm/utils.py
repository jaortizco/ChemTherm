"""
Utilities module.

"""


def strip_species_name(species_name: str) -> str:
    """
    Remove a species phase identifiers like gas (g), liquid (l)
    and solid (s).

    """
    return species_name.replace(
        "(g)", "").replace(
            "(l)", "").replace(
                "(s)", "")


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
            mydict[item] = lst[index+1]

    return mydict
