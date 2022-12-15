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
