import json
import pathlib

from chemtherm import utils


class Elements:
    """
    Class to handle elements data.

    """
    def __init__(self, species_name: str) -> None:
        """
        Get the elements that make up the specified species.

        """
        species_name = utils.strip_species_name(species_name)

        data = self._load_elements()

        self.elements = data[species_name]

    def _load_elements(self) -> dict:
        """
        Load elements database.

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_elements.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)
