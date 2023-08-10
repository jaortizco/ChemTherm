import json
import pathlib

from chemtherm import utils


class MolecularWeight:
    """
    Class to handle critical constants data.

    """
    def __init__(self, species_name: str) -> None:
        """
        Get the critial constants for the specified species.

        """
        self.species_name = species_name

    def load_molecular_weight(self) -> dict:
        """
        Critical constants.

        Parameters
        ----------
        species : list
            species to get Cp.

        Returns
        -------
        molecular_weight : list
            Molecular weights.

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_molecular_weight.json")

        with open(db_file, "r") as jfile:
            data = json.load(jfile)

        return data[utils.strip_species_name(self.species_name)]
