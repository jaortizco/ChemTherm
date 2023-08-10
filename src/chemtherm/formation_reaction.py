import json
import pathlib


class FormationReaction:
    """
    Class to handle formation reaction data.

    """

    def __init__(self, species_name: str):
        """
        Get the formation reactions for the specified species.

        """
        data = self._load_formation_reaction()

        self.species = data[species_name]["species"]
        self.nu = data[species_name]["nu"]

    def _load_formation_reaction(self) -> dict:
        """
        Load formation reaction database.

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_formation_reactions.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)
