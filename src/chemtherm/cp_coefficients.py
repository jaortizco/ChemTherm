import json
import pathlib

import numpy as np


class CpCoefficients:
    """
    Class to handle Cp coefficient data.

    """
    def __init__(self, species_name: str) -> None:
        """
        Get the Cp coefficients for the specified species.

        """
        data = self._load_cp_coefficients()

        self.A = data[species_name]["coeff"]["A"]
        self.B = data[species_name]["coeff"]["B"]
        self.C = data[species_name]["coeff"]["C"]
        self.D = data[species_name]["coeff"]["D"]
        self.E = data[species_name]["coeff"]["E"]

        self.array = np.array([self.A, self.B, self.C, self.D, self.E])

    def _load_cp_coefficients(self) -> dict:
        """
        Load Cp coefficients database.

        Parameters
        ----------
        species : list
            Species to get the Cp data for.

        Returns
        -------
        dict
            Cp data.

        """
        # Load Cp coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_heat_capacity.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)
