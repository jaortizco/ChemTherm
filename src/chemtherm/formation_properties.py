import json
import pathlib

import numpy as np


class FormationProperties:
    """
    Class to handle formation properties data.

    """

    def __init__(self, species_name: str) -> None:
        """
        Get the formation properties for the specified species.

        """
        data = self._load_formation_properties()

        self.Hf0 = data[species_name]["Hf0"]
        self.Gf0 = data[species_name]["Gf0"]
        self.S0 = data[species_name]["S0"]
        self.Hcomb = data[species_name]["Hcomb"]

        self.array = np.array([self.Hf0, self.Gf0, self.S0, self.Hcomb])

    def _load_formation_properties(self) -> dict:
        """
        Load formation properties database.

        Returns
        -------
        form_props : dict
            Formation properties at 298.15 K. Hf0 in J mol^-1, Gf0 in J mol^-1,
            S0 in J mol^-1 K^-1, and H_comb in J mol^-1.

        Notes
        -----
        - The formation properties are taken from Table 2-179 of Perry's
        Chemical Engineering Handbook, 8th Edition. At the same time,
        values in that table were taken from the Design Institute for Physical
        Properties (DIPPR) of the American Institute of Chemical Engineers
        (AIChE), copyright 2007 AIChE and reproduced with permission of AICHE
        and of the DIPPR Evaluated Process Design Data Project Steering
        Committee. Their source should be cited as R. L. Rowley,
        W. V. Wilding, J. L. Oscarson, Y. Yang, N. A. Zundel, T. E. Daubert,
        R. P. Danner, DIPPR® Data Compilation of Pure Chemical Properties,
        Design Institute for Physical Properties, AIChE, New York (2007).

        - For CH2ClF2, the formation properties are taken from the "Standard
        Thermodynamic Properties of Chemical Substances" table in
        Section 5, 5-4 of the CRC Handbook of Chemistry and Physics
        (84th edition). Lide, D. R. (2003)

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_formation_properties.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)
