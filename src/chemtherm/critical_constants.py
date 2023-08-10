import json
import pathlib

import numpy as np

from chemtherm import utils


class CriticalConstants:
    """
    Class to handle critical constants data.

    Tc in K, Pc in Pa, and Vc in cm^3 mol^-1.

    """

    def __init__(self, species_name: str) -> None:
        """
        Get the critial constants for the specified species.

        """
        data = self._load_critical_constants()

        species_name = utils.strip_species_name(species_name)

        self.Tc = data[species_name]["Tc"]
        self.Pc = data[species_name]["Pc"]*1e6
        self.Vc = data[species_name]["Vc"]*1e3
        self.Zc = data[species_name]["Zc"]
        self.w = data[species_name]["w"]

        self.array = np.array([self.Tc, self.Pc, self.Vc, self.Zc, self.w])

    def _load_critical_constants(self) -> dict:
        """
        Critical constants.

        Returns
        -------
        crit_cons : list
            Critical constants.
            Tc in K, Pc in MPa, and Vc in m^3 kmol^-1.

        Notes
        -----
        The critical constants are taken from Table 2-141 of Perry's Chemical
        Engineering Handbook, 8th Edition. At the same time, values in that
        table were taken from the Design Institute for Physical Properties
        (DIPPR) of the American Institute of Chemical Engineers (AIChE),
        copyright 2007 AIChE and reproduced with permission of AICHE and of
        the DIPPR Evaluated Process Design Data Project Steering Committee.
        Their source should be cited as R. L. Rowley, W. V. Wilding,
        J. L. Oscarson, Y. Yang, N. A. Zundel, T. E. Daubert, R. P. Danner,
        DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
        for Physical Properties, AIChE, New York (2007).

        The number of digits provided for the acentric factor was chosen for
        uniformity of appearence and formatting; these do not represent the
        uncertainties of the physical quantities, but are the result of
        calculations from the standard thermophysical property formulation
        within a fixed format.

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_critical_constants.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)
