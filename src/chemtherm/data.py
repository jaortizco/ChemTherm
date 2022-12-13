import json
import pathlib

import numpy as np
import numpy.typing as npt


class CriticalConstants:
    """
    Class to handle critical constants data.

    """
    def get_critical_constants(self, species: str) -> npt.NDArray[np.float64]:
        """
        Get the critial constants for the specified species.

        """
        data = self._load_critical_constants()

        species = species.replace(
            "(g)", "").replace(
                "(l)", "").replace(
                    "(s)", "")

        crit_cons = np.array([
            data[species]["Tc"],
            data[species]["Pc"]*1e6,
            data[species]["Vc"]*1e3,
            data[species]["Zc"],
            data[species]["w"]])

        return crit_cons

    def _load_critical_constants(self):
        """
        Critical constants.

        Parameters
        ----------
        species : list
            species to get Cp.

        Returns
        -------
        crit_cons : list
            Critical constants.
            Tc in K, Pc in Pa, and Vc in m^3 mol^-1.

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


class CpCoefficients:
    """
    Class to handle Cp coefficient data.

    """
    def get_cp_coefficients(self, species: str) -> npt.NDArray[np.float64]:
        """
        Get the Cp coefficients for the specified species.

        """
        data = self._load_cp_coefficients()

        return np.array(data[species]["coeff"])

    def _load_cp_coefficients(self):
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
        db_file = pathlib.Path(dir, "data/db_cp.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)


class FormationProperties:
    """
    Class to handle formation properties data.

    """
    def get_formation_properties(
            self, species: str) -> npt.NDArray[np.float64]:
        """
        Get the formation properties for the specified species.

        """
        data = self._load_formation_properties()

        form_props = np.array([
                data[species]["Hf0"],
                data[species]["Gf0"],
                data[species]["S0"],
                data[species]["Hcomb"]])

        return form_props

    def _load_formation_properties(self):
        """
        Load formation properties database.

        Parameters
        ----------
        species : list
            Species to get the formation properties for.

        Returns
        -------
        form_props : dict
            Formation properties at 298.15 K. Hf0 in J mol^-1, Gf0 in J mol^-1,
            S0 in J mol^-1 K^-1, and H_comb in J mol^-1.

        Notes
        -----
        The formation properties are taken from Table 2-179 of Perry's
        Chemical Engineering Handbook, 8th Edition. At the same time,
        values in that table were taken from the Design Institute for Physical
        Properties (DIPPR) of the American Institute of Chemical Engineers
        (AIChE), copyright 2007 AIChE and reproduced with permission of AICHE
        and of the DIPPR Evaluated Process Design Data Project Steering
        Committee. Their source should be cited as R. L. Rowley,
        W. V. Wilding, J. L. Oscarson, Y. Yang, N. A. Zundel, T. E. Daubert,
        R. P. Danner, DIPPR® Data Compilation of Pure Chemical Properties,
        Design Institute for Physical Properties, AIChE, New York (2007).

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_formation_properties.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)


class FormationReaction:
    """
    Class to handle formation reaction data.

    """
    def get_formation_reaction(self, species: str):
        """
        Get the formation reactions for the specified species.

        """
        data = self._load_formation_reaction()
        return data[species]

    def _load_formation_reaction(self):
        """
        Load formation reaction database.

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_formation_reactions.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)


class Elements:
    """
    Class to handle elements data.

    """
    def get_elements(self, species: str) -> dict:
        """
        Get the elements that make up the specified species.

        """
        species = species.replace(
            "(g)", "").replace(
                "(l)", "").replace(
                    "(s)", "")

        data = self._load_elements()
        return data[species]

    def _load_elements(self):
        """
        Load elements database.

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_elements.json")

        with open(db_file, "r") as jfile:
            return json.load(jfile)
