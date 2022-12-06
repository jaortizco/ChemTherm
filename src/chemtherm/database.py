import json
import pathlib

import numpy as np


class Database:
    """
    Class to handle thermodynamic data.

    """
    def __init__(self):
        self._load_critical_constants()
        self._load_formation_properties()
        self._load_heat_capacity_coeffients()
        self._load_formation_reactions()

    def get_critical_constants(self, species):
        """
        Get the critial constants for the specified species.

        """
        crit_cons = np.zeros((len(species), 5))
        for ii, sp in enumerate(species):
            sp = sp.replace("(g)", "").replace("(l)", "").replace("(s)", "")
            crit_cons[ii, :] = np.array([
                self._crit_cons_data[sp]["Tc"],
                self._crit_cons_data[sp]["Pc"]*1e6,
                self._crit_cons_data[sp]["Vc"]*1e3,
                self._crit_cons_data[sp]["Zc"],
                self._crit_cons_data[sp]["w"]])

        return crit_cons

    def get_Cp_coefficients(self, species):
        """
        Get the Cp coefficients for the specified species.

        """
        Cp_coeff = np.zeros(
            (len(species), len(self._Cp_data[species[0]]["coeff"])))
        for ii, sp in enumerate(species):
            Cp_coeff[ii, :] = self._Cp_data[sp]["coeff"]

        return Cp_coeff

    def get_formation_properties(self, species):
        """
        Get the formation properties for the specified species.

        """
        form_props = np.zeros((len(species), 4))
        for ii, sp in enumerate(species):
            form_props[ii, :] = np.array([
                self._form_props_data[sp]["Hf0"],
                self._form_props_data[sp]["Gf0"],
                self._form_props_data[sp]["S0"],
                self._form_props_data[sp]["Hcomb"]])

        return form_props

    def get_formation_reactions(self, species):
        """
        Get the formation reactions for the specified species.

        """
        frxns = []
        for sp in species:
            frxns.append(self._formation_rxns_data[sp])

        return frxns

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
            self._crit_cons_data = json.load(jfile)

    def _load_heat_capacity_coeffients(self):
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
            self._Cp_data = json.load(jfile)

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
            self._form_props_data = json.load(jfile)

    def _load_formation_reactions(self):
        """
        Loade formation reaction database.

        Parameters
        ----------
        species : list
            List of strings containing the name of each species involved in
            the equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].

        Returns
        -------
        frxn : list
            Formation reactions for each species.

        """
        # Load coefficients database
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_formation_reactions.json")

        with open(db_file, "r") as jfile:
            self._formation_rxns_data = json.load(jfile)
