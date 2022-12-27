import re

import numpy as np

from chemtherm import rxn, utils
from chemtherm.cp_coefficients import CpCoefficients
from chemtherm.critical_constants import CriticalConstants
from chemtherm.elements import Elements
from chemtherm.formation_properties import FormationProperties
from chemtherm.formation_reaction import FormationReaction


class Species:
    def __init__(self, name: str):
        self.name = name

        self.crit_cons = CriticalConstants(self.name)
        self.cp_coeffs = CpCoefficients(self.name)
        self.form_props = FormationProperties(self.name)
        self.form_rxn = FormationReaction(self.name)
        self.elements = Elements(self.name)

    def calculate_atom_stoichiometry(self) -> None:
        """
        Get the different type of atoms that make up a molecule and their
        number of acurrences.

        Parameters
        ----------
        species : list
            List of strings containing the name of each species involved in
            the equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].

        Returns
        -------
        atom_stoichiometryc : dict
            Different type of atoms that make up a molecule and their number
            of acurrences.

        """
        atom_stoic = re.findall(
            r'[A-Z][a-z]*|\d+',
            re.sub(
                r'[A-Z][a-z]*(?![\da-z])', r'\g<0>1',
                self.name))

        self.atom_stoic = utils.list2dict(atom_stoic)

    def properties_at_T(
            self, T: float, Tref: float = 298.15) -> tuple[
            np.float64, np.float64, np.float64]:
        """
        Compute the enthalpy, the gibbs energy, and the entropy of a given
        reaction at a specified temperature.

        Parameters
        ----------
        species : list
            List of strings containing the name of each species involved in
            the reaction. For example, ["H2(g)", "O2(g)", "H2O(g)"].
        nu : array
            Stoichiometry coefficients of each especies.
        T : float
            Temperature in K.
        Tref : float, optional
            Reference temperature in K.

        Returns
        -------
        Hrxn : float
            Enthalpy of reaction at the given temperature in J mol^-1.
        Grxn : float
            Gibbs free energy of reaction at the given temperature in J mol^-1.
        Srxn : float
            Entropy of reaction at the given temperature in J mol^-1 K^-1.

        """
        Hf0, S0 = (np.zeros(len(self.form_rxn.species)) for _ in range(2))
        Cp_coeff = np.zeros(
            (len(self.form_rxn.species), len(self.cp_coeffs.array)))
        for i, species in enumerate(self.form_rxn.species):
            Hf0[i] = FormationProperties(species).Hf0
            S0[i] = FormationProperties(species).S0
            Cp_coeff[i, :] = CpCoefficients(species).array

        nu = np.array(self.form_rxn.nu)

        Hrxn, Grxn, Srxn = rxn.reaction_properties(
            T, Hf0, S0, Cp_coeff, nu, Tref)

        return Hrxn, Grxn, Srxn  # type: ignore
