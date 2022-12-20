import numpy as np

from chemtherm import rxn
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

    def properties_at_T(
            self, T: float, Tref: float = 298.15) -> tuple[
                float, float, float]:
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

        Hrxn0 = np.sum(nu*Hf0)
        Srxn0 = np.sum(nu*S0)

        Hrxn = rxn.enthalpy(Hrxn0, nu, Cp_coeff, T, Tref)
        Srxn = rxn.entropy(Srxn0, nu, Cp_coeff, T, Tref)
        Grxn = Hrxn - T*Srxn

        return Hrxn, Grxn, Srxn
