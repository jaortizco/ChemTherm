import numpy as np

from chemtherm import rxn
from chemtherm.data import (CpCoefficients, CriticalConstants, Elements,
                            FormationProperties, FormationReaction)


class Species:
    def __init__(self, name: str):
        self.name = name

        # I have to instantiate all classes. Is it be better to use functions
        # instead.

        self.crit_cons = CriticalConstants().get_critical_constants(self.name)
        self.cp_coeffs = CpCoefficients().get_cp_coefficients(self.name)
        self.form_props = FormationProperties().get_formation_properties(
            self.name)
        self.form_rxn = FormationReaction().get_formation_reaction(self.name)
        self.elements = Elements().get_elements(self.name)

    def properties_at_T(
            self, T: float, Tref: float = 298.15) -> tuple[
                float, float, float]:
        """
        Compute the enthalpy, the gibbs energy, and the entropy of a given
        reaction at a specified temperature.

        Parameters
        ----------
        species : list
            List of strings containing the name of each species involved in the
            reaction. For example, ["H2(g)", "O2(g)", "H2O(g)"].
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
        form_props = np.zeros(
            (len(self.form_rxn["species"]), len(self.form_props)))
        cp_coeff = np.zeros(
            (len(self.form_rxn["species"]), len(self.cp_coeffs)))
        for i, species in enumerate(self.form_rxn["species"]):
            form_props[i, :] = FormationProperties().get_formation_properties(
                species)
            cp_coeff[i, :] = CpCoefficients().get_cp_coefficients(species)

        Hf0 = form_props[:, 0]
        S0 = form_props[:, 2]

        nu = np.array(self.form_rxn["nu"])

        Hrxn0 = np.sum(nu*Hf0)
        Srxn0 = np.sum(nu*S0)

        Hrxn = rxn.enthalpy(Hrxn0, nu, cp_coeff, T, Tref)
        Srxn = rxn.entropy(Srxn0, nu, cp_coeff, T, Tref)
        Grxn = Hrxn - T*Srxn

        return Hrxn, Grxn, Srxn
