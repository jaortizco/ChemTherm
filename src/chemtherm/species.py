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

        self.critical_constants = CriticalConstants(self.name)
        self.cp_coefficients = CpCoefficients(self.name)
        self.formation_properties = FormationProperties(self.name)
        self.form_rxn = FormationReaction(self.name)
        self.elements = Elements(self.name)

        self._set_crit_cons()
        self._set_cp_coeffs()
        self._set_form_props()

    def _set_crit_cons(self) -> None:
        self.crit_cons = np.array([
            self.critical_constants.Tc,
            self.critical_constants.Pc,
            self.critical_constants.Vc,
            self.critical_constants.Zc,
            self.critical_constants.w,])

    def _set_cp_coeffs(self) -> None:
        self.cp_coeffs = np.array([
            self.cp_coefficients.A,
            self.cp_coefficients.B,
            self.cp_coefficients.C,
            self.cp_coefficients.D,
            self.cp_coefficients.E])

    def _set_form_props(self) -> None:
        self.form_props = np.array([
            self.formation_properties.Hf0,
            self.formation_properties.Gf0,
            self.formation_properties.S0,
            self.formation_properties.Hcomb])

    # def get_

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
        Hf0, S0 = (np.zeros(len(self.form_rxn.species)) for _ in range(2))
        cp_coeff = np.zeros(
            (len(self.form_rxn.species), len(self.cp_coeffs)))
        for i, species in enumerate(self.form_rxn.species):
            Hf0[i] = FormationProperties(species).Hf0
            S0[i] = FormationProperties(species).S0
            cp_coeff[i, :] = CpCoefficients(species).array

        nu = np.array(self.form_rxn.nu)

        Hrxn0 = np.sum(nu*Hf0)
        Srxn0 = np.sum(nu*S0)

        Hrxn = rxn.enthalpy(Hrxn0, nu, cp_coeff, T, Tref)
        Srxn = rxn.entropy(Srxn0, nu, cp_coeff, T, Tref)
        Grxn = Hrxn - T*Srxn

        return Hrxn, Grxn, Srxn
