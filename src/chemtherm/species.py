import re

import numpy as np
import numpy.typing as npt

from chemtherm import thermoprops as thp
from chemtherm import utils
from chemtherm.cp_coefficients import CpCoefficients
from chemtherm.critical_constants import CriticalConstants
from chemtherm.dipole_moments import DipoleMoment
from chemtherm.elements import Elements
from chemtherm.formation_properties import FormationProperties
from chemtherm.formation_reaction import FormationReaction
from chemtherm.molecular_weight import MolecularWeight


class Species:

    def __init__(self, name: str):
        self.name = name

        self.crit_cons = CriticalConstants(self.name)
        self.cp_coeffs = CpCoefficients(self.name)
        self.form_props = FormationProperties(self.name)
        self.form_rxn = FormationReaction(self.name)
        self.elements = Elements(self.name)
        self.M = MolecularWeight(self.name).load_molecular_weight()
        self.dipole = DipoleMoment(self.name).load_dipole_moment()

    def calculate_atom_stoichiometry(self) -> None:
        """
        Get the different type of atoms that make up a species and their
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
            re.sub(r'[A-Z][a-z]*(?![\da-z])', r'\g<0>1', self.name)
        )

        self.atom_stoic = utils.list2dict(atom_stoic)

    def reaction_properties(
        self,
        T: float,
        Tref: float = 298.15
    ) -> tuple[np.float64, np.float64, np.float64]:
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
            (len(self.form_rxn.species), len(self.cp_coeffs.array))
        )
        for i, species in enumerate(self.form_rxn.species):
            Hf0[i] = FormationProperties(species).Hf0
            S0[i] = FormationProperties(species).S0
            Cp_coeff[i, :] = CpCoefficients(species).array

        nu = np.array(self.form_rxn.nu)

        Hrxn, Grxn, Srxn = thp.reaction_properties(
            T, Hf0, S0, Cp_coeff, nu, Tref=Tref
        )

        return Hrxn, Grxn, Srxn  # type: ignore

    def thermodynamic_properties(
        self,
        T: float,
        Tref: float = 298.15,
    ) -> npt.NDArray[np.float64]:

        return thp.thermodynamic_properties(
            T,
            self.form_props.Hf0,
            self.form_props.S0,
            self.cp_coeffs.array,
            Tref=Tref
        )  # type: ignore

    def calculate_gas_viscosity(self, T: float) -> float:
        """
        Calculate the pure gas viscosity using Ching's method.

        Parameters
        ----------
        T : float
            Temperature in K.

        Returns
        -------
        eta : float
            Viscosity in micro Poise (muP).

        Notes
        -----
        The method of Chung is used as described in The Properties of Gases
        and Liquids (5th edition). Poling, B. E., Prausnitz, J. M., &
        O'Connell, J. P. (2000). Chapter 9.

        """
        A, B, C, D, E, F = (
            1.16145, 0.14874, 0.52487, 0.77320, 2.16178, 2.43787
        )

        T_star = 1.2593*(T/self.crit_cons.Tc)

        # Viscosity collision integral
        Omega = A*T_star**-B + C*np.exp(-D*T_star) + E*np.exp(-F*T_star)

        # Reduced dipole moment
        mu_r = 131.3*self.dipole/(self.crit_cons.Vc*self.crit_cons.Tc)**0.5

        # Factor to account for molecular shapes and polarities
        Fc = 1 - 0.2756*self.crit_cons.w + 0.059035*mu_r**4

        return 40.785*(Fc*(self.M*T)**0.5/(self.crit_cons.Vc**(2/3)*Omega))
