import logging

import numpy as np
import numpy.typing as npt

from chemtherm import utils
from chemtherm.species import Species

logger = logging.getLogger(__name__)


class Mixture:

    def __init__(self, species_list: list[Species]) -> None:
        self.species_list = species_list
        self.num_species = len(species_list)

        self._set_crit_cons()
        self._set_cp_coefficients()
        self._set_formation_properties()
        # self._mix_formation_reactions()
        self._set_elements()

    def calculate_stoichiometry_matrix(self) -> None:
        """
        Calculate the stoichiometry matrix for the mixture.

        The matrix rows represent each species while the columns represent
        each atom.

        Notes
        -----
        These matrix can be used for calculating the atom balance in the Gibbs
        free energy minimization method, for example.

        """
        self.stoic_matrix = np.zeros((self.num_species, len(self.elements)))
        for i, species in enumerate(self.species_list):
            species.calculate_atom_stoichiometry()
            for j, element in enumerate(self.elements):
                self.stoic_matrix[i, j] = float(
                    species.atom_stoic.get(element, 0)
                )

    def calculate_viscosity(
        self,
        T: float,
        yi: npt.NDArray[np.float64],
    ) -> float:
        """
        Calculate the viscosity of a gas mixture using Reichenberg's method.

        Parameters
        ----------
        T : float
            Temperature in K.
        yi : array_like
            Mole fraction of all components.

        Returns
        -------
        eta_mix : float
            Viscosity in micro Poise (muP).

        Notes
        -----
        The method of Reichenberg is used as described in The Properties of
        Gases and Liquids (5th edition). Poling, B. E., Prausnitz, J. M., &
        O'Connell, J. P. (2000). Chapter 9.

        """
        eta_i = np.array(
            [sp.calculate_gas_viscosity(T) for sp in self.species_list],
            dtype=np.float64
        )

        Mi = np.array([sp.M for sp in self.species_list])

        Hij = self._reichenberg_matrix(T, eta_i, Mi)

        Ki = self._K_factor(yi, eta_i, Hij, Mi)

        eta_mix = 0
        for i in range(yi.size):
            sum1 = np.sum([Hij[i, j]*Ki[j] for j in range(i)])
            sum2 = np.sum(
                [
                    Hij[i, j]*Hij[i, k]*Ki[j]*Ki[k] for j in range(yi.size)
                    for k in range(yi.size) if j != i if k != i
                ]
            )
            eta_mix += Ki[i]*(1 + 2*sum1 + sum2)

        return eta_mix

    def _reichenberg_matrix(
        self,
        T: float,
        eta_i: npt.NDArray[np.float64],
        Mi: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the Reichenberg matrix.

        Parameters
        ----------
        T : float
            Temperature in K.
        eta_i : array_like
            Pure gas viscosity in micro Poise (muP).
        Mi : array_like
            Molecular weight in h mol^-1.

        Returns
        -------
        Hij : array_like
            Reichenberg matrix.

        """
        mu_ri = self._dimensionless_dipole()

        Tri = np.array([T/sp.crit_cons.Tc for sp in self.species_list])

        FRi = self._polar_correction(Tri, mu_ri)
        Ui = self._U_factor(Tri, FRi)

        Ci = Mi**(0.25)/(eta_i*Ui)**0.5
        Tr_ij = np.sqrt(np.outer(Tri, Tri))
        mu_rij = np.sqrt(np.outer(mu_ri, mu_ri))

        FRij = self._polar_correction(Tr_ij, mu_rij)
        Uij = self._U_factor(Tr_ij, FRij)

        M_mix = (np.outer(Mi, Mi)/(32*np.add.outer(Mi, Mi)**3))**0.5
        C = np.add.outer(Ci, Ci)

        return M_mix*C**2*Uij

    def _K_factor(
        self,
        yi: npt.NDArray[np.float64],
        eta_i: npt.NDArray[np.float64],
        Hij: npt.NDArray[np.float64],
        Mi: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate K factor.

        Parameters
        ----------
        yi : array_like
            Mole fraction of all components.
        eta_i : array_like
            Pure gas viscosity in micro Poise (muP).
        Hij : array_like
            Reichenberg matrix.
        Mi : array_like
            Molecular weight in h mol^-1.

        Returns
        -------
        Ki : array_like
            K factor.

        """
        Ki = np.zeros(yi.size)
        for i in range(yi.size):
            aux_sum = np.sum(
                [
                    yi[j]*Hij[i, j]*(3 + 2*Mi[j]/Mi[i])
                    for j in range(yi.size) if j != i
                ]
            )
            Ki[i] = yi[i]*eta_i[i]/(yi[i] + eta_i[i]*aux_sum)

        return Ki

    def _dimensionless_dipole(self) -> npt.NDArray[np.float64]:
        """
        Adimensionalize the dipole moments of all element in the mixture.

        Returns
        -------
        mu_r : array_like
            Dimensionless dipole moment.

        """

        def mu_r(sp: Species):
            return 52.46*sp.dipole**2*sp.crit_cons.Pc*1e-5/sp.crit_cons.Tc**2

        return np.array([mu_r(species) for species in self.species_list])

    def _polar_correction(
        self,
        Tr: npt.NDArray[np.float64],
        mu_r: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the polar correction factor.

        Paremeters
        ----------
        Tr : array_like
            Reduced temperature.
        mu_r : array_like
            Dimensionless dipole moment.
        
        Returns
        -------
        FR : array_like
            Polar correction factor.

        """
        return (Tr**3.5 + (10*mu_r)**7)/(Tr**3.5*(1 + (10*mu_r)**7))

    def _U_factor(
        self,
        Tr: npt.NDArray[np.float64],
        FR: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate U factor.

        Paremeters
        ----------
        Tr : array_like
            Reduced temperature.
        FR : array_like
            Polar correction factor.
        
        Returns
        -------
        Ui : array_like
            U factor.

        """
        return ((1 + 0.36*Tr*(Tr-1))**(1/6)/Tr**0.5)*FR

    def _set_crit_cons(self) -> None:
        """
        Get the critical constants for the mix as a 2D numpy array.

        """
        self.crit_cons = np.zeros((len(self.species_list), 5))

        for i, species in enumerate(self.species_list):
            self.crit_cons[i, :] = species.crit_cons.array

    def _set_cp_coefficients(self) -> None:
        """
        Get the cp coefficients for the mix as a 2D numpy array.

        """
        self.cp_coeffs = np.zeros((len(self.species_list), 5))

        for i, species in enumerate(self.species_list):
            self.cp_coeffs[i, :] = species.cp_coeffs.array

    def _set_formation_properties(self) -> None:
        """
        Get the formation properties for the mix as a 2D numpy array.

        """
        self.form_props = np.zeros((len(self.species_list), 4))

        for i, species in enumerate(self.species_list):
            self.form_props[i, :] = species.form_props.array

    def _set_elements(self) -> None:
        """
        Get the elements tha make up all species in the mix.

        """
        self.elements = []
        for species in self.species_list:
            self.elements.extend(species.elements.elements)

        self.elements = utils.remove_duplicates(self.elements)

    def __repr__(self) -> str:
        return f"Mix: {[species.name for species in self.species_list]}"


def main():
    print("Mixture class")


if __name__ == "__main__":
    main()
