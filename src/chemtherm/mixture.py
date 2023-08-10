import logging

import numpy as np

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

    # def _mix_formation_reactions(self):
    #     """
    #     Get the formation properties for the mix as a 2D numpy array.

    #     """
    #     self.form_rxns = np.zeros(
    #         (len(self.species_list), len(self.species_list[0].form_rxn)))

    #     for i, species in enumerate(self.species_list):
    #         self.form_rxns[i, :] = species.form_rxn

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
