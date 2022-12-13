import logging

import numpy as np

from chemtherm.species import Species

logger = logging.getLogger(__name__)


class Mixture:
    def __init__(self, species_list: list[Species]):
        self.species_list = species_list
        self.num_species = len(species_list)

        self._mix_critical_constants()
        self._mix_cp_coefficients()
        self._mix_formation_properties()
        # self._mix_formation_reactions()
        self._mix_elements()

    def _mix_critical_constants(self):
        """
        Get the critical constants for the mix as a 2D numpy array.

        """
        self.crit_cons = np.zeros(
            (len(self.species_list), len(self.species_list[0].crit_cons)))

        for i, species in enumerate(self.species_list):
            self.crit_cons[i, :] = species.crit_cons

    def _mix_cp_coefficients(self):
        """
        Get the cp coefficients for the mix as a 2D numpy array.

        """
        self.cp_coeffs = np.zeros(
            (len(self.species_list), len(self.species_list[0].cp_coeffs)))

        for i, species in enumerate(self.species_list):
            self.cp_coeffs[i, :] = species.cp_coeffs

    def _mix_formation_properties(self):
        """
        Get the formation properties for the mix as a 2D numpy array.

        """
        self.form_props = np.zeros(
            (len(self.species_list), len(self.species_list[0].form_props)))

        for i, species in enumerate(self.species_list):
            self.form_props[i, :] = species.form_props

    # def _mix_formation_reactions(self):
    #     """
    #     Get the formation properties for the mix as a 2D numpy array.

    #     """
    #     self.form_rxns = np.zeros(
    #         (len(self.species_list), len(self.species_list[0].form_rxn)))

    #     for i, species in enumerate(self.species_list):
    #         self.form_rxns[i, :] = species.form_rxn

    def _mix_elements(self):
        """
        Get the elements tha make up all species in the mix.

        """
        self.elements = []
        for species in self.species_list:
            self.elements.extend(species.elements)

        # Remove duplicates by using dictionaries
        self.elements = list(dict.fromkeys(self.elements))

    def __repr__(self):
        return f"Mix: {[species.name for species in self.species_list]}"


def main():
    print("Mixture class")


if __name__ == "__main__":
    main()
