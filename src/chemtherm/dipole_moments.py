import json
import pathlib

from chemtherm import utils


class DipoleMoment:
    """
    Class to handle critical constants data.

    """

    def __init__(self, species_name: str) -> None:
        """
        Get the critial constants for the specified species.

        """
        self.species_name = species_name

    def load_dipole_moment(self) -> float:
        """
        Load the dipole moment.

        Returns
        -------
        dipole_moment : float
            Dipole moments in debye units (D).

        Notes
        -----
        The Dipole moments are taken from the CRC Handbook of Chemistry and
        Physics (84th edition). Lide, D. R. (2003). Chapter 9 (9-45).

        This table gives values of the electric dipole moment for about 800
        molecules. When available, values determined by microwave
        spectroscopy, molecular beam electric resonance, and other
        high-resolution spectroscopic techniques were selected. Otherwise,
        the values come from measurements of the dielectric constant in the
        gas phase or, if these do not exist, in the liquid phase. Compounds
        are listed by molecular formula in Hill order; compounds not
        containing carbon are listed first, followed by compounds containing
        carbon. The dipole moment μ is given in debye units (D). The
        conversion factor to SI units is 1 D = 3.33564e-30 C m.

        """
        dir = pathlib.Path(__file__).resolve().parent
        db_file = pathlib.Path(dir, "data/db_dipole_moments.json")

        with open(db_file, "r") as jfile:
            data = json.load(jfile)

        return data[utils.strip_species_name(self.species_name)]
