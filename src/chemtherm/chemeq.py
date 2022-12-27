import logging

import numpy as np
import numpy.typing as npt
from scipy import optimize

from chemtherm import phycons as pc
from chemtherm import rxn
from chemtherm.mixture import Mixture
from chemtherm.peng_robinson import peng_robinson

logger = logging.getLogger(__name__)


def especies_properties(
        mix: Mixture, T: float, Tref: float = 298.15) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]]:
    """
    Calculate the formation enthalpy, Gibbs energy and entropy of each species
    at the specified temperature.

    species : list
        List of strings containing the name of each species involved in the
        equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].
    frxn : list
        Formation reactions for each species.
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
    Hrxn, Grxn, Srxn = (np.zeros(mix.num_species) for _ in range(3))
    for i, species in enumerate(mix.species_list):
        Hrxn[i], Grxn[i], Srxn[i] = species.properties_at_T(T, Tref)

    return Hrxn, Grxn, Srxn


def atom_balance(n: npt.NDArray, mix: Mixture) -> npt.NDArray:
    """
    Calculate the total number of moles of each atom in the mixture.

    Parameters
    ----------
    n : array_like
        Number of moles of each species in mol.
    species : list
        List of strings containing the name of each species involved in the
        equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].
    elements : list
        List of strings containing the name of each element that makes up all
        species. For example, ["H", "O"].

    Returns
    -------
    n_atom : array
        Number of moles of each atom in mol.

    """
    n_atom = np.zeros((mix.num_species, len(mix.elements)))
    for i, species in enumerate(mix.species_list):
        for j, element in enumerate(mix.elements):
            if element in species.atom_stoic:
                n_atom[i, j] = n[i]*int(species.atom_stoic[element])

    return np.sum(n_atom, axis=0)


def activity(T, P, y, crit_cons, P0=1):
    """
    Calculate the activity of each species at a given temperature, pressure
    and composition.

    Parameters
    ----------
    T : float
        Temperature in K.
    P : float
        Pressure in bar.
    y : array_like
        Mol fraction of each species.
    crit_cons : array_like
        Critical constants for each species. The order of columns should be
        `[Tc, Pc, Vc, Zc, w]`.
    P0 : float, optional
        Reference pressure. Commonly set to 1 bar.

    Returns
    -------
    a : array_like
        Activity of species

    Notes
    -----
    Only gas species are supported at the moment.

    """
    # Ideal gas
    # phi = np.ones(y.size)

    # Real gas
    phi = peng_robinson(
        T, P, y, crit_cons[:, 0], crit_cons[:, 1], crit_cons[:, 4])

    return y*phi*P/P0


def gibbs_objfun(
        n: npt.NDArray[np.float64], T: float, P: float,
        Grxn: npt.NDArray[np.float64], mix: Mixture) -> float:
    """
    Objective function for the Gibbs energy minimization.

    This function is called by `gibbs_minimization`

    Parameters
    ----------
    T : float
        Temperature in K.
    P : float
        Pressure in bar.
    n0 : array_like
        Initial number of moles of each species in mol.
    species : list
        List of strings containing the name of each species involved in the
        equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].
    elements : list
        List of strings containing the name of each element that makes up all
        species. For example, ["H", "O"].

    Returns
    -------
    float
        Total Gibbs free energy of the system.

    """
    y = n/np.sum(n)

    GRT = Grxn/(pc.R*T)

    a = activity(T, P, y, mix.crit_cons)

    # It it possible for the number of moles of one of the species to become
    # zero or slightly negative during iteration, causing issues with the
    # natural logarithm. Therefore, the warnings during computing the logaritm
    # are ignored.
    with np.errstate(divide="ignore", invalid="ignore"):
        lna = np.log(a)
        value = n*(GRT + lna)

    return value.sum()


def gibbs_minimization(
        T: float, P: float, n0: npt.NDArray, mix: Mixture,
        Tref: float = 298.15) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Find the number of moles and the mole fractions at the equilibrium state
    of the involved species by minimizing the total Gibbs free energy.

    The total number of atoms of each element must be conserved.

    Parameters
    ----------
    T : float
        Temperature in K.
    P : float
        Pressure in bar.
    n0 : array_like
        Initial number of moles of each species in mol.
    species : list
        List of strings containing the name of each species involved in the
        equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].
    elements : list
        List of strings containing the name of each element that makes up all
        species. For example, ["H", "O"].

    Returns
    -------
    n_eq : array_like
        Number of moles of each species at the equilibrium state in mol.
    y_eq : array_like
        Mole fraction of each species at the equilibrium state.

    """
    scaling_factor = np.sum(n0)
    n0 = n0/scaling_factor
    # -------------------------------------------------------------------------
    _, Grxn, _ = especies_properties(mix, T)

    for species in mix.species_list:
        species.calculate_atom_stoichiometry()

    atom_moles_init = atom_balance(n0, mix)
    # -------------------------------------------------------------------------
    # Constraints:
    # 1) The number of moles of each species must be greater than zero
    # 2) The total number of atoms must be conserved
    contraints = [
        {
            "type": "ineq",
            "fun": lambda n: n
        },
        {
            "type": "eq",
            "fun": (lambda n: atom_moles_init - atom_balance(n, mix))
        }]
    # -------------------------------------------------------------------------
    n_eq_init = np.copy(n0)
    n_eq_init[n_eq_init == 0] = 1e-12

    opts = {"ftol": 1e-14, "maxiter": 300}
    sol = optimize.minimize(
        gibbs_objfun, n_eq_init,
        args=(T, P, Grxn, mix),
        method="SLSQP", options=opts, constraints=contraints)

    if not sol.success:
        msg = f"Gibbs minimization failed: {sol.message}"
        logger.warning(msg)
    # -------------------------------------------------------------------------
    n_eq = sol.x * scaling_factor
    y_eq = n_eq/np.sum(n_eq)
    # -------------------------------------------------------------------------
    return n_eq, y_eq


def eq_cons(
        mix: Mixture, nu: npt.NDArray[np.int32], T: float,
        Tref: float = 298.15) -> tuple[float, float, float, float]:
    """
    Compute the enthalpy, the gibbs energy, the entropy, and the equilibrium
    constant of a given reaction at a specified temperature.

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
    K : float
        Equilibrium constant.
    Hrxn : float
        Enthalpy of reaction at the given temperature in J mol^-1.
    Grxn : float
        Gibbs free energy of reaction at the given temperature in J mol^-1.
    Srxn : float
        Entropy of reaction at the given temperature in J mol^-1 K^-1.

    """

    Hrxn, Grxn, Srxn = rxn.reaction_properties(
        T, mix.form_props[:, 0], mix.form_props[:, 2], mix.cp_coeffs, nu,
        Tref)

    Kp = np.exp(-Grxn / (pc.R*T))

    return Kp, Hrxn, Grxn, Srxn
