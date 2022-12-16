import logging
import re

import numpy as np
import numpy.typing as npt
from scipy import optimize

from chemtherm import phycons as con
from chemtherm import rxn
from chemtherm.mixture import Mixture

logger = logging.getLogger(__name__)


def peng_robinson(
        T: float, P: float, y: npt.NDArray, Tc: npt.NDArray,
        Pc: npt.NDArray, w: npt.NDArray):
    """
    Calculate the compressibility factor of gas a mixture and the fugacity
    coefficients of its components using the Peng-Robinson equation of state
    and the van der Vaals mixing rule.

    Parameters
    ----------
    T : scalar
      Temperature in K
    P : scalar
      Pressure in bar. It will be converted to Pa within this function.
    y : column vector
      Mole fraction of all gas species

    Returns
    -------
    phi : scalar
      Fugacity coefficients of each species in the mixture.

    """
    P = P * 1e5  # bar --> Pa
    # -------------------------------------------------------------------------
    # Calculate a and b for pure components
    k = 0.37464 + 1.54226*w - 0.26992 * w**2

    alpha = (1 + k * (1 - np.sqrt(T/Tc)))**2

    a = 0.45724 * ((con.R**2*Tc**2)/Pc)*alpha
    b = 0.07780 * (con.R*Tc/Pc)
    # -------------------------------------------------------------------------
    # Calculate amix and bmix using the vaan der Waals mixing rule
    aij = np.zeros((a.size, a.size))
    for i in range(a.size):
        for j in range(a.size):
            aij[i, j] = np.sqrt(a[i] * a[j])

    amix = y @ aij @ y
    bmix = y @ b
    # -------------------------------------------------------------------------
    # Calculate A and B
    Aij = (aij * P) / (con.R * T)**2
    Bi = (b * P) / (con.R * T)

    Amix = (amix*P)/(con.R*T)**2
    Bmix = (bmix*P)/(con.R*T)
    # -------------------------------------------------------------------------
    # Calculate liquid and gas compressibility factor by finding the roots of
    # the third grade polinomial
    pol = [
        1, (-1+Bmix), (Amix - 3*Bmix**2 - 2*Bmix),
        (-Amix*Bmix + Bmix**2 + Bmix**3)]

    Z = np.roots(pol)
    Z = np.real(Z)

    Zmix = np.max(Z)
    # -------------------------------------------------------------------------
    # Calculate the fugacity coefficients
    lnphi = (
        (Bi/Bmix)*(Zmix-1) - np.log(Zmix-Bmix)
        - (Amix/(2*np.sqrt(2)*Bmix))*((2*(Aij@y)/Amix)-(Bi/Bmix))*np.log(
            (Zmix+(1+np.sqrt(2))*Bmix)/(Zmix+(1-np.sqrt(2))*Bmix)))
    # -------------------------------------------------------------------------
    return np.exp(lnphi)


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


def list2dict(lst):
    """
    Convert a list into a dictionary.

    The first element of the list will be the first key of the dictonary and
    the second element of the list will be the first value of the dictionary.

    Parameters
    ----------
    lst : list
        List to be converted into a dictionary.

    Returns
    -------
    dict
        Dictionary.

    """
    mydict = {}
    for index, item in enumerate(lst):
        if index % 2 == 0:
            mydict[item] = lst[index+1]

    return mydict


def atom_stoichiometry(species: str) -> dict:
    """
    Get the different type of atoms that make up a molecule and their number
    of acurrences.

    Parameters
    ----------
    species : list
        List of strings containing the name of each species involved in the
        equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].

    Returns
    -------
    atom_stoichiometryc : dict
        Different type of atoms that make up a molecule and their number
        of acurrences.

    """
    atom_stoic = re.findall(
        r'[A-Z][a-z]*|\d+',
        re.sub(r'[A-Z][a-z]*(?![\da-z])', r'\g<0>1',
               species))

    return list2dict(atom_stoic)


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
        atom_stoic = atom_stoichiometry(species.name)
        for j, element in enumerate(mix.elements):
            if element in atom_stoic:
                n_atom[i, j] = n[i]*int(atom_stoic[element])

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
        T, P, y,
        crit_cons[:, 0], crit_cons[:, 1], crit_cons[:, 4])

    return y*phi*P/P0


def gibbs_objfun(
        n: npt.NDArray, T: float, P: float, Grxn: npt.NDArray,
        mix: Mixture) -> float:
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

    GRT = Grxn/(con.R*T)

    a = activity(T, P, y, mix.crit_cons)

    # It it possible for the number of moles of one of the species to become
    # zero or slightly negative during iteration, causing issues with the
    # natural logarithm. Therefore, the warnings during computing the logaritm
    # are ignored.
    with np.errstate(divide="ignore", invalid="ignore"):
        lna = np.log(a)
        value = n*(GRT + lna)

    return np.sum(value)


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

    Hrxn, Grxn, Srxn = reaction_properties(mix, nu, T, Tref=298.15)
    Kp = np.exp(-Grxn / (con.R*T))

    return Kp, Hrxn, Grxn, Srxn


def reaction_properties(
        mix: Mixture, nu: npt.NDArray[np.int32], T: float,
        Tref: float = 298.15):
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
    Hf0 = mix.form_props[:, 0]
    S0 = mix.form_props[:, 2]

    Hrxn0 = np.sum(nu*Hf0)
    Srxn0 = np.sum(nu*S0)

    Hrxn = rxn.enthalpy(Hrxn0, nu, mix.cp_coeffs, T, Tref)
    Srxn = rxn.entropy(Srxn0, nu, mix.cp_coeffs, T, Tref)
    Grxn = Hrxn - T*Srxn

    return Hrxn, Grxn, Srxn
