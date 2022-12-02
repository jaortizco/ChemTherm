"""
Thermodynamics module.

"""
import json
import logging
import pathlib
import re

import numpy as np
from scipy import integrate, optimize


logger = logging.getLogger(__name__)


R = (101325*22.414) / (1000*273.15)  # Ideal gas constant in J mol^-1 K^-1
F = 96485  # Faraday constant in A s mol^-1


def critical_constants(species):
    """
    Critical constants.

    Parameters
    ----------
    species : list
        species to get Cp.

    Returns
    -------
    crit_cons : list
        Critical constants.
        Tc in K, Pc in Pa, and Vc in m^3 mol^-1.

    Notes
    -----
    The critical constants are taken from Table 2-141 of Perry's Chemical
    Engineering Handbook, 8th Edition. At the same time, values in that table
    were taken from the Design Institute for Physical Properties (DIPPR) of
    the American Institute of Chemical Engineers (AIChE), copyright 2007
    AIChE and reproduced with permission of AICHE and of the DIPPR Evaluated
    Process Design Data Project Steering Committee. Their source should be
    cited as R. L. Rowley, W. V. Wilding, J. L. Oscarson,
    Y. Yang, N. A. Zundel, T. E. Daubert, R. P. Danner,
    DIPPR® Data Compilation of Pure Chemical Properties, Design Institute for
    Physical Properties, AIChE, New York (2007).

    The number of digits provided for the acentric factor was chosen for
    uniformity of appearence and formatting; these do not represent the
    uncertainties of the physical quantities, but are the result of
    calculations from the standard thermophysical property formulation within
    a fixed format.

    """
    check_species(species)
    # -------------------------------------------------------------------------
    # Load coefficients database
    dir = pathlib.Path(__file__).resolve().parent
    database_file = pathlib.Path(dir, "db_critical_constants.json")

    with open(database_file, "r") as jfile:
        data = json.load(jfile)
    # -------------------------------------------------------------------------
    crit_cons = np.zeros((len(species), 5))
    for ii, sp in enumerate(species):
        sp = sp.replace("(g)", "").replace("(l)", "").replace("(s)", "")
        crit_cons[ii, :] = np.array([
            data[sp]["Tc"], data[sp]["Pc"]*1e6, data[sp]["Vc"]*1e3,
            data[sp]["Zc"], data[sp]["w"]])
    # -------------------------------------------------------------------------
    return crit_cons


def formation_properties(species):
    """
    Load formation properties database.

    Parameters
    ----------
    species : list
        Species to get the formation properties for.

    Returns
    -------
    form_props : dict
        Formation properties at 298.15 K. Hf0 in J mol^-1, Gf0 in J mol^-1,
        S0 in J mol^-1 K^-1, and H_comb in J mol^-1.

    Notes
    -----
    The formation properties are taken from Table 2-179 of Perry's Chemical
    Engineering Handbook, 8th Edition. At the same time, values in that table
    were taken from the Design Institute for Physical Properties (DIPPR) of
    the American Institute of Chemical Engineers (AIChE), copyright 2007
    AIChE and reproduced with permission of AICHE and of the DIPPR Evaluated
    Process Design Data Project Steering Committee. Their source should be
    cited as R. L. Rowley, W. V. Wilding, J. L. Oscarson,
    Y. Yang, N. A. Zundel, T. E. Daubert, R. P. Danner,
    DIPPR® Data Compilation of Pure Chemical Properties, Design Institute for
    Physical Properties, AIChE, New York (2007).

    """
    check_species(species)
    # -------------------------------------------------------------------------
    # Load coefficients database
    dir = pathlib.Path(__file__).resolve().parent
    database_file = pathlib.Path(dir, "db_formation_properties.json")

    with open(database_file, "r") as jfile:
        data = json.load(jfile)
    # -------------------------------------------------------------------------
    return data


def cp_coeffients(species):
    """
    Load Cp coefficients database.

    Parameters
    ----------
    species : list
        Species to get the Cp data for.

    Returns
    -------
    dict
        Cp data.

    """
    check_species(species)
    # -------------------------------------------------------------------------
    # Load coefficients database
    dir = pathlib.Path(__file__).resolve().parent
    database_file = pathlib.Path(dir, "db_cp.json")

    with open(database_file, "r") as jfile:
        data = json.load(jfile)
    # -------------------------------------------------------------------------
    return data


def check_species(species):
    """
    Check if variable `species` is a list.

    An exeption is raised if not.

    """
    if not isinstance(species, list):
        err = (
            "The variable `species` should be a list.\n"
            + "Check the parameters information.")
        raise ValueError(err)


def peng_robinson(T, P, y, Tc, Pc, w):
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
    k = 0.37464 + 1.54226 * w - 0.26992 * w**2

    alpha = (1 + k * (1 - np.sqrt(T/Tc)))**2

    a = 0.45724 * ((R**2*Tc**2)/Pc)*alpha
    b = 0.07780 * (R*Tc/Pc)
    # -------------------------------------------------------------------------
    # Calculate amix and bmix using the vaan der Waals mixing rule
    aij = np.zeros((a.size, a.size))
    for ii in range(a.size):
        for jj in range(a.size):
            aij[ii, jj] = np.sqrt(a[ii] * a[jj])

    amix = y @ aij @ y
    bmix = y @ b
    # -------------------------------------------------------------------------
    # Calculate A and B
    Aij = (aij * P) / (R * T)**2
    Bi = (b * P) / (R * T)

    Amix = (amix*P)/(R*T)**2
    Bmix = (bmix*P)/(R*T)
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


def heat_capacity(Cp_coeff, T):
    """
    Calculate the ideal gas heat capacity at a given temperature.

    This funtion evaluates the enthalpy of reaction of any reaction at
    a given temperature.

    Parameters
    ----------
    T : float
        Temperature in K.
    Tref : float, optional
        Reference temperature in K.
    H0rxn : float
        Enthalpy of reaction at the standard state in J mol^-1.
    Cp_coefs : array
        Matrix of heat capacity constants fitted to hyperbolic functions.
        Columns: A, B, C, D, E.
        Rows represent different species.
    v : array
        Stoichiometry coefficients of each especies.

    Returns
    -------
    Cp : float
        Heat capacity in J mol^-1 K^-1.

    Notes
    -----
    Cp coefficients should be taken from Table 2-156 of Perry's Chemical
    Engineering Handbook, 8th Edition. At the same time, values in this
    table were taken from the Design Institute for Physical Properties (DIPPR)
    of the American Institute of Chemical Engineers (AIChE), copyright 2007
    AIChE and reproduced with permission of AICHE and of the DIPPR
    Evaluated Process Design Data Project Steering Committee.

    Cp = A + B*((C/T)/sinh(C/T))^2 + D*((E/T)/cosh(E/T))^2
    where Cp is in J kmol^-1 K^-1 and T is in K.

    """
    Cp = (
        Cp_coeff[0]*1e5
        + (Cp_coeff[1]*1e5
            * ((Cp_coeff[2]*1e3 / T) / np.sinh(Cp_coeff[2]*1e3 / T))**2)
        + (Cp_coeff[3]*1e5
            * ((Cp_coeff[4] / T) / np.cosh(Cp_coeff[4] / T))**2))

    #  Cp is converted to J mol^-1 K^-1
    return Cp / 1000


def heat_capacity_rxn(nu, coeff, T):
    """
    Compute the change in the heat capacity due to the reaction.

    Parameters
    ----------
    nu : array
        Stoichiometry coefficients of each especies.
    Cp_coefs : array
        Matrix of heat capacity constants fitted to hyperbolic functions.
        Columns: A, B, C, D, E.
        Rows represent different species.
    T : float
        Temperature in K.

    Returns
    -------
    float
        Cp of reaction in J mol^-1 K^-1.

    """
    Cp = np.zeros(nu.size)
    for ii in range(nu.size):
        Cp[ii] = heat_capacity(coeff[ii], T)

    return np.sum(nu*Cp)


def enthalpy(Hf0, nu, Cp_coeff, T, Tref=298.15):
    """
    Evaluate the enthalpy of reaction at a given temperature.

    Parameters
    ----------
    T : float
        Temperature in K.
    Tref : float, optional
        Reference temperature in K.
    H0rxn : float
        Enthalpy of reaction at the standard state in J mol^-1.
    Cp_coefs : array
        Matrix of heat capacity constants fitted to hyperbolic functions.
        Columns: A, B, C, D, E.
        Rows represent different species.
    nu : array
        Stoichiometry coefficients of each especies.

    Returns
    -------
    Hrxn : float
        Enthalpy of reaction at the given temperature in J mol^-1.

    """
    def integral(T, nu, Cp_coeff):
        return heat_capacity_rxn(nu, Cp_coeff, T)

    return Hf0 + integrate.quad(integral, Tref, T, args=(nu, Cp_coeff))[0]


def entropy(S0, nu, Cp_coeff, T, Tref=298.15):
    """
    Evaluate the entropy of reaction at a given temperature.

    Parameters
    ----------
    T : float
        Temperature in K.
    Tref : float, optional
        Reference temperature in K.
    H0rxn : float
        Enthalpy of reaction at the standard state in J mol^-1.
    Cp_coefs : array
        Matrix of heat capacity constants fitted to hyperbolic functions.
        Columns: A, B, C, D, E.
        Rows represent different species.
    v : array
        Stoichiometry coefficients of each especies.

    Returns
    -------
    Srxn : float
        Entropy of reaction at the given temperature in J mol^-1 K^-1.

    """
    def integral(T, nu, Cp_coeff):
        return heat_capacity_rxn(nu, Cp_coeff, T)/T

    return S0 + integrate.quad(integral, Tref, T, args=(nu, Cp_coeff))[0]


def properties(species, frxns, T, Tref=298.15):
    """
    Calculate the enthalpy, Gibbs energy and entropy of each species at the
    specified temperature.

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
    n = len(species)

    Cp_data = cp_coeffients(species)
    ncoeff = len(Cp_data[species[0]]["coeff"])
    form_props = formation_properties(species)

    Hrxn, Srxn = (np.zeros(n) for _ in range(2))
    for ii in range(n):
        Cp_coeff = np.zeros((len(frxns[ii]["species"]), ncoeff))
        nu = np.asarray(frxns[ii]["nu"])

        Hf0, S0 = (np.zeros(nu.size) for _ in range(2))
        for jj in range(nu.size):
            Cp_coeff[jj, :] = Cp_data[frxns[ii]["species"][jj]]["coeff"]
            Hf0[jj] = form_props[frxns[ii]["species"][jj]]["Hf0"]
            S0[jj] = form_props[frxns[ii]["species"][jj]]["S0"]

        Hrxn0 = np.sum(nu*Hf0)
        Srxn0 = np.sum(nu*S0)

        Hrxn[ii] = enthalpy(Hrxn0, nu, Cp_coeff, T, Tref)
        Srxn[ii] = entropy(Srxn0, nu, Cp_coeff, T, Tref)

    Grxn = Hrxn - T*Srxn

    return Hrxn, Grxn, Srxn


def formation_reactions(species):
    """
    Get the formation reaction for each species.

    Parameters
    ----------
    species : list
        List of strings containing the name of each species involved in the
        equilibrium. For example, ["H2(g)", "O2(g)", "H2O(g)"].

    Returns
    -------
    frxn : list
        Formation reactions for each species.

    """
    check_species(species)
    # -------------------------------------------------------------------------
    # Load coefficients database
    dir = pathlib.Path(__file__).resolve().parent
    database_file = pathlib.Path(dir, "db_formation_reactions.json")

    with open(database_file, "r") as jfile:
        data = json.load(jfile)
    # -------------------------------------------------------------------------
    frxns = []
    for sp in species:
        frxns.append(data[sp])
    # -------------------------------------------------------------------------
    return frxns


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


def atom_stoichiometry(species):
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


def atom_balance(n, species, elements):
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
    n_atom = np.zeros((len(species), len(elements)))
    for ii, spp in enumerate(species):
        atom_stoic = atom_stoichiometry(spp)
        for jj, element in enumerate(elements):
            if element in atom_stoic:
                n_atom[ii, jj] = n[ii]*int(atom_stoic[element])

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


def gibbs_objfun(n, T, P, Grxn, species):
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

    GRT = Grxn/(R*T)

    crit_cons = critical_constants(species)
    a = activity(T, P, y, crit_cons)

    # It it possible for the number of moles of one of the species to become
    # zero or slightly negative during iteration, causing issues with the
    # natural logarithm. Therefore, the warnings during computing the logaritm
    # are ignored.
    with np.errstate(divide="ignore", invalid="ignore"):
        lna = np.log(a)
        value = n*(GRT + lna)

    return np.sum(value)


def gibbs_minimization(T, P, n0, species, elements):
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
    frxns = formation_reactions(species)
    _, Grxn, _ = properties(species, frxns, T)
    atom_moles_init = atom_balance(n0, species, elements)
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
            "fun": (lambda n:
                    atom_moles_init - atom_balance(n, species, elements))
        }]
    # -------------------------------------------------------------------------
    n_eq_init = np.copy(n0)
    n_eq_init[n_eq_init == 0] = 1e-12

    opts = {
        "ftol": 1e-12, "eps": 1e-10, "maxiter": 200}
    sol = optimize.minimize(
        gibbs_objfun, n_eq_init,
        args=(T, P, Grxn, species),
        method="SLSQP", options=opts, constraints=contraints)

    if not sol.success:
        msg = f"Gibbs minimization failed: {sol.message}"
        logger.warning(msg)
    # -------------------------------------------------------------------------
    n_eq = sol.x * scaling_factor
    y_eq = n_eq/np.sum(n_eq)
    # -------------------------------------------------------------------------
    return n_eq, y_eq


def main():
    print("ChemTherm package!")


if __name__ == "__main__":
    main()
