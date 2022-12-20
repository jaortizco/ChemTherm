import numpy as np
import numpy.typing as npt
from scipy import integrate


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
    table were taken from the Design Institute for Physical Properties
    (DIPPR) of the American Institute of Chemical Engineers (AIChE),
    copyright 2007 AIChE and reproduced with permission of AICHE and of
    the DIPPR Evaluated Process Design Data Project Steering Committee.

    Cp = A + B*((C/T)/sinh(C/T))^2 + D*((E/T)/cosh(E/T))^2
    where Cp is in J kmol^-1 K^-1 and T is in K.

    """
    Cp = (
        Cp_coeff[0]
        + (Cp_coeff[1] * ((Cp_coeff[2]/T) / np.sinh(Cp_coeff[2]/T))**2)
        + (Cp_coeff[3] * ((Cp_coeff[4]/T) / np.cosh(Cp_coeff[4]/T))**2))

    #  Cp is converted to J mol^-1 K^-1
    return Cp / 1000


def heat_capacity_rxn(
        nu: npt.NDArray[np.float64], coeff: npt.NDArray[np.float64],
        T: float) -> np.float64:
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
    for i in range(nu.size):
        Cp[i] = heat_capacity(coeff[i], T)

    return np.sum(nu*Cp)


def enthalpy(
        Hf0: np.float64, nu: npt.NDArray, Cp_coeff: npt.NDArray,
        T: float, Tref: float = 298.15) -> float:
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


def entropy(
        S0: np.float64, nu: npt.NDArray, Cp_coeff: npt.NDArray,
        T: float, Tref: float = 298.15) -> float:
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
