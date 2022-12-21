import numpy as np
import numpy.typing as npt
from scipy import integrate


def heat_capacity(T: float, coeff: npt.NDArray[np.float64]) -> float:
    """
    Calculate the ideal gas heat capacity at a given temperature.

    This funtion evaluates the enthalpy of reaction of any reaction at
    a given temperature.

    Parameters
    ----------
    T : float
        Temperature in K.
    coeff : array_like
        Heat capacity coefficients coefficients fitted to hyperbolic functions.

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
        coeff[0]
        + (coeff[1] * ((coeff[2]/T) / np.sinh(coeff[2]/T))**2)
        + (coeff[3] * ((coeff[4]/T) / np.cosh(coeff[4]/T))**2))

    #  Cp is converted to J mol^-1 K^-1
    return Cp / 1000


def heat_capacity_rxn(
        T: float, nu: npt.NDArray[np.float64], coeff: npt.NDArray[np.float64],
        ) -> float:
    """
    Compute the change in the heat capacity due to the reaction.

    Parameters
    ----------
    T : float
        Temperature in K.
    nu : array
        Stoichiometry coefficients of each especies.
    coeff : array_like
        Matrix of heat capacity constants fitted to hyperbolic functions.
        Rows: represent different species.
        Columns: A, B, C, D, E.

    Returns
    -------
    float
        Cp of reaction in J mol^-1 K^-1.

    """
    Cp = np.zeros(nu.size)
    for i in range(nu.size):
        Cp[i] = heat_capacity(T, coeff[i])

    return nu@Cp  # type: ignore


def enthalpy_integral(
        T: float, nu: npt.NDArray, Cp_coeff: npt.NDArray,
        Tref: float = 298.15) -> float:
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
        return heat_capacity_rxn(T, nu, Cp_coeff)

    return integrate.quad(integral, Tref, T, args=(nu, Cp_coeff))[0]


def entropy_integral(
        T: float, nu: npt.NDArray, Cp_coeff: npt.NDArray,
        Tref: float = 298.15) -> float:
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
        return heat_capacity_rxn(T, nu, Cp_coeff)/T

    return integrate.quad(integral, Tref, T, args=(nu, Cp_coeff))[0]


def reaction_properties(
        T: float, Hf0: npt.NDArray[np.float64], S0: npt.NDArray[np.float64],
        Cp_coeff: npt.NDArray[np.float64], nu: npt.NDArray[np.int32],
        Tref: float = 298.15) -> tuple[float, float, float]:
    """
    Compute the enthalpy, the gibbs energy, and the entropy of a given
    reaction at a specified temperature.

    Parameters
    ----------
    T : float
        Temperature in K.
    Hf0 : array_like
        Standard enthalpy of formation of each species in J mol^-1.
    S0 : array_like
        Standard entropy of each species in J mol^-1 K^-1.
    Cp_coeff : array_like
        Matrix of heat capacity constants fitted to hyperbolic functions.
        Rows: represent different species.
        Columns: A, B, C, D, E.
    nu : array
        Stoichiometry coefficients of each especies.
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

    Hrxn0 = nu@Hf0
    Srxn0 = nu@S0

    Hrxn = Hrxn0 + enthalpy_integral(T, nu, Cp_coeff, Tref)
    Srxn = Srxn0 + entropy_integral(T, nu, Cp_coeff, Tref)
    Grxn = Hrxn - T*Srxn

    return Hrxn, Grxn, Srxn  # type: ignore
