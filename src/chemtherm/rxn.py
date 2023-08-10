import numpy as np
import numpy.typing as npt
from scipy import integrate


def heat_capacity(T: float, coeff: npt.NDArray[np.float64]) -> float:
    """
    Calculate the ideal gas heat capacity at a given temperature.

    Parameters
    ----------
    T : float
        Temperature in K.
    coeff : array_like
        Heat capacity coefficients.

    Returns
    -------
    Cp : float
        Heat capacity in J mol^-1 K^-1.

    Notes
    -----
    Cp coefficients are taken from the Chemical Properties Handbook: Physical,
    Thermodynamics, Environmental Transport, Safety & Health Related
    Properties for Organic & Chemical engineering, 1st Edition. Carl L. Yaws.
    McGraw-Hill Education, 1999. ISBN 0070734011, 9780070734012.

    Cp = A + B*T + C*T^2 + D*T^3 + E*T^4
    where Cp is in J mol^-1 K^-1 and T is in K.

    """
    return (
        coeff[0] + coeff[1]*T + coeff[2]*T**2 + coeff[3]*T**3 + coeff[4]*T**4
    )


def enthalpy_integral(
    T: float,
    Cp_coeff: npt.NDArray,
    Tref: float = 298.15,
) -> float:
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
        Matrix of heat capacity coefficients.
        Columns: A, B, C, D, E.
        Rows represent different species.
    nu : array
        Stoichiometry coefficients of each especies.

    Returns
    -------
    Hrxn : float
        Enthalpy of reaction at the given temperature in J mol^-1.

    """

    def integral(T, Cp_coeff):
        return heat_capacity(T, Cp_coeff)

    return integrate.quad(integral, Tref, T, args=(Cp_coeff))[0]


def entropy_integral(
    T: float,
    Cp_coeff: npt.NDArray,
    Tref: float = 298.15,
) -> float:
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
        Matrix of heat capacity coefficients.
        Columns: A, B, C, D, E.
        Rows represent different species.
    v : array
        Stoichiometry coefficients of each especies.

    Returns
    -------
    Srxn : float
        Entropy of reaction at the given temperature in J mol^-1 K^-1.

    """

    def integral(T, Cp_coeff):
        return heat_capacity(T, Cp_coeff)/T

    return integrate.quad(integral, Tref, T, args=(Cp_coeff))[0]


def reaction_properties(
    T: float,
    Hf0: npt.NDArray[np.float64],
    S0: npt.NDArray[np.float64],
    Cp_coeff: npt.NDArray[np.float64],
    nu: npt.NDArray[np.int32],
    Tref: float = 298.15,
) -> tuple[float, float, float]:
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
        Matrix of heat capacity coefficients.
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

    Hrxn0 = nu @ Hf0
    Srxn0 = nu @ S0
    Cprxn_coeff = nu @ Cp_coeff

    Hrxn = Hrxn0 + enthalpy_integral(T, Cprxn_coeff, Tref)
    Srxn = Srxn0 + entropy_integral(T, Cprxn_coeff, Tref)
    Grxn = Hrxn - T*Srxn

    return Hrxn, Grxn, Srxn  # type: ignore
