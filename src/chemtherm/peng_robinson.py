import numpy as np
import numpy.typing as npt
from chemtherm import phycons as pc


def peng_robinson(
        T: float, P: float, y: npt.NDArray[np.float64],
        Tc: npt.NDArray[np.float64], Pc: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
    k = 0.37464 + 1.54226*w - 0.26992*w**2

    alpha = (1 + k*(1-np.sqrt(T/Tc)))**2

    a = 0.45724*((pc.R**2*Tc**2)/Pc)*alpha
    b = 0.07780*(pc.R*Tc/Pc)
    # -------------------------------------------------------------------------
    aij, amix, bmix = _vaanderWaals_mixing_rule(y, a, b)
    # -------------------------------------------------------------------------
    # Calculate A and B
    Aij = (aij*P)/(pc.R*T)**2
    Bi = (b*P)/(pc.R*T)

    Amix = (amix*P)/(pc.R*T)**2
    Bmix = (bmix*P)/(pc.R*T)
    # -------------------------------------------------------------------------
    Zmix = _compressibility_factor(Amix, Bmix)
    lnphi = _calculate_lnphi(y, Aij, Bi, Amix, Bmix, Zmix)

    return np.exp(lnphi)


def _vaanderWaals_mixing_rule(
        y: npt.NDArray[np.float64], a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Calculate amix and bmix using the vaan der Waals mixing rule.

    Notes
    -----
    At the momment the binary interaction parameter kij is assumed to be 1.

    """
    aij = np.sqrt(np.outer(a, a))

    amix = y@aij@y
    bmix = y@b

    return aij, amix, bmix  # type: ignore


def _compressibility_factor(Amix: float, Bmix: float) -> float:
    """
    Calculate liquid and gas compressibility factor by finding the roots of
    the third grade polynomial.

    """
    poly = np.polynomial.Polynomial([
        (-Amix*Bmix + Bmix**2 + Bmix**3),
        (Amix - 3*Bmix**2 - 2*Bmix), (-1+Bmix), 1])

    Z = poly.roots()
    Z = np.real(Z)

    return np.max(Z)


def _calculate_lnphi(
        y: npt.NDArray[np.float64], Aij: npt.NDArray[np.float64],
        Bi: npt.NDArray[np.float64], Amix: float, Bmix: float, Zmix: float
        ) -> npt.NDArray[np.float64]:
    """
    Calculate the natural logarithm of the fugacity coefficients.

    """
    A = (Bi/Bmix) * (Zmix-1)
    B = Zmix - Bmix
    C = (Amix/(2*np.sqrt(2)*Bmix)) * ((2*(Aij@y)/Amix)-(Bi/Bmix))
    D = (Zmix+(1+np.sqrt(2))*Bmix) / (Zmix+(1-np.sqrt(2))*Bmix)

    return A - np.log(B) - C*np.log(D)
