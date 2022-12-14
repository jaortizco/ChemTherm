"""
Module with some calculation examples.

"""
import time

import matplotlib.pyplot as plt
import numpy as np

from chemtherm import chemeq
from chemtherm.mixture import Mixture
from chemtherm.species import Species


def example_smith_vanness_abbott():
    """
    Example of a complex gas-phase equilibrium calculation.

    Smith, Van Ness, and Abbott [Introduction to Chemical Engineering
    Thermodynamics, 5th ed., Example 15.13, pp. 602-604; 6th ed.,
    Example 13.14, pp. 511-513; 7th ed., Example 13.14, pp. 527-528;
    8th ed., Example 14.14, pp. 568-569; 9th ed., Example 14.14, pp. 580-582,
    McGraw-Hill, New York (1996, 2001, 2005, 2017, 2022)].

    Notes
    -----
    General application of the method to multicomponent, multiphase systems
    is treated by Iglesias-Silva et al. [Fluid Phase Equilib. 210: 229-245
    (2003)] and by Sotyan, Ghajar, and Gasem [Ind. Eng. Chem. Res. 42:
    3786-3801 (2003)].

    """
    P = 1  # bar
    T = 1000  # °C --> K
    # -------------------------------------------------------------------------
    species_names = ["CH4(g)", "H2O(g)", "CO(g)", "H2(g)", "CO2(g)"]

    species_list = []
    for name in species_names:
        species_list.append(Species(name))

    mix = Mixture(species_list)
    # -------------------------------------------------------------------------
    n0 = np.array([2, 3, 0, 0, 0])
    # -------------------------------------------------------------------------
    n_eq, y_eq = chemeq.gibbs_minimization(T, P, n0, mix)
    # -------------------------------------------------------------------------
    y_eq_example = np.array([0.0196, 0.0980, 0.1743, 0.6710, 0.0371])
    diff = np.abs(y_eq_example - y_eq) / y_eq_example * 100
    # -------------------------------------------------------------------------
    data = np.column_stack((y_eq, y_eq_example, diff))

    print("Comparison: Book (Smith, Van Ness, Abbott)")
    print(f"Temperature: {T} K; Pressure: {P} bar")
    print()
    print(["Calculated y", "Reported y", "Difference (%)"])
    print(data)


def example_web_app():
    """
    Example of a complex gas-phase equilibrium calculation.

    Compare results with thoses obtained using the web application from
    Colorado State University.
    https://navier.engr.colostate.edu/code/code-4/index.html

    """
    P = 3  # bar
    T = 950  # °C --> K
    # -------------------------------------------------------------------------
    species_names = ["CH4(g)", "H2O(g)", "CO(g)", "H2(g)", "CO2(g)"]

    species_list = []
    for name in species_names:
        species_list.append(Species(name))

    mix = Mixture(species_list)
    # -------------------------------------------------------------------------
    n0 = np.array([1, 1, 0, 0, 0])
    # -------------------------------------------------------------------------
    n_eq, y_eq = chemeq.gibbs_minimization(T, P, n0, mix)
    # -------------------------------------------------------------------------
    y_eq_example = np.array([0.17977, 0.13054, 0.11089, 0.52958, 0.049229])
    diff = np.abs(y_eq_example - y_eq) / y_eq_example * 100
    # -------------------------------------------------------------------------
    data = np.column_stack((y_eq, y_eq_example, diff))

    print("Comparison: Book (Smith, Van Ness, Abbott)")
    print(f"Temperature: {T} K; Pressure: {P} bar")
    print()
    print(["Calculated y", "Reported y", "Difference (%)"])
    print(data)


def example_plot():
    """
    Plot equilibrium mole flows.

    """
    P = 1  # bar
    T = np.linspace(100, 800, 100) + 273.15  # °C --> K
    # -------------------------------------------------------------------------
    species_names = ["CH4(g)", "H2O(g)", "CO(g)", "H2(g)", "CO2(g)"]

    species_list = []
    for name in species_names:
        species_list.append(Species(name))

    mix = Mixture(species_list)
    # -------------------------------------------------------------------------
    n0 = np.array([7.89E-03, 1.97E-02, 0, 0, 0])
    # -------------------------------------------------------------------------
    n_eq, y_eq = (np.zeros((T.size, n0.size)) for _ in range(2))
    for ii in range(T.size):
        n_eq[ii, :], y_eq[ii, :] = chemeq.gibbs_minimization(
            T[ii], P, n0, mix)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(T-273.15, n_eq)
    # ax.legend(species, frameon=False, loc="best", fontsize=14, ncol=2)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Equilibrium amount (kmol)")
    ax.set_xlim(100, 800)
    ax.set_ylim(0, 0.03)

    fig.savefig("plot.svg", format="svg")
    # -------------------------------------------------------------------------
    _, ax = plt.subplots(constrained_layout=True)

    ax.plot(T-273.15, y_eq*100)
    # ax.legend(species, frameon=False, loc="best", fontsize=14, ncol=2)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Mole fraction (%)")

    ax.set_ylim(0, 100)


def equilibrium_constant():
    T = 600 + 273.15  # °C --> K
    # -------------------------------------------------------------------------
    species_names = ["H2O(g)", "H2(g)", "O2(g)"]

    species_list = []
    for name in species_names:
        species_list.append(Species(name))

    mix = Mixture(species_list)

    nu = np.array([1, -1, -0.5])
    Kp, _Hrxn, _Grxn, _Srxn = chemeq.eq_cons(mix, nu, T)
    print(Kp)


def main():
    start_time = time.perf_counter()

    # example_smith_vanness_abbott()
    # example_web_app()
    example_plot()
    # equilibrium_constant()

    run_time = time.perf_counter() - start_time
    print(run_time)

    plt.show()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()")
    main()
