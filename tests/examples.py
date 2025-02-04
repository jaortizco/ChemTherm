"""
Module with some calculation examples.

"""
import matplotlib.pyplot as plt
import numpy as np

from chemtherm import chemeq, utils
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
    T = 1000  # K
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
    diff = np.abs(y_eq_example - y_eq)/y_eq_example*100
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
    diff = np.abs(y_eq_example - y_eq)/y_eq_example*100
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
    for i, Ti in enumerate(T):
        n_eq[i, :], y_eq[i, :] = chemeq.gibbs_minimization(Ti, P, n0, mix)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(T - 273.15, n_eq)
    # ax.legend(species, frameon=False, loc="best", fontsize=14, ncol=2)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Equilibrium amount (kmol)")
    ax.set_xlim(100, 800)
    ax.set_ylim(0, 0.03)

    fig.savefig("plot.svg", format="svg")
    # -------------------------------------------------------------------------
    _, ax = plt.subplots(constrained_layout=True)

    ax.plot(T - 273.15, y_eq*100)
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


def viscosity_pure_substance():
    """
    Example prediction of the viscosity of molecular nitrogen at 227°C.

    Chung's method is used.

    Notes
    -----
    Experimental data taken from Table 9-2 of The Properties of Gases and
    Liquids (5th edition). McGraw-Hill Professional Pub.  Poling, B. E.,
    Prausnitz, J. M., & O'Connell, J. P. (2000).

    """
    T = 227 + 273.15  # °C --> K
    exp_value = 258

    species = Species("N2(g)")

    calc_value = species.calculate_gas_viscosity(T)
    error = (calc_value-exp_value)/exp_value*100

    print("Viscosity calculation for molecular nitrogen")
    print("Calculated    Experimental    Difference (%)")
    print(f"{calc_value:.3f}       {exp_value:.3f}         {error:.3f}")
    print()


def viscosity_mixture():
    """
    Example prediction of the viscosity of a nitrogen-hydrogen mixture
    at 100°C.

    Reichenberg's method is used.

    Notes
    -----
    Experimental data taken from Table 9-4 of The Properties of Gases and
    Liquids (5th edition). McGraw-Hill Professional Pub. Poling, B. E.,
    Prausnitz, J. M., & O'Connell, J. P. (2000).

    """
    T = 100 + 273.15  # °C --> K
    y1 = 0.51
    exp_value = 190.3

    species_names = ["N2(g)", "H2(g)"]

    species_list = []
    for name in species_names:
        species_list.append(Species(name))

    mix = Mixture(species_list)

    calc_value = mix.calculate_viscosity(T, np.array([y1, 1 - y1]))
    error = (calc_value-exp_value)/exp_value*100

    print("Viscosity calculation for a nitrogen-hydrogen mixture")
    print("Calculated    Experimental    Difference (%)")
    print(f"{calc_value:.3f}       {exp_value:.3f}         {error:.3f}")
    print()


def chemical_potential():
    # P = 1  # bar
    T = np.linspace(100, 1200, 100) + 273.15  # °C --> K
    # -------------------------------------------------------------------------
    species = Species("CH4(g)")

    mu = np.zeros(T.size)
    for i, Ti in enumerate(T):
        _, mu[i], _ = species.thermodynamic_properties(Ti)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(T - 273.15, mu*1e-3)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Chemical potential ($\mathrm{kJ \,\, mol^{-1}}$)")


def ammonia_equilibrium():
    """
    Calculation of ammonia equilibrium as function of temperature at a
    specific pressure.

    Comparison with the following paper:
    Ristig, S. et al. Ammonia Decomposition in the Process Chain for a
    Renewable Hydrogen Supply. Chem. Ing. Tech. 94, 1413–1425 (2022).

    """
    P = 100  # bar
    T = np.linspace(200, 800, 100) + 273.15  # °C --> K
    # -------------------------------------------------------------------------
    species_names = ["H2(g)", "N2(g)", "NH3(g)"]

    species_list = []
    for name in species_names:
        species_list.append(Species(name))

    mix = Mixture(species_list)
    # -------------------------------------------------------------------------
    n0 = np.array([0, 0, 1])
    # -------------------------------------------------------------------------
    n_eq, y_eq = (np.zeros((T.size, n0.size)) for _ in range(2))
    for i, Ti in enumerate(T):
        n_eq[i, :], y_eq[i, :] = chemeq.gibbs_minimization(Ti, P, n0, mix)
    # -------------------------------------------------------------------------
    _, ax = plt.subplots(constrained_layout=True)

    ax.plot(T - 273.15, y_eq)
    ax.legend(species_names, frameon=False, loc="best")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Mole fraction ($-$)")
    # -------------------------------------------------------------------------
    _, ax = plt.subplots(constrained_layout=True)

    ax.semilogy(T - 273.15, y_eq[:, 2])
    ax.axhline(y=0.01, linestyle='--', color="k")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Mole fraction ($-$)")

    import matplotlib.ticker as ticker
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
        )


# @utils.profiler()
@utils.timer
def main():
    # example_smith_vanness_abbott()
    # example_web_app()
    # example_plot()
    # equilibrium_constant()

    # viscosity_pure_substance()
    # viscosity_mixture()

    # chemical_potential()
    ammonia_equilibrium()


if __name__ == "__main__":
    main()
    plt.show()
