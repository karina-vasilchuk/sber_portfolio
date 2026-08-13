"""
Figures for the numerical experiment.

The script reproduces the main figures based on deterministic
and stochastic simulations of the model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from src.parameters import (
    EPSILON_1,
    EPSILON_2,
    MU_1,
    MU_2,
    T,
    TAU_DET,
    THETA_DET,
    THETA_BAR_DET,
    SIGMA_1_BASE,
    SIGMA_2_BASE,
    RHO_BASE,
)

from experiments.numerical_experiment import (
    production,
    simulate_batch,
    simulate_trajectory,
    z_sng,
)


# ============================================================
# Настройки графиков
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "figure.figsize": (7, 5.5),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================
# Детерминированная оптимальная траектория
# ============================================================

def det_optimal_trajectory(num_points=500):

    def sys_01(t, y):
        x1, x2 = y
        F = production(x1, x2)

        return [
            -MU_1 * x1,
            F / EPSILON_2 - MU_2 * x2,
        ]

    sol1 = solve_ivp(
        sys_01,
        (0, TAU_DET),
        [2.0, 1.0],
        t_eval=np.linspace(
            0,
            TAU_DET,
            num_points,
        ),
    )

    def sys_sng(t, y):
        x1, x2 = y
        F = production(x1, x2)

        return [
            F - MU_1 * x1,
            F - MU_2 * x2,
        ]

    sol2 = solve_ivp(
        sys_sng,
        (TAU_DET, THETA_DET),
        [
            sol1.y[0, -1],
            sol1.y[1, -1],
        ],
        t_eval=np.linspace(
            TAU_DET,
            THETA_DET,
            num_points,
        ),
    )

    def sys_10(t, y):
        x1, x2 = y
        F = production(x1, x2)

        return [
            F / EPSILON_1 - MU_1 * x1,
            -MU_2 * x2,
        ]

    sol3 = solve_ivp(
        sys_10,
        (THETA_DET, THETA_BAR_DET),
        [
            sol2.y[0, -1],
            sol2.y[1, -1],
        ],
        t_eval=np.linspace(
            THETA_DET,
            THETA_BAR_DET,
            num_points,
        ),
    )

    def sys_00(t, y):
        x1, x2 = y

        return [
            -MU_1 * x1,
            -MU_2 * x2,
        ]

    sol4 = solve_ivp(
        sys_00,
        (THETA_BAR_DET, T),
        [
            sol3.y[0, -1],
            sol3.y[1, -1],
        ],
        t_eval=np.linspace(
            THETA_BAR_DET,
            T,
            num_points,
        ),
    )

    t = np.concatenate([
        sol1.t,
        sol2.t[1:],
        sol3.t[1:],
        sol4.t[1:],
    ])

    x1 = np.concatenate([
        sol1.y[0],
        sol2.y[0, 1:],
        sol3.y[0, 1:],
        sol4.y[0, 1:],
    ])

    x2 = np.concatenate([
        sol1.y[1],
        sol2.y[1, 1:],
        sol3.y[1, 1:],
        sol4.y[1, 1:],
    ])

    return t, x1, x2


# ============================================================
# Рисунок 6.1
# ============================================================

def fig_6_1():
    t, x1, x2 = det_optimal_trajectory()

    idx_tau = np.argmin(
        np.abs(t - TAU_DET)
    )

    idx_theta = np.argmin(
        np.abs(t - THETA_DET)
    )

    idx_theta_bar = np.argmin(
        np.abs(t - THETA_BAR_DET)
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        x1,
        x2,
        "k-",
        linewidth=2,
        label="optimal trajectory",
    )

    x1_ray = np.linspace(
        0,
        max(x1) * 1.1,
        100,
    )

    plt.plot(
        x1_ray,
        z_sng * x1_ray,
        "gray",
        linestyle=":",
        label=r"$L_{\mathrm{sng}}$",
    )

    points = [
        (0, r"$x^0$"),
        (idx_tau, r"$x(\tau)$"),
        (idx_theta, r"$x(\theta)$"),
        (idx_theta_bar, r"$x(\bar{\theta})$"),
        (-1, r"$x(T)$"),
    ]

    for idx, label in points:
        plt.plot(
            x1[idx],
            x2[idx],
            "ko",
        )

        plt.text(
            x1[idx] + 0.05,
            x2[idx] - 0.12,
            label,
        )

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    plt.grid(
        True,
        linestyle=":",
        alpha=0.5,
    )

    plt.legend(
        loc="upper left",
        frameon=False,
    )

    plt.title(
        "Deterministic optimal trajectory"
    )

    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "deterministic_phase_trajectory.pdf"
    )

    plt.show()


# ============================================================
# Рисунок 6.2
# ============================================================

def fig_6_2(
    N_mean=200,
    N_vis=100,
    seed=42,
):
    rng = np.random.RandomState(seed)

    t_common = np.linspace(
        0,
        T,
        500,
    )

    all_z = np.zeros(
        (N_mean, len(t_common))
    )

    for i in range(N_mean):
        (
            t,
            x1,
            x2,
            _,
            _,
            _,
            _,
        ) = simulate_trajectory(
            SIGMA_1_BASE,
            SIGMA_2_BASE,
            RHO_BASE,
            rng,
        )

        z = x2 / x1

        all_z[i] = np.interp(
            t_common,
            t,
            z,
        )

    mean_z = np.mean(
        all_z,
        axis=0,
    )

    std_z = np.std(
        all_z,
        axis=0,
    )

    indices = rng.choice(
        N_mean,
        size=min(N_vis, N_mean),
        replace=False,
    )

    plt.figure()

    for idx in indices:
        plt.plot(
            t_common,
            all_z[idx],
            color="gray",
            alpha=0.35,
            linewidth=0.6,
        )

    plt.plot(
        t_common,
        mean_z,
        "k-",
        linewidth=2,
        label=r"$\mathbb{E}[z(t)]$",
    )

    plt.fill_between(
        t_common,
        mean_z - 1.96 * std_z,
        mean_z + 1.96 * std_z,
        color="gray",
        alpha=0.2,
        label="95% interval",
    )

    t_det, x1_det, x2_det = (
        det_optimal_trajectory()
    )

    z_det = x2_det / x1_det

    plt.plot(
        t_det,
        z_det,
        "k--",
        linewidth=1.5,
        label=r"$z^{(0)}(t)$",
    )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$z(t)$")

    plt.grid(
        True,
        linestyle=":",
        alpha=0.5,
    )

    plt.legend(
        frameon=False,
    )

    plt.title(
        "Stochastic trajectories of $z(t)$"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "stochastic_z_trajectories.pdf"
    )

    plt.show()


# ============================================================
# Доверительные интервалы x1 и x2
# ============================================================

def confidence_paths(
    N=200,
    seed=42,
):
    rng = np.random.RandomState(seed)

    t_common = np.linspace(
        0,
        T,
        500,
    )

    x1_paths = np.zeros(
        (N, len(t_common))
    )

    x2_paths = np.zeros(
        (N, len(t_common))
    )

    for i in range(N):
        (
            t,
            x1,
            x2,
            _,
            _,
            _,
            _,
        ) = simulate_trajectory(
            SIGMA_1_BASE,
            SIGMA_2_BASE,
            RHO_BASE,
            rng,
        )

        x1_paths[i] = np.interp(
            t_common,
            t,
            x1,
        )

        x2_paths[i] = np.interp(
            t_common,
            t,
            x2,
        )

    return (
        t_common,
        x1_paths,
        x2_paths,
    )


# ============================================================
# Рисунок 6.3a
# ============================================================

def fig_6_3_x1(N=200):
    (
        t,
        x1_paths,
        _,
    ) = confidence_paths(N)

    mean_x1 = np.mean(
        x1_paths,
        axis=0,
    )

    std_x1 = np.std(
        x1_paths,
        axis=0,
    )

    t_det, x1_det, _ = (
        det_optimal_trajectory()
    )

    plt.figure()

    plt.plot(
        t,
        mean_x1,
        "k-",
        label=r"$\mathbb{E}[x_1(t)]$",
    )

    plt.fill_between(
        t,
        mean_x1 - 1.96 * std_x1,
        mean_x1 + 1.96 * std_x1,
        color="gray",
        alpha=0.3,
        label="95% interval",
    )

    plt.plot(
        t_det,
        x1_det,
        "k--",
        linewidth=1,
        label=r"$x_1^{(0)}(t)$",
    )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$x_1(t)$")

    plt.grid(
        True,
        linestyle=":",
        alpha=0.5,
    )

    plt.legend(
        frameon=False,
    )

    plt.title(
        "Monte Carlo interval for $x_1(t)$"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "x1_confidence_interval.pdf"
    )

    plt.show()


# ============================================================
# Рисунок 6.3b
# ============================================================

def fig_6_3_x2(N=200):
    (
        t,
        _,
        x2_paths,
    ) = confidence_paths(N)

    mean_x2 = np.mean(
        x2_paths,
        axis=0,
    )

    std_x2 = np.std(
        x2_paths,
        axis=0,
    )

    t_det, _, x2_det = (
        det_optimal_trajectory()
    )

    plt.figure()

    plt.plot(
        t,
        mean_x2,
        "k-",
        label=r"$\mathbb{E}[x_2(t)]$",
    )

    plt.fill_between(
        t,
        mean_x2 - 1.96 * std_x2,
        mean_x2 + 1.96 * std_x2,
        color="gray",
        alpha=0.3,
        label="95% interval",
    )

    plt.plot(
        t_det,
        x2_det,
        "k--",
        linewidth=1,
        label=r"$x_2^{(0)}(t)$",
    )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$x_2(t)$")

    plt.grid(
        True,
        linestyle=":",
        alpha=0.5,
    )

    plt.legend(
        frameon=False,
    )

    plt.title(
        "Monte Carlo interval for $x_2(t)$"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "x2_confidence_interval.pdf"
    )

    plt.show()


# ============================================================
# Рисунок 6.4
# ============================================================

def fig_6_4(N=5000):
    rho_values = [
        -0.5,
        0.0,
        0.5,
    ]

    E_tau = []
    E_theta_bar = []
    E_J = []

    for rho in rho_values:
        (
            taus,
            _,
            theta_bars,
            Js,
        ) = simulate_batch(
            N,
            SIGMA_1_BASE,
            SIGMA_2_BASE,
            rho,
        )

        E_tau.append(
            np.mean(taus)
        )

        E_theta_bar.append(
            np.mean(theta_bars)
        )

        E_J.append(
            np.mean(Js)
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].plot(
        rho_values,
        E_tau,
        "ko-",
    )

    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"$\mathbb{E}[\tau]$")

    axes[1].plot(
        rho_values,
        E_theta_bar,
        "ko-",
    )

    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel(
        r"$\mathbb{E}[\bar{\theta}]$"
    )

    axes[2].plot(
        rho_values,
        E_J,
        "ko-",
    )

    axes[2].set_xlabel(r"$\rho$")
    axes[2].set_ylabel(
        r"$\mathbb{E}[J]$"
    )

    for ax in axes:
        ax.grid(
            True,
            linestyle=":",
            alpha=0.5,
        )

    plt.suptitle(
        "Influence of noise correlation"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rho_influence.pdf"
    )

    plt.show()


# ============================================================
# Рисунок 6.5
# ============================================================

def fig_6_5(N=5000):
    sigma_values = [
        0.10,
        0.15,
        0.20,
    ]

    E_tau = []
    E_theta_bar = []
    E_J = []

    for sigma in sigma_values:
        (
            taus,
            _,
            theta_bars,
            Js,
        ) = simulate_batch(
            N,
            sigma,
            sigma,
            0.0,
        )

        E_tau.append(
            np.mean(taus)
        )

        E_theta_bar.append(
            np.mean(theta_bars)
        )

        E_J.append(
            np.mean(Js)
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].plot(
        sigma_values,
        E_tau,
        "ko-",
    )

    axes[0].set_xlabel(r"$\sigma$")
    axes[0].set_ylabel(r"$\mathbb{E}[\tau]$")

    axes[1].plot(
        sigma_values,
        E_theta_bar,
        "ko-",
    )

    axes[1].set_xlabel(r"$\sigma$")
    axes[1].set_ylabel(
        r"$\mathbb{E}[\bar{\theta}]$"
    )

    axes[2].plot(
        sigma_values,
        E_J,
        "ko-",
    )

    axes[2].set_xlabel(r"$\sigma$")
    axes[2].set_ylabel(
        r"$\mathbb{E}[J]$"
    )

    for ax in axes:
        ax.grid(
            True,
            linestyle=":",
            alpha=0.5,
        )

    plt.suptitle(
        "Influence of volatility"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "sigma_influence.pdf"
    )

    plt.show()


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":

    fig_6_1()
    fig_6_2()

    fig_6_3_x1()
    fig_6_3_x2()

    fig_6_4()
    fig_6_5()