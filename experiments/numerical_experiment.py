"""
Numerical experiment for the stochastic optimal control model.

The script implements the numerical experiment described in Chapter 6:
- deterministic switching points;
- baseline Monte Carlo experiment;
- influence of noise correlation;
- influence of volatility.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad

from pathlib import Path
import csv

from src.parameters import (
    EPSILON_1,
    EPSILON_2,
    MU_1,
    MU_2,
    MU,
    NU,
    T,
    X1_0,
    X2_0,
    Z_0,
    TAU_DET,
    THETA_DET,
    THETA_BAR_DET,
    SIGMA_1_BASE,
    SIGMA_2_BASE,
    RHO_BASE,
    SIGMA_THETA_BAR,
    DT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Базовые настройки эксперимента
# ============================================================

N_SIM = 10_000
SEED = 42

RHO_VALUES = [-0.5, 0.0, 0.5]
SIGMA_VALUES = [0.10, 0.15, 0.20]

J_DET = 2.451


# ============================================================
# Сингулярное значение z
# ============================================================

def z_sng_val(eps1, mu_val):
    eps2 = 1.0 - eps1

    def f(z):
        return z**eps2 - z**(-eps1) - mu_val

    if mu_val == 0:
        return 1.0

    if mu_val > 0:
        lo, hi = 1.0, 1.0
        while f(hi) < 0:
            hi *= 2.0
    else:
        lo, hi = 0.001, 1.0
        while f(lo) > 0:
            lo /= 2.0

    for _ in range(100):
        mid = (lo + hi) / 2.0

        if f(mid) < 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


z_sng = z_sng_val(EPSILON_1, MU)


# ============================================================
# Поправка к tau
# ============================================================

def compute_D_tau():
    """
    Коэффициент D_tau для поправки второго порядка
    к моменту выхода на сингулярный луч.
    """

    def z0_fun(t):
        return (
            -1 / (EPSILON_2 * MU)
            + np.exp(EPSILON_1 * MU * t)
            * (
                1 / (EPSILON_2 * MU)
                + Z_0**EPSILON_1
            )
        ) ** (1 / EPSILON_1)

    z_avg, _ = quad(z0_fun, 0, TAU_DET)
    z_avg /= TAU_DET

    dz_avg = (z_sng - Z_0) / TAU_DET

    D_tau = z_avg * TAU_DET / dz_avg

    return D_tau


D_tau = compute_D_tau()


# ============================================================
# Детерминированная функция Q0 и поправка к theta_bar
# ============================================================

A1 = EPSILON_1 * MU * z_sng**(-EPSILON_2)
A2 = EPSILON_2 * MU * z_sng**EPSILON_1


def W_fun(t):
    return np.exp(
        EPSILON_2 * MU * (t - THETA_DET)
    ) - 1.0


def Q0(t):
    return (
        (EPSILON_1 / EPSILON_2)
        * (A2 - W_fun(t))
        / (A1 + W_fun(t))
    )


def dQ0_dt(t):
    W = W_fun(t)
    dW = EPSILON_2 * MU * (W + 1.0)

    return (
        -(EPSILON_1 / EPSILON_2)
        * (A1 + A2)
        * dW
        / (A1 + W) ** 2
    )


B_val, _ = quad(Q0, THETA_DET, THETA_BAR_DET)

C_rho = B_val / abs(
    dQ0_dt(THETA_BAR_DET)
)


# ============================================================
# Производственная функция
# ============================================================

def production(x1, x2):
    return x1**EPSILON_1 * x2**EPSILON_2


# ============================================================
# Моменты переключения
# ============================================================

def switching_times(sigma1, sigma2, rho, rng):
    """
    Генерация моментов переключения для одной реализации.

    tau получает поправку второго порядка.
    theta остаётся в нулевом приближении.
    theta_bar содержит детерминированную поправку
    и случайную нормальную компоненту.
    """

    noise_term = (
        sigma1**2
        - rho * sigma1 * sigma2
    )

    delta_tau = -D_tau * noise_term
    tau = TAU_DET + delta_tau

    theta = THETA_DET

    delta_theta_bar = C_rho * noise_term

    theta_bar = (
        THETA_BAR_DET
        + delta_theta_bar
        + rng.normal(0, SIGMA_THETA_BAR)
    )

    tau = max(0.0, tau)
    theta = max(tau + 0.01, theta)

    theta_bar = max(
        theta + 0.01,
        min(theta_bar, T - 0.01),
    )

    return tau, theta, theta_bar


# ============================================================
# Управление
# ============================================================

def control(t, tau, theta, theta_bar):
    if t < tau:
        return 0.0, 1.0

    if t < theta:
        return EPSILON_1, EPSILON_2

    if t < theta_bar:
        return 1.0, 0.0

    return 0.0, 0.0


# ============================================================
# Один шаг Эйлера-Маруямы
# ============================================================

def sde_step(x1, x2, u1, u2, sigma1, sigma2, rho, rng):
    F = production(x1, x2)

    xi1 = rng.normal()
    xi2 = rng.normal()

    dW1 = np.sqrt(DT) * xi1

    dW2 = np.sqrt(DT) * (
            rho * xi1
            + np.sqrt(1 - rho ** 2) * xi2
    )

    dx1 = (
        (u1 / EPSILON_1 * F - MU_1 * x1) * DT
        + sigma1 * x1 * dW1
    )

    dx2 = (
        (u2 / EPSILON_2 * F - MU_2 * x2) * DT
        + sigma2 * x2 * dW2
    )

    return x1 + dx1, x2 + dx2


# ============================================================
# Одна стохастическая реализация
# ============================================================

def simulate_once(sigma1, sigma2, rho, rng):
    tau, theta, theta_bar = switching_times(
        sigma1,
        sigma2,
        rho,
        rng,
    )

    x1 = X1_0
    x2 = X2_0

    J = 0.0
    t = 0.0

    while t < T:
        u1, u2 = control(
            t,
            tau,
            theta,
            theta_bar,
        )

        F = production(x1, x2)

        consumption = (
            1 - u1 - u2
        ) * F

        J += (
            np.exp(-NU * t)
            * consumption
            * DT
        )

        x1, x2 = sde_step(
            x1,
            x2,
            u1,
            u2,
            sigma1,
            sigma2,
            rho,
            rng,
        )

        t += DT

    return tau, theta, theta_bar, J

def simulate_trajectory(
    sigma1,
    sigma2,
    rho,
    rng,
):
    tau, theta, theta_bar = switching_times(
        sigma1,
        sigma2,
        rho,
        rng,
    )

    t_arr = [0.0]
    x1_arr = [X1_0]
    x2_arr = [X2_0]
    J_arr = [0.0]

    x1 = X1_0
    x2 = X2_0
    J = 0.0
    t = 0.0

    while t < T:
        u1, u2 = control(
            t,
            tau,
            theta,
            theta_bar,
        )

        F = production(x1, x2)

        consumption = (
            1 - u1 - u2
        ) * F

        J += (
            np.exp(-NU * t)
            * consumption
            * DT
        )

        x1, x2 = sde_step(
            x1,
            x2,
            u1,
            u2,
            sigma1,
            sigma2,
            rho,
            rng,
        )

        t += DT

        t_arr.append(t)
        x1_arr.append(x1)
        x2_arr.append(x2)
        J_arr.append(J)

    return (
        np.array(t_arr),
        np.array(x1_arr),
        np.array(x2_arr),
        np.array(J_arr),
        tau,
        theta,
        theta_bar,
    )

# ============================================================
# Monte Carlo
# ============================================================

def simulate_batch(
    N,
    sigma1,
    sigma2,
    rho,
    seed=SEED,
):
    rng = np.random.RandomState(seed)

    taus = np.empty(N)
    thetas = np.empty(N)
    theta_bars = np.empty(N)
    Js = np.empty(N)

    for i in range(N):
        (
            taus[i],
            thetas[i],
            theta_bars[i],
            Js[i],
        ) = simulate_once(
            sigma1,
            sigma2,
            rho,
            rng,
        )

    return taus, thetas, theta_bars, Js


# ============================================================
# Детерминированные точки
# ============================================================

def det_switch_points():

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
        [X1_0, X2_0],
        t_eval=[TAU_DET],
        rtol=1e-8,
    )

    x1_tau, x2_tau = sol1.y[:, -1]

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
        [x1_tau, x2_tau],
        t_eval=[THETA_DET],
        rtol=1e-8,
    )

    x1_theta, x2_theta = sol2.y[:, -1]

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
        [x1_theta, x2_theta],
        t_eval=[THETA_BAR_DET],
        rtol=1e-8,
    )

    x1_theta_bar, x2_theta_bar = sol3.y[:, -1]

    return {
        "tau": (
            TAU_DET,
            x1_tau,
            x2_tau,
            z_sng,
        ),
        "theta": (
            THETA_DET,
            x1_theta,
            x2_theta,
            z_sng,
        ),
        "theta_bar": (
            THETA_BAR_DET,
            x1_theta_bar,
            x2_theta_bar,
            x2_theta_bar / x1_theta_bar,
        ),
    }

# ============================================================
# Сохранение таблиц
# ============================================================

def save_table(filename, header, rows):
    path = TABLES_DIR / filename

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

# ============================================================
# Таблица 6.1
# ============================================================

def print_table_6_1():
    pts = det_switch_points()

    print(
        "\nТаблица 6.1. "
        "Детерминированные моменты переключения."
    )

    print(
        "Момент     | Значение | "
        "z       | x1      | x2"
    )

    print(
        "-----------|----------|"
        "---------|---------|---------"
    )

    labels = {
        "tau": "tau",
        "theta": "theta",
        "theta_bar": "theta_bar",
    }

    rows = []

    for key in ["tau", "theta", "theta_bar"]:
        t, x1, x2, z = pts[key]

        print(
            f"{labels[key]:9s} | "
            f"{t:.6f} | "
            f"{z:.6f} | "
            f"{x1:.4f} | "
            f"{x2:.4f}"
        )

        rows.append([
            labels[key],
            f"{t:.6f}",
            f"{z:.6f}",
            f"{x1:.4f}",
            f"{x2:.4f}",
        ])

    save_table(
        "table_6_1_deterministic_switching_points.csv",
        [
            "moment",
            "value",
            "z",
            "x1",
            "x2",
        ],
        rows,
    )

# ============================================================
# Таблица 6.2
# ============================================================

def print_table_6_2(N=N_SIM):
    taus, thetas, theta_bars, Js = simulate_batch(
        N,
        SIGMA_1_BASE,
        SIGMA_2_BASE,
        RHO_BASE,
    )

    print(
        f"\nТаблица 6.2. "
        f"Базовый случай (N={N})."
    )

    print(
        "Величина | Детерм. значение | "
        "Среднее | Станд. откл."
    )

    data = [
        ("tau", taus, TAU_DET),
        ("theta", thetas, THETA_DET),
        ("theta_bar", theta_bars, THETA_BAR_DET),
        ("J", Js, J_DET),
    ]

    rows = []

    for name, values, det in data:
        mean = np.mean(values)
        std = np.std(values)

        print(
            f"{name:9s} | "
            f"{det:.6f} | "
            f"{mean:.4f} | "
            f"{std:.4f}"
        )

        rows.append([
            name,
            f"{det:.6f}",
            f"{mean:.4f}",
            f"{std:.4f}",
        ])

    save_table(
        "table_6_2_baseline_monte_carlo.csv",
        [
            "variable",
            "deterministic_value",
            "mean",
            "std",
        ],
        rows,
    )

# ============================================================
# Таблица 6.3
# ============================================================

def print_table_6_3(N=N_SIM):
    print(
        "\nТаблица 6.3. "
        "Влияние корреляции."
    )

    print(
        "rho  | E[tau] | E[theta] | "
        "E[theta_bar] | E[J]"
    )

    rows = []

    for rho in RHO_VALUES:
        taus, thetas, theta_bars, Js = simulate_batch(
            N,
            SIGMA_1_BASE,
            SIGMA_2_BASE,
            rho,
        )

        E_tau = np.mean(taus)
        E_theta = np.mean(thetas)
        E_theta_bar = np.mean(theta_bars)
        E_J = np.mean(Js)

        print(
            f"{rho:4.1f} | "
            f"{E_tau:.4f} | "
            f"{E_theta:.4f} | "
            f"{E_theta_bar:.4f} | "
            f"{E_J:.4f}"
        )

        rows.append([
            f"{rho:.1f}",
            f"{E_tau:.4f}",
            f"{E_theta:.4f}",
            f"{E_theta_bar:.4f}",
            f"{E_J:.4f}",
        ])

    save_table(
        "table_6_3_correlation.csv",
        [
            "rho",
            "E_tau",
            "E_theta",
            "E_theta_bar",
            "E_J",
        ],
        rows,
    )


# ============================================================
# Таблица 6.4
# ============================================================

def print_table_6_4(N=N_SIM):
    print(
        "\nТаблица 6.4. "
        "Влияние волатильности (rho=0)."
    )

    print(
        "sigma | E[tau] | E[theta] | "
        "E[theta_bar] | E[J] | Потери"
    )

    rows = []

    for sigma in SIGMA_VALUES:
        taus, thetas, theta_bars, Js = simulate_batch(
            N,
            sigma,
            sigma,
            0.0,
        )

        E_tau = np.mean(taus)
        E_theta = np.mean(thetas)
        E_theta_bar = np.mean(theta_bars)
        E_J = np.mean(Js)

        loss = (
            J_DET - E_J
        ) / J_DET * 100

        print(
            f"{sigma:.2f} | "
            f"{E_tau:.4f} | "
            f"{E_theta:.4f} | "
            f"{E_theta_bar:.4f} | "
            f"{E_J:.4f} | "
            f"{loss:.1f}%"
        )

        rows.append([
            f"{sigma:.2f}",
            f"{E_tau:.4f}",
            f"{E_theta:.4f}",
            f"{E_theta_bar:.4f}",
            f"{E_J:.4f}",
            f"{loss:.1f}",
        ])

    save_table(
        "table_6_4_volatility.csv",
        [
            "sigma",
            "E_tau",
            "E_theta",
            "E_theta_bar",
            "E_J",
            "loss_percent",
        ],
        rows,
    )


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":

    print(f"z_sng = {z_sng:.6f}")
    print(f"D_tau = {D_tau:.6f}")
    print(f"C_rho = {C_rho:.6f}")

    print_table_6_1()
    print_table_6_2()
    print_table_6_3()
    print_table_6_4()