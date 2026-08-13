# Stochastic Optimal Control: Numerical Experiment

This repository contains the numerical experiment from my bachelor's thesis on a stochastic optimal control problem with two state variables and regime switching.

The project focuses on switching times, stochastic trajectories, the objective functional, and sensitivity to volatility and noise correlation.

## Model

The system contains two state variables, x1(t) and x2(t), with a Cobb-Douglas production function

$$
F(x_1, x_2) = x_1^{\varepsilon_1} x_2^{\varepsilon_2}.
$$

The optimal control consists of four regimes:

1. investment in the second sector;
2. singular regime;
3. investment in the first sector;
4. terminal consumption regime.

The deterministic switching times used in the numerical experiment are

$$
\tau^{(0)} = 0.394764,
$$

$$
\theta^{(0)} = 1.105905,
$$

$$
\bar{\theta}^{(0)} = 2.001445.
$$

The stochastic dynamics are simulated using the Euler-Maruyama method.

Correlated shocks are generated from two independent standard normal variables. For each time step,

$$
\Delta W_1 = \sqrt{\Delta t}\,\xi_1,
$$

$$
\Delta W_2 =
\sqrt{\Delta t}
\left(
\rho \xi_1 +
\sqrt{1-\rho^2}\,\xi_2
\right),
$$

where xi1 and xi2 are independent standard normal variables.

## Numerical experiment

The baseline stochastic specification is:

```text
sigma_1 = 0.15
sigma_2 = 0.15
rho = 0.0
N = 10 000 Monte Carlo simulations
dt = 0.001
```

The experiment evaluates:

- second-order corrections to switching times;
- Monte Carlo distributions of switching times;
- stochastic trajectories of the state variables;
- the expected value of the objective functional;
- sensitivity to noise correlation;
- sensitivity to volatility.

## Project structure

```text
sber_portfolio/
├── src/
│   ├── __init__.py
│   └── parameters.py
│
├── experiments/
│   ├── __init__.py
│   ├── numerical_experiment.py
│   └── figures.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── README.md
├── requirements.txt
└── .gitignore
```

### `src/parameters.py`

Contains:

- model parameters;
- initial conditions;
- deterministic switching times;
- baseline stochastic parameters;
- numerical parameters.

### `experiments/numerical_experiment.py`

Contains:

- computation of the singular ratio `z_sng`;
- calculation of second-order corrections to switching times;
- Euler-Maruyama simulation;
- Monte Carlo experiment;
- sensitivity analysis with respect to `rho` and `sigma`;
- generation of numerical tables.

### `experiments/figures.py`

Generates:

- deterministic phase trajectory;
- stochastic trajectories of `z(t)`;
- Monte Carlo intervals for `x1(t)` and `x2(t)`;
- influence of correlation `rho`;
- influence of volatility `sigma`.

## Results

For the baseline case

```text
sigma_1 = sigma_2 = 0.15
rho = 0
N = 10 000
```

the simulation produces approximately

```text
E[tau]       = 0.3924
E[theta]     = 1.1059
E[theta_bar] = 2.0261
E[J]         = 2.1890
```

The experiments show that:

- the expected first switching time decreases as volatility increases;
- the expected calibration switching time increases as volatility increases;
- positive noise correlation reduces the stochastic shift of the switching times;
- the expected objective functional decreases as volatility increases.

The numerical results are stored in:

```text
results/tables/
```

The generated figures are stored in:

```text
results/figures/
```

## Running the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the numerical experiment from the project root:

```bash
python -m experiments.numerical_experiment
```

Generate the figures:

```bash
python -m experiments.figures
```

## Dependencies

- Python 3
- NumPy
- SciPy
- Matplotlib

## Notes

The project is based on a theoretical model and does not use external empirical data.

The numerical experiment implements the approximation scheme described in Chapter 6 of the thesis and studies the stochastic behavior of the model under different volatility and correlation specifications.
