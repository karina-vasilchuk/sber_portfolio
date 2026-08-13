# Stochastic Optimal Control: Numerical Experiment

This repository contains the numerical experiment from my bachelor's thesis on a stochastic optimal control problem with two state variables and regime switching.

The project focuses on the numerical analysis of switching times, stochastic trajectories and the value of the objective functional under multiplicative noise.

## Model

The system contains two state variables, $x_1(t)$ and $x_2(t)$, with Cobb-Douglas production function

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
\tau^{(0)} = 0.394764,\qquad
\theta^{(0)} = 1.105905,\qquad
\bar{\theta}^{(0)} = 2.001445.
$$

The stochastic dynamics are simulated using the Euler-Maruyama method.

## Numerical experiment

The baseline stochastic specification is

```text
sigma_1 = 0.15
sigma_2 = 0.15
rho = 0.0
N = 10 000 Monte Carlo simulations
dt = 0.001
```

The experiment evaluates:

- stochastic corrections to switching times;
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

Contains the model parameters, initial conditions, deterministic switching times and baseline stochastic parameters used in the numerical experiment.

### `experiments/numerical_experiment.py`

Contains:

- computation of the singular ratio \(z_{\mathrm{sng}}\);
- calculation of second-order corrections to switching times;
- Euler-Maruyama simulation;
- Monte Carlo experiment;
- sensitivity analysis with respect to correlation and volatility;
- generation of numerical tables.

### `experiments/figures.py`

Generates the main graphical results of the experiment:

- deterministic phase trajectory;
- stochastic trajectories of \(z(t)\);
- Monte Carlo confidence intervals for \(x_1(t)\) and \(x_2(t)\);
- influence of correlation \(\rho\);
- influence of volatility \(\sigma\).

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
E[theta_bar] = 2.0264
E[J]         = 2.1878
```

The experiments show that:

- the expected first switching time decreases as volatility increases;
- the expected calibration switching time increases with volatility;
- positive noise correlation reduces the stochastic shift of the switching times;
- the expected objective functional decreases as volatility increases.

The generated numerical results are stored in:

```text
results/tables/
```

and figures are stored in:

```text
results/figures/
```

## Running the project

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Run the numerical experiment from the project root

```bash
python -m experiments.numerical_experiment
```

### Generate the figures

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

The numerical experiment studies the stochastic behavior of the model under different volatility and correlation specifications.
