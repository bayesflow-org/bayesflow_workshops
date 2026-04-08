# Copilot instructions

## Amortized inference workflow
- For simulation-based inference, amortized Bayesian inference, neural posterior estimation, neural likelihood estimation, neural ratio estimation, posterior amortization, and BayesFlow workflows, use the `amortized-workflow` skill from the `skills` folder.
- Use the amortized-workflow skill for model setup, simulator assumptions, training, validation, calibration, recovery, posterior contraction, and reporting.

## Project defaults
- Do not automatically create a new conda / virtual environment.
- If a conda environment named `bf` exists, use it. If not, create a new one.
- Prefer BayesFlow/PyTorch/JAX tools already present in that environment.