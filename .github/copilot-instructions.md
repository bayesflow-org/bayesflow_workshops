# Copilot instructions

## Amortized inference workflow
- For simulation-based inference, amortized Bayesian inference, neural posterior estimation, neural likelihood estimation, neural ratio estimation, posterior amortization, and BayesFlow workflows, use the `amortized-workflow` skill from the `skills` folder.
- **BEFORE writing any BayesFlow code**, read ALL of the following files in full — do not skip any:
  1. `skills/amortized-workflow/SKILL.md`
  2. `skills/amortized-workflow/references/adapter.md`
  3. `skills/amortized-workflow/references/conditioning.md`
  4. `skills/amortized-workflow/references/model-sizes.md`
  5. `skills/amortized-workflow/references/custom-summary.md`
  6. `skills/amortized-workflow/references/image-generation.md`
- **Do NOT generate any BayesFlow code if you have not read all of the above.** Partial reads produce non-compliant code that silently breaks inference.

## Project defaults
- Do not automatically create a new conda / virtual environment.
- If a conda environment named `bf` exists, use it. If not, create a new one.
- Prefer BayesFlow/PyTorch/JAX tools already present in that environment.
