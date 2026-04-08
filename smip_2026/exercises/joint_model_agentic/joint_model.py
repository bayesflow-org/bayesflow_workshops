"""
Implements model M_1c from [1].

[1] Ghaderi-Kangavari, A., Rad, J. A., & Nunez, M. D. (2023). 
A general integrative neurocognitive modeling framework to jointly 
describe EEG and decision-making on single trials. 
Computational Brain & Behavior, 6(3), 317-376.
"""

import numpy as np
from numba import njit


def draw_prior():
    """Draws samples from the prior distributions of m1c, as described in Ghaderi-Kangavari et al. (2023)"""

    prior_draws = np.random.uniform(
        low=(-3.0, 0.5, 0.1, 0.05, 0.06, 0.0, 0.0),
        high=(3.0, 2.0, 0.9, 0.4, 0.6, 0.1, 0.1)
    )

    prior_draws = {
        "drift": prior_draws[0],
        "boundary": prior_draws[1],
        "beta": prior_draws[2],
        "mu_tau_e": prior_draws[3],
        "tau_m": prior_draws[4],
        "sigma": prior_draws[5],
        "varsigma": prior_draws[6]
    }

    return prior_draws


@njit
def diffusion_trial(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, s=1.0, dt=5e-3):
    """Simulates a trial from the joint diffusion model m1b."""

    c = np.sqrt(dt) * s
    n_steps = 0.0
    evidence = boundary * beta

    while evidence > 0 and evidence < boundary:
        evidence += drift * dt + c * np.random.normal()
        n_steps += 1.0

    z = 0
    while True:
        # visual encoding
        tau_encoding = mu_tau_e + np.random.uniform(-0.5 * np.sqrt(12) * varsigma, 0.5 * np.sqrt(12) * varsigma)
        z = np.random.normal(tau_encoding, sigma)
        if z > 0 and z < 0.5:
            break

    rt = n_steps * dt + tau_encoding + tau_m

    if evidence >= boundary:
        return (rt, 1.0, z)
    return (rt, 0.0, z)


def simulate_trials(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, num_trials=120):
    """Simulates a diffusion process for trials."""

    data = np.empty((num_trials, 3))
    for i in range(num_trials):
        data[i] = diffusion_trial(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma)
    return dict(data=data)
