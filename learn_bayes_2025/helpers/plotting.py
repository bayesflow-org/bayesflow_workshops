import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_lv_trajectores(
    samples: dict[str, np.ndarray], 
    variable_keys: list[str], 
    variable_names: list[str], 
    fill_colors: list[str] = ["blue", "darkred"],
    num_to_plot: int = 20,
    confidence: float = 0.95, 
    alpha: float = 0.2, 
    observations: dict[str, np.ndarray] = None,
    ax=None
):
    t_span = samples["t"][0]
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12,3))
    
    for i, key in enumerate(variable_keys):

        if observations is not None:     
            ax.scatter(observations["observed_t"], observations["observed_"+key], color=fill_colors[i], marker="x", label="Observed " + variable_names[i].lower())

        central, L, U = trajectory_aggregation(samples[key], confidence=confidence)
        ax.plot(t_span, central, color=fill_colors[i], label="Median " + variable_names[i].lower())
        ax.fill_between(t_span, L, U, color=fill_colors[i], alpha=alpha, label=rf"{int((confidence) * 100)}$\%$ Confidence Bands")

        # plot 20 trajectory samples
        for j in range(num_to_plot):
            if j == 0:
                label = f"{variable_names[i]} trajectories"
            else:
                label = None
            ax.plot(t_span, samples[key][j], color=fill_colors[i], alpha=alpha, label=label)
        
    ax.set_xlabel("t")
    ax.set_ylabel("population")
    ax.legend()
    sns.despine(ax=ax)


def trajectory_aggregation(traj, confidence=0.95):
    alpha = 1 - confidence
    quantiles = np.quantile(traj, [alpha/2, 0.5, 1-alpha/2], axis=0).T
    median = quantiles[:,1]
    lower = quantiles[:,0]
    upper = quantiles[:,2]
    return median, lower, upper
