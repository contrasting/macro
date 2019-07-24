import matplotlib.pyplot as plt
import numpy as np

def plot_var_forc(prior, forc, err_upper, err_lower,
                  index=None, names=None, plot_stderr=True,
                  legend_options=None):

    n, k = prior.shape
    rows, cols = k, 1 # for the plot, not the data

    fig = plt.figure(figsize=(10, 30))

    prange = np.arange(n)
    rng_f = np.arange(n - 1, n + len(forc))
    rng_err = np.arange(n, n + len(forc))

    for j in range(k):
        ax = plt.subplot(rows, cols, j+1)

        p1 = ax.plot(prange, prior[:, j], 'k', label='Observed')
        p2 = ax.plot(rng_f, np.r_[prior[-1:, j], forc[:, j]], 'k--',
                     label='Forecast')

        if plot_stderr:
            p3 = ax.plot(rng_err, err_upper[:, j], 'k-.',
                         label='Forc 2 STD err')
            ax.plot(rng_err, err_lower[:, j], 'k-.')

        if names is not None:
            ax.set_title(names[j])

#         if legend_options is None:
#             legend_options = {"loc": "upper right"}
#         ax.legend(**legend_options)
    return fig

