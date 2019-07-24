import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def ridge_regress(y, X):
    # plot them first
    plt.plot(y)
    plt.show()
    plt.plot(X)
    plt.show()

    from sklearn import linear_model

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit(X, y)

    print("Alpha:")
    print(reg.alpha_)
    print("R squared:")
    print(reg.score(X, y))

    return reg

def kernel_ridge(y, X, k):
    # plot them first
    plt.plot(y)
    plt.show()
    plt.plot(X)
    plt.show()

    from sklearn.kernel_ridge import KernelRidge

    model = KernelRidge(kernel=k)
    model.fit(X, y)

    print("R squared:")
    print(model.score(X, y))

    return model


# apparently unnecessary
def rename_col(X, suffix):
    assert isinstance(X, str), "suffix has to be of type str"

    col_name = {}

    for series in X:
        col_name[series] = col_name + suffix

    return X.rename(columns=col_name)

# unused for now
class RidgeAR:

    def __init__(self, reg, y, X_lagged):
        self.reg = reg
        self.y = y
        self.X_lagged = X_lagged

    def forecast(self):
        fitted_val = pd.Series(self.reg.predict(self.X_lagged),
                      index = self.y.index)

        plt.plot(self.y)
        plt.plot(fitted_val)
        plt.show()
        
        return fitted_val


def get_lags(X, lags):
    X_lagged = pd.DataFrame()

    for i in range(lags):
        temp = X.shift(i)
        X_lagged = pd.concat([X_lagged, temp], axis=1)

    return X_lagged


def ridge_ar(y, X, lags):
    assert isinstance(X, pd.DataFrame), "X has to be of type pd.DataFrame"

    X_lagged = get_lags(X, lags)

    # get rid of nans

    y = y[lags:]
    X_lagged = X_lagged[lags:]

    print("Feature variable: ")
    print(y.name)
    print("Dimensions of lagged X: ")
    print(X_lagged.shape)
    print("Fitting regression...")

    return ridge_regress(y, X_lagged)


def kernel_ridge_ar(y, X, lags, kernel):
    assert isinstance(X, pd.DataFrame), "X has to be of type pd.DataFrame"

    X_lagged = get_lags(X, lags)

    # get rid of nans

    y = y[lags:]
    X_lagged = X_lagged[lags:]

    print("Feature variable: ")
    print(y.name)
    print("Dimensions of lagged X: ")
    print(X_lagged.shape)
    print("Fitting regression...")

    return kernel_ridge(y, X_lagged, kernel)
    