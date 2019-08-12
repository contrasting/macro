import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler, Subset

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


def which_fluc(data: pd.DataFrame, value: float):
    for series in data:
        col = data[series]
        for data in col:
            if data < -value or data > value:
                print(col.describe())
                break


def get_lags(X, lags):
    X_lagged = pd.DataFrame()

    # wow, there was a bug here...
    for i in range(1, lags + 1):
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
    
def is_outlier(points, thresh=3.5):
    # https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting

    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def evaluate_model(net: nn.Module, val_loader: DataLoader, criterion: nn.MSELoss):
    net.eval()
    val_loss = 0
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            y, X = data
            y, X = y.float(), X.float()
            y_pred = net(X)
            loss = criterion(y, y_pred)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def train_one_epoch(net: nn.Module, train_loader: DataLoader, criterion: nn.MSELoss, optimizer: optim):
    net.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        
        y, X = data
        y, X = y.float(), X.float()
        
        optimizer.zero_grad()
        
        y_pred = net(X)
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_train_loss = epoch_loss / len(train_loader)
    return avg_train_loss
    

def get_average(losses: list):
    total = 0
    for i in losses:
        total += i
    return total/len(losses)



# early stopping implementation
# inspired by https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d

def train_window(net: nn.Module, criterion: nn.MSELoss, optimizer: optim, window):
    running_val_loss = []
    prev_val_loss = 1000000  # very big number
    cnt = 0
    patience = 10
    ma = 5
    
    for epoch in range(5000):
            
        train_loss = train_one_epoch(net, window.trainloader, criterion, optimizer)
        val_loss = evaluate_model(net, window.validationloader, criterion)
        running_val_loss.append(val_loss)
        
        if epoch % 5 == 0:
            print("[epoch: %d] train loss: %.3f, val loss: %.3f"
                  % (epoch + 1, train_loss, val_loss))
        
        avg_val_loss = get_average(running_val_loss)
    
        if avg_val_loss > prev_val_loss:
            if cnt > patience:
                break
            else:
                cnt += 1
        else:
            cnt = 0  # reset
            
        prev_val_loss = avg_val_loss
        
        # restrict to moving average
        while len(running_val_loss) > ma:
            running_val_loss.pop(0)

        # manual training stopper
        if epoch % 500 == 0 and epoch != 0:
            cnt_train = input("Continue training? True or False")
            if cnt_train != True:
                break
                
    print("Finished window")


class Window:

    def __init__(self, trainloader, validationloader):
        self.trainloader = trainloader
        self.validationloader = validationloader
    