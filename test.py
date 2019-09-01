from abc import ABCMeta, abstractmethod

class BaseLinearRegression:

    def fit(self):
        # self.Y, self.X = Y, X
        self.coef = 7

        return self

class BayesianModel(BaseLinearRegression):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_prior(self):
        pass

    @abstractmethod
    def get_posterior_distribution(self):
        pass

    @abstractmethod
    def gibbs_sampling(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

class BayesianLinearRegression(BayesianModel):

    def estimate(self):
        self.coef = 3
        ols = BaseLinearRegression()
        ols.fit()
        print(self.coef)
        print(ols.coef)
        return self

if __name__ == "__main__":
    b = BayesianLinearRegression()
    b.estimate()