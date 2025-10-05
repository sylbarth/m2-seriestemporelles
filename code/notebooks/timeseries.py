
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor


def show_performance(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("Mean Squared Error: {:0.1f}".format(mse))
    print("R-squared:  {:0.1f}".format(r2))
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(y_true.values, color='tab:blue')
    ax.plot(y_pred, color='tab:red')

def plot(data):
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.autofmt_xdate()
    ax.plot(data)
    ax.legend()
    ax.grid()

def plot_decision_tree(data, tree_to_plot):
    plt.figure(figsize=(16, 16))
    plot_tree(tree_to_plot, feature_names=data.columns.tolist(), filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree from Random Forest")
    plt.show()


def timeseries_train_test_split(X, y, train_size):
    X_train = X.iloc[:train_size]
    y_train = y[:train_size]
    X_test  = X.iloc[train_size:]
    y_test  = y[train_size:]
    return X_train, X_test, y_train, y_test

def timeseries_model_eval(model, X, y):
    y_pred = model.predict(X)
    show_performance(y, y_pred)

def to_bins(metrics, num=10):
    _c = metrics.copy()
    _q = metrics.quantile(q=np.linspace(.0, 1, num=num+1), interpolation='lower')
    for col in _c.columns:
        bins = list(dict.fromkeys(_q[col]))
        _c[col] = 1 + pd.cut(metrics[col], bins=bins, labels=False, include_lowest=True)
    return _c




class TimeSeriesModel:

    def __init__(self, data, target, train_size=0.7):

        self.data = data.copy()
        self.target = target

        # if "date" in data.columns:
        #     try:
        #         self.data.set_index("date", inplace=True)
        #     except Exception as e:
        #         print(e)

        self.X = self.data.loc[:,~data.columns.isin(["date", target])]
        self.y = self.data[target]

        self.train_size = train_size
        if self.train_size < 1:
            self.train_size = int(train_size * len(self.data))
        
        self.X_train = self.X.iloc[:self.train_size]
        self.y_train = self.y[:self.train_size]
        self.X_test  = self.X.iloc[self.train_size:]
        self.y_test  = self.y[self.train_size:]

    def plot(self):
        fig, ax = plt.subplots(figsize=(16, 5))
        fig.autofmt_xdate()
        ax.plot(self.data.loc[:,self.target])
        # ax.plot(self.y_true.values, color='tab:blue')
        ax.legend()
        ax.grid()


class TimeSeriesRandomForest(TimeSeriesModel):

    def fit(self, random_state=42):
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state, oob_score=True)
        self.fit = self.model.fit(self.X_train, self.y_train)
        return self.fit