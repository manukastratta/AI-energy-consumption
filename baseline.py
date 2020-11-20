import csv
import datetime
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
import statsmodels.api as sm
from utils import powerset, forecast_accuracy
from dataUtils import Data
import copy
from multivariableRegression import split_dataset, get_manual_accuracy, get_error_analysis



def plot_linear_relationship(df, x_name):
    plt.scatter(df[x_name], df["MWh"], color='green')
    plt.title(f'{x_name} vs Energy Consumption (MWh)', fontsize=14)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel('Energy Consumption', fontsize=14)
    plt.grid(True)
    plt.show()


def plot_baseline(X_test, Y_test, Y_pred, intercept, coef):
    plt.title("Temperature vs Energy, Baseline Linear Regression")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Energy Consumption (MWh)")

    plt.text(-4, 1450, f"Intercept: {round(float(intercept),3)}, \nCoefficient: {round(float(coef),3)}")

    plt.plot(X_test, Y_pred, color="b") # the line!
    plt.scatter(X_test, Y_test, color='g') # the actual points!

    plt.show()

def baseline_simple_LR(filename):
    df_train, df_test = split_dataset(filename)

    X_train = df_train[["tempC"]]
    Y_train = df_train["MWh"]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)

    print('Intercept: ', regr.intercept_)
    print('Coefficients: ', regr.coef_)

    # Make predictions using the testing set
    X_test = df_test[["tempC"]]
    Y_test = df_test["MWh"]
    Y_pred = regr.predict(X_test)

    # Evaluation / Error analysis
    get_error_analysis(Y_test, Y_pred)
    plot_baseline(X_test, Y_test, Y_pred, regr.intercept_, regr.coef_)


if __name__ == "__main__":
    # filename = "texas_2009_to_2019_dataset01.csv"
    filename = "texas_2009_to_2019_dataset_with_temporal_features.csv"

    baseline_simple_LR(filename)
