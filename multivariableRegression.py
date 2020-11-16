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

index_2018_start = 78887

def plot_linear_relationship(df, x_name):
    plt.scatter(df[x_name], df["MWh"], color='green')
    plt.title(f'{x_name} vs Energy Consumption (MWh)', fontsize=14)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel('Energy Consumption', fontsize=14)
    plt.grid(True)
    plt.show()

def plot_baseline(intercept, coef, X_test, Y_pred):
    plt.title("Temperature vs Energy, Baseline Linear Regression")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Energy Consumption (MWh)")

    plt.text(0, 1400, f"Intercept: {round(float(intercept),3)}, \nCoefficient: {round(float(coef),3)}")

    plt.scatter(X_test, Y_pred, color='g')
    # plt.plot(X, model.predict(X),color='b')

    plt.show()

def get_statsmodels_table(X, Y):
    """
        OLS Regression Results                            
        ==============================================================================
        Dep. Variable:                    MWh   R-squared:                       0.558
        Model:                            OLS   Adj. R-squared:                  0.558
        Method:                 Least Squares   F-statistic:                     7149.
        Date:                Sun, 15 Nov 2020   Prob (F-statistic):               0.00
        Time:                        16:53:03   Log-Likelihood:            -6.0017e+05
        No. Observations:               96406   AIC:                         1.200e+06
        Df Residuals:                   96388   BIC:                         1.201e+06
        Df Model:                          17                                         
        Covariance Type:            nonrobust                                         
        ==============================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        const        530.3403      2.004    264.682      0.000     526.413     534.268
        tempC         21.5788      0.425     50.821      0.000      20.747      22.411
        HeatIndexC    -1.9317      0.397     -4.864      0.000      -2.710      -1.153
        WindChillC   -11.4889      0.410    -28.014      0.000     -12.293     -10.685
        humidity      -1.4092      0.022    -63.235      0.000      -1.453      -1.366
        uvIndex      -12.8722      0.628    -20.487      0.000     -14.104     -11.641
        jan           90.1910      1.566     57.586      0.000      87.121      93.261
        feb           33.8232      1.449     23.348      0.000      30.984      36.663
        mar          -46.6812      1.382    -33.788      0.000     -49.389     -43.973
        apr          -89.7150      1.337    -67.089      0.000     -92.336     -87.094
        may          -27.7993      1.440    -19.302      0.000     -30.622     -24.977
        jun          108.4988      1.754     61.854      0.000     105.061     111.937
        jul          165.9890      1.954     84.935      0.000     162.159     169.819
        aug          172.0845      1.979     86.939      0.000     168.205     175.964
        sep           90.3389      1.492     60.533      0.000      87.414      93.264
        oct          -41.8207      1.343    -31.139      0.000     -44.453     -39.188
        nov          -16.8303      1.444    -11.652      0.000     -19.661     -13.999
        dec           92.2617      1.850     49.865      0.000      88.635      95.888
        peakH        300.7522      1.079    278.652      0.000     298.637     302.868
        notPeakH     229.5881      1.164    197.171      0.000     227.306     231.870
        ==============================================================================
        Omnibus:                     5483.009   Durbin-Watson:                   0.098
        Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6684.159
        Skew:                           0.582   Prob(JB):                         0.00
        Kurtosis:                       3.558   Cond. No.                     8.73e+17
        ==============================================================================
    """
    X = sm.add_constant(X) # adding a constant
    
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X) 
    
    print_model = model.summary()
    print(print_model)

def get_manual_accuracy(Y_test, Y_pred, percent=0.1):
    correct = 0
    for i in range(len(Y_test)):
        true_val = Y_test[i]
        pred = Y_pred[i]
        margin = percent * true_val
        if pred-margin <= true_val <= pred+margin:
            correct+=1
    accuracy = correct / len(Y_test)
    return accuracy

def multivariable_regression(filename, features_list):
    """
    COLUMN FIELDS: cols: Date,MWh,uvIndex,HeatIndexC,WindChillC,humidity,tempC, 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', "peakH", "notPeakH"

    If JUST the temp/energy (simple LR):
    Intercept: 625.7157440367006
    Coefficients: 
        "tempC"
        [10.6531254]
    mean_sq_err:  25097.500364439336
    Manual accuracy:  0.38396118721461187
    """
    df_total = pd.read_csv(filename)
    df_total.dropna(inplace=True) # only 1 row is dropped

    df_train = df_total.iloc[:index_2018_start, :]
    df_test = df_total.iloc[index_2018_start:, :]
    df_test = df_test.reset_index() # start rows counting at 0
    
    # plot_linear_relationship(df, "tempC")
    # X_train = df_train[["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex"]]
    X_train = df_train[features_list]
    Y_train = df_train["MWh"]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)

    print('Intercept: ', regr.intercept_)
    print('Coefficients: ', regr.coef_)

    # get_statsmodels_table(df_total[features_list], df_total["MWh"])

    # Make predictions using the testing set
    # X_test = df_test[["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex"]]
    X_test = df_test[features_list]
    Y_test = df_test["MWh"]

    Y_pred = regr.predict(X_test)
    mean_sq_err = metrics.mean_squared_error(Y_test, Y_pred)
    print("mean_sq_err: ", mean_sq_err)

    manual_accuracy = get_manual_accuracy(Y_test, Y_pred, percent=0.1)
    print("Manual accuracy: ", manual_accuracy)

    res = forecast_accuracy(Y_pred, Y_test)
    print(res)


def experiment_w_features(filename, train_filename, test_filename):
    """
    Function that runs Linear Regression with all combinations of features,
    eg. ["tempC", "HeatIndexC"] or ["WindChillC", "tempC", "uvIndex"]. 
    Used to determine which combination of variables achieve best results.
    """
    all_features = ["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex"]
    l = list(range(1,5))
    indices_set = [x for x in powerset(l)]
    for s in indices_set:
        if not s: continue
        s.append(0)
        features = [all_features[i] for i in s]
        print(f"\n************ RESULTS FOR FEATURES: {features} ************")
        multivariable_regression(train_filename, test_filename, features)


def baseline_simple_LR(train_filename, test_filename):
    df_train = pd.read_csv(train_filename) 
    df_test = pd.read_csv(test_filename)

    df_train.dropna(inplace=True) # TODO: shouldn't be getting NANs here
    df_test.dropna(inplace=True)

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

    # Evaluation
    mean_sq_err = metrics.mean_squared_error(Y_test, Y_pred)
    print("mean_sq_err: ", mean_sq_err)
    manual_accuracy = get_manual_accuracy(Y_test, Y_pred, percent=0.1)
    print("Manual accuracy: ", manual_accuracy)
    r2 = metrics.r2_score(Y_test, Y_pred)
    print("r2 score: ", r2)

    res = forecast_accuracy(Y_pred, Y_test)
    print(res)

    # Plot results
    plot_baseline(regr.intercept_, regr.coef_, X_test, Y_pred)


if __name__ == "__main__":
    # filename = "texas_2009_to_2019_dataset01.csv"
    filename = "texas_2009_to_2019_dataset_with_temporal_features.csv"
    # train_filename = "train_texas_2009_to_2017_dataset.csv"     # 80% of dataset
    # test_filename = "test_texas_2018_to_2019_dataset.csv"       # 20% of dataset

    # split_data(filename)
    # create_year_data(filename, "2019")

    # data = Data("san_diego_2019_dataset.csv")
    # simple_linear_regression(filename)

    features = ["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex", 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', "peakH", "notPeakH"]
    # baseline_simple_LR(train_filename, test_filename)
    multivariable_regression(filename, features)
    # add_peak_hours_to_dataset()

    

    
"""
Notes for myself, TODO:
- Create a new feature that takes into account the month (1,2,...,12)
- Figure out why I have NANs in my 2 datasets & fix them
"""