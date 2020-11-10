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

def plot_linear_relationship(df, x_name):
    plt.scatter(df[x_name], df["MWh"], color='green')
    plt.title(f'{x_name} vs Energy Consumption (MWh)', fontsize=14)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel('Energy Consumption', fontsize=14)
    plt.grid(True)
    plt.show()

def get_statsmodels_table(X, Y):
    """
                                        OLS Regression Results                            
        ==============================================================================
        Dep. Variable:                    MWh   R-squared:                       0.387
        Model:                            OLS   Adj. R-squared:                  0.387
        Method:                 Least Squares   F-statistic:                     1106.
        Date:                Sun, 08 Nov 2020   Prob (F-statistic):               0.00
        Time:                        14:48:53   Log-Likelihood:                -55653.
        No. Observations:                8760   AIC:                         1.113e+05
        Df Residuals:                    8754   BIC:                         1.114e+05
        Df Model:                           5                                         
        Covariance Type:            nonrobust                                         
        ==============================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        const        847.4236      7.488    113.167      0.000     832.745     862.102
        tempC         27.6532      1.565     17.674      0.000      24.586      30.720
        HeatIndexC     6.0404      1.424      4.242      0.000       3.249       8.832
        WindChillC   -24.9396      1.461    -17.068      0.000     -27.804     -22.075
        humidity      -2.8734      0.074    -38.859      0.000      -3.018      -2.728
        uvIndex       -1.2631      1.678     -0.753      0.452      -4.552       2.026
        ==============================================================================
        Omnibus:                      374.057   Durbin-Watson:                   0.061
        Prob(Omnibus):                  0.000   Jarque-Bera (JB):              419.643
        Skew:                           0.531   Prob(JB):                     7.51e-92
        Kurtosis:                       2.847   Cond. No.                         388.
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

def multivariable_regression(train_filename, test_filename, features_list):
    """
    COLUMN FIELDS: cols: Date,MWh,uvIndex,HeatIndexC,WindChillC,humidity,tempC

    If JUST the temp/energy (simple LR):
    Intercept: 625.7157440367006
    Coefficients: 
        "tempC"
        [10.6531254]
    mean_sq_err:  25097.500364439336
    Manual accuracy:  0.38396118721461187

    Predictions, when split train/test:
        Intercept:  829.2318130846693
        Coefficients: 
        ["tempC",      "HeatIndexC",   "WindChillC",    "humidity",     "uvIndex"]]
        [ 26.77415029  12.09465205      -29.3946928     -2.81113462     -2.37345755]
        
        mean_sq_err:  21629.355399131226
        Manual accuracy:  0.43767123287671234 --> 43% accuracy within 10% of the true value
    
    
    Without uvIndex, got slightly better: Manual accuracy:  0.43818493150684934, see results.txt
    """
    df_train = pd.read_csv(train_filename) 
    df_test = pd.read_csv(test_filename)

    df_train.dropna(inplace=True) # TODO: shouldn't be getting NANs here
    df_test.dropna(inplace=True)

    # plot_linear_relationship(df, "tempC")
    # X_train = df_train[["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex"]]
    X_train = df_train[features_list]
    Y_train = df_train["MWh"]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)

    print('Intercept: ', regr.intercept_)
    print('Coefficients: ', regr.coef_)

    # get_statsmodels_table(X,Y)

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
    multivariable_regression(train_filename, test_filename, ["tempC"])

if __name__ == "__main__":
    filename = "texas_2009_to_2019_dataset01.csv"
    train_filename = "train_texas_2009_to_2017_dataset.csv"     # 80% of dataset
    test_filename = "test_texas_2018_to_2019_dataset.csv"       # 20% of dataset

    # split_data(filename)
    # create_year_data(filename, "2019")

    # data = Data("san_diego_2019_dataset.csv")
    # simple_linear_regression(filename)

    features = ["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex"]
    # baseline_simple_LR(train_filename, test_filename)
    multivariable_regression(train_filename, test_filename, features)

    

    
"""
Notes for myself, TODO:
- Create a new feature that takes into account the month (1,2,...,12)
- Figure out why I have NANs in my 2 datasets & fix them
"""