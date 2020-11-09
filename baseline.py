import csv
import datetime
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
import statsmodels.api as sm
from utils import powerset, forecast_accuracy


class Data:
    def __init__(self, filename):
        self.filename = filename # 
        self.fields = []
        self.rows = [] 
        self.read_data(self.filename) # initialize data

    def read_data(self, filename):
        # reading csv file 
        with open(filename, 'r') as csvfile: 
            # creating a csv reader object 
            csvreader = csv.reader(csvfile) 
            
            # extracting field names through first row 
            self.fields = next(csvreader) 

            # extracting each data row one by one 
            for row in csvreader: 
                self.rows.append(row) 

            # get total number of rows 
            print("Total num of rows: ", len(self.rows))
    
    def get_field_names(self):
        print('Field names are:' + ', '.join(field for field in self.fields)) 

    def print_sample(self):
        print('First 5 rows are: ') 
        for row in self.rows[:5]: 
            print(row)


# Write to new csv, with only 2019 data
def datetime_experimentation():
    rows_2019 = []
    for row in data.rows:
        date_time_str = row[0] # eg. 2014-01-01 04:00:00
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        

    for row in rows_2019[:5]:
        print(row)


def create_year_data(filename, year):
    all_data = Data(filename) # reads and initializes data

    rows = []
    for row in all_data.rows:
        if year in row[0]:
            rows.append(row)

    new_filename = f"texas_{year}_dataset.csv"
    # writing to csv file  
    with open(new_filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
            
        # writing the fields  
        csvwriter.writerow(all_data.fields)  
            
        # writing the data rows  
        csvwriter.writerows(rows)

def split_data(filename):
    all_data = Data(filename) # reads and initializes data
    train_filename = "train_texas_2009_to_2017_dataset.csv"
    test_filename = "test_texas_2018_to_2019_dataset.csv"

    rows_train = []
    rows_test = []
    for row in all_data.rows:
        if "2018" in row[0] or "2019" in row[0]:
            rows_test.append(row)
        else:
            rows_train.append(row)

    with open(train_filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)              
        csvwriter.writerow(all_data.fields)  
        csvwriter.writerows(rows_train)

    with open(test_filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)              
        csvwriter.writerow(all_data.fields)  
        csvwriter.writerows(rows_test)

def linear_regression_simple(filename):
    df = pd.read_csv(filename)
    
    X = np.asarray(df["tempC"]).reshape((-1,1)) # needs to have shape (8760, 1)
    y = np.asarray(df["MWh"]) # needs to have shape (8760,)

    model = LinearRegression().fit(X, y)
    intercpt = model.intercept_
    coef = model.coef_
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    plt.title("Texas, Simple Linear Regression")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Consumption (MWh)")

    plt.text(0, 1400, f"Intercept: {round(float(intercpt),3)}, \nCoefficient: {round(float(coef),3)}")

    plt.scatter(X, y,color='g')
    plt.plot(X, model.predict(X),color='b')

    plt.show()



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
    all_features = ["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex"]
    l = list(range(1,5))
    indices_set = [x for x in powerset(l)]
    print("indices_set: ", indices_set)
    for s in indices_set:
        if not s: continue
        s.append(0)
        features = [all_features[i] for i in s]

        print("")
        print(f"************ RESULTS FOR FEATURES: {features} ************")
        multivariable_regression(train_filename, test_filename, features)

if __name__ == "__main__":
    filename = "texas_2009_to_2019_dataset01.csv"
    train_filename = "train_texas_2009_to_2017_dataset.csv"     # 80% of dataset
    test_filename = "test_texas_2018_to_2019_dataset.csv"       # 20% of dataset

    # split_data(filename)
    # create_year_data(filename, "2019")

    # data = Data("san_diego_2019_dataset.csv")
    # linear_regression_simple(filename)

    features = ["tempC", "HeatIndexC", "WindChillC", "humidity", "uvIndex"]
    multivariable_regression(train_filename, test_filename, features)

    
"""
Notes for myself, TODO:
- Create a new feature that takes into account the month (1,2,...,12)
- Figure out why I have NANs in my 2 datasets & fix them
"""