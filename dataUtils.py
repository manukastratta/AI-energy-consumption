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

def simple_linear_regression(filename):
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