import csv
import datetime

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


def create_2019_data():
    filename = "san_diego_2019_dataset.csv"

    rows_2019 = []
    for row in all_data.rows:
        if "2019" in row[0]:
            rows_2019.append(row)

    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
            
        # writing the fields  
        csvwriter.writerow(all_data.fields)  
            
        # writing the data rows  
        csvwriter.writerows(rows_2019)


all_data = Data("san_diego_2014_to_2019_dataset.csv") # reads and initializes data


