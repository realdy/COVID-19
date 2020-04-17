import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.patches as mpatches
import matplotlib

def initialize():
    path = './csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    row_drop = ['Province/State', 'Country/Region', 'Lat', 'Long']
    file = pd.read_csv(path)
    country_names = (file['Country/Region'])
    trans_file = file.transpose()
    trans_file.columns = country_names
    trans_file = trans_file.drop(row_drop)
    trans_file = trans_file.groupby(lambda x:x, axis=1).sum()
    sum_column = trans_file.sum(axis=1)
    trans_file['Total_Count'] = sum_column
    day_count = []
    for x in range(1, len(trans_file) + 1):
        day_count.append(x)
    trans_file['Day_Count'] = day_count
    print(trans_file)
    return trans_file, country_names

def create_lin_reg_plot(X, y, X_test, y_pred, country_name, days, intercept, coef, type):
    plt.scatter(X, y, color='gray')
    plt.plot(X_test, y_pred, color = 'red', linewidth=2)
    if(type == 'lin'):
        plt.title('Linear Regression of Daily Covid-19 Cases in ' + country_name + ' Across ' + days + ' Days')
        plt.ylabel('Covid-19 Cases')
        red_patch = mpatches.Patch(color='red', label= 'y = ' + str(round(intercept[0], 3)) + ' + ' + str(round(coef[0][0], 3)) + 'X')
    if(type == 'exp'):
        plt.title('Exponential Regression of Daily Covid-19 Cases in ' + country_name + ' Across ' + days + ' Days')
        plt.ylabel('Covid-19 Cases e^y')
        red_patch = mpatches.Patch(color='red', label= 'y = ' + str(round(intercept[0], 3)) + ' + ' + str(round(coef[0][0], 3)) + 'X')
    plt.xlabel('Day Count')
    plt.legend(handles=[red_patch])
    plt.show()

def create_linear_regression(df, country_name):
    print(country_name)
    X = df['Day_Count'].values.reshape(-1, 1)
    y = df[country_name].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print(regressor.intercept_)
    print(regressor.coef_)
    y_pred = regressor.predict(X_test)
    res = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    res.to_csv('.\Results\\' + country_name + 'Results.csv')
    create_lin_reg_plot(X, y, X_test, y_pred, country_name, str(len(df)), regressor.intercept_, regressor.coef_, 'lin')

##Code edits that I just made
def create_exp_regression(df, country_name):
    print(country_name)
    X = df['Day_Count'].values.reshape(-1, 1)
    y = df[country_name].values.reshape(-1, 1)
    y = np.where(y==0, .0000001, y)
    y = np.log(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print(regressor.intercept_)
    print(regressor.coef_)
    y_pred = regressor.predict(X_test)
    res = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    res.to_csv('.\Results\\' + country_name + 'Exponential_Results.csv')
    create_lin_reg_plot(X, y, X_test, y_pred, country_name, str(len(df)), regressor.intercept_, regressor.coef_, 'exp')

def main():
    df, country_names = initialize()
    #df.to_csv('Test.csv')
    create_linear_regression(df, "Total_Count")
    create_linear_regression(df, "US")
    create_exp_regression(df, "Total_Count")
    create_exp_regression(df, "US")
    #for name in country_names:
     #   create_linear_regression(df, name)
      #  exit()
        #create_plot(df, name)

main()
