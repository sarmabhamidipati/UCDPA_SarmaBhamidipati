"""
This program detects breast cancer based on wisconsin data.
Data Source : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

It also detects NY Weather Forecast Data
Data Source : https://www.timeanddate.com/weather/usa/new-york/ext

"""

# Import libraries for data loading , plotting , train test split, supervised learning models
# classification and tuning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Import sklearn related modules like train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# import requests for web scraping for weather forecast
import requests
# Beautiful Soup is a Python library for pulling data out of HTML and XML files
from bs4 import BeautifulSoup
# import regular expressions package
import re

#

# Data Loading using pandas
bc_df = pd.read_csv('data/data.csv')
# Print the first 5 rows of data
print(bc_df.head())
# count the no of rows and columns in the data set
print(bc_df.shape)
# get information about columns
print(bc_df.info())
# print column information.
print(bc_df.columns)

# Data cleaning
# check for missing data and get the count of empty values in each column
# count = 0 means no missing data otherwise missing data
print(bc_df.isna().sum())
# drop the columns or features which are not required in this case it is 'ID', and 'Unnamed: 32'
bc_df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
# get the new count with the no of rows and columns in the data set
print(bc_df.shape)
# get information about columns after dropping column 'Unnamed: 32'
print(bc_df.info())
# Print the first 5 rows of data after dropping columns
print(bc_df.head())

# Get the unique count of each column/Feature
bcw_dict = {}
# print(list(df.columns))
for i in list(bc_df.columns):
    bcw_dict[i] = bc_df[i].value_counts().shape[0]
# print(bcw_dict)
df_cnt = pd.DataFrame(bcw_dict, index=["unique count"]).transpose()
print(df_cnt)

# Get a count of the number of Malignant(M) or Benign(B) cells. Benign means no cancer.
print(bc_df['diagnosis'].value_counts())

# List of different plot types used.
plot_type = ['countplot', 'pairplot', 'heatmap']


# define a custom function for plotting dataframe data.
def custom_sns_plot(p_df, p_plot_type, axis):
    if p_plot_type in plot_type:
        if p_plot_type == 'countplot':
            sns.countplot(x="diagnosis", hue="diagnosis", data=p_df).set(title='Malignant vs Benign Count')
            return

        elif p_plot_type == 'pairplot':
            sns.pairplot(p_df, hue="diagnosis")
            return
        elif p_plot_type == 'heatmap':
            axis = sns.heatmap(p_df, linewidths=.5, annot=True, fmt='.0%', vmin=0, vmax=1, center=0). \
                set(title='Correlation using Heat Map')
            return axis


# Represent this diagnosis count graphically using sea born
custom_sns_plot(bc_df, 'countplot', '')
plt.savefig('image/malignant_vs_benign.png')
plt.show()

# get the information about datatypes to see which columns needs to be encoded
print(bc_df.dtypes)
# Transform categorical value to numerical format
# convert Diagnosis feature data to numerical form using encoding and convert Benign : 0, Malignant 1
labelencoder_Y = LabelEncoder()
# df.diagnosis = labelencoder_Y.fit_transform(df.diagnosis)
bc_df.iloc[:, 0] = labelencoder_Y.fit_transform(bc_df.iloc[:, 0].values)

# get the information about datatypes after the column is encoded.
print(bc_df.dtypes)

# print(df.iloc[:, 0:5])
# Create a pair plot
custom_sns_plot(bc_df.iloc[:, 0:5], 'pairplot', '')
plt.savefig('image/diagnosis_pair_plot.png')
plt.show()

# Print the new data set
bc_df.head(5)

# Correlation and Heat Map
# get the correlation of the columns
df_corr = bc_df.iloc[:, 0:11].corr()

# print(df_corr)

# Represent correlation with a graph from columns diagnosis up to fractal_dimension_mean
# we are not taking all columns for this heat map
f, ax = plt.subplots(figsize=(12, 12))
ax = custom_sns_plot(df_corr, 'heatmap', ax)
plt.savefig('image/corr_heatmap.png')
plt.show()

# Split the data set into independent/Features/Predictor Variables (X) and dependent data/Target/Response Variable(y)
# X has the features of the cancer patients
X = bc_df.iloc[:, 1:31].values
# y has the diagnosis whether the patient has cancer or not
y = bc_df.iloc[:, 0].values

# Training And Test Data
# Split the data into 75% training and 25% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardization and Normalization
# Scale the data
scaler = StandardScaler()
# transform independent data and store this back to the same independent data
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Create a custom function for supervised learning models and fit the data.
# Create a function for the model to detect cancer
def models_to_detect_cancer(i_x_train, i_y_train):
    # Logistic Regression Model
    lrm = LogisticRegression(random_state=0)
    lrm.fit(i_x_train, i_y_train)

    # Decision Tree Classifier
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
    dtc.fit(i_x_train, i_y_train)

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rfc.fit(i_x_train, i_y_train)

    # k-Nearest Neighbours
    knn = KNeighborsClassifier()
    knn.fit(i_x_train, i_y_train)

    # Support Vector Classification
    svm = SVC(kernel='rbf', random_state=0)
    svm.fit(i_x_train, i_y_train)

    # print the model accuracy on the training data
    print('model accuracy on the training data')
    print('[0]Logistic Regression Training Accuracy:', lrm.score(i_x_train, i_y_train))
    print('[1]Decision Tree Classifier Training Accuracy:', dtc.score(i_x_train, i_y_train))
    print('[2]Random Forest Classifier Training Accuracy:', rfc.score(i_x_train, i_y_train))
    print('[3]k-Nearest Neighbors Training Accuracy:', knn.score(i_x_train, i_y_train))
    print('[4]Support Vector Classification Training Accuracy:', svm.score(i_x_train, i_y_train))
    print()
    return lrm, dtc, rfc, knn, svm


# Get all of the four models used
model = models_to_detect_cancer(X_train, y_train)

print(type(model), len(model))
for i in range(len(model)):
    print('Model = ', i, model[i])

# Data Prediction and classification

# Logistic Regression Model Predictions
ypred_lorm = model[0].predict(X_test)
cm_lorm = confusion_matrix(y_test, ypred_lorm)
acc_lorm = accuracy_score(y_test, ypred_lorm)

# Decision Tree Classifier Model Predictions
ypred_dtc = model[1].predict(X_test)
cm_dtc = confusion_matrix(y_test, ypred_dtc)
acc_dtc = accuracy_score(y_test, ypred_dtc)

# Random Forest Classifier Model Predictions
ypred_rfc = model[2].predict(X_test)
cm_rfc = confusion_matrix(y_test, ypred_rfc)
acc_rfc = accuracy_score(y_test, ypred_rfc)

# k-Nearest Neighbors Model Predictions
ypred_knn = model[3].predict(X_test)
cm_knn = confusion_matrix(y_test, ypred_knn)
acc_knn = accuracy_score(y_test, ypred_knn)

# Support Vector Classification Model Predictions
ypred_svm = model[4].predict(X_test)
cm_svm = confusion_matrix(y_test, ypred_svm)
acc_svm = accuracy_score(y_test, ypred_svm)

# Accuracy score comparision across all the models chosen above
# create a list containing model and accuracy score
prediction_columns = ["Model", "Accuracy Score"]

# create a dictionary with Key as Model and Value Pair as Accuracy Score
score_dict_pred = {"Model": ["Logistic Regression",
                             "Decision Tree Classifier",
                             "Random Forest Classifier",
                             "k-Nearest Neighbors",
                             "Support Vector Classification"
                             ],
                   "Accuracy Score": [format(acc_lorm),
                                      format(acc_dtc),
                                      format(acc_rfc),
                                      format(acc_knn),
                                      format(acc_svm)
                                      ]
                   }

score_df_predictions = pd.DataFrame(score_dict_pred)

print(score_df_predictions.sort_values(by='Accuracy Score', ascending=False))

print('It is clear from our score that model Random Forest Classifier has the highest accuracy score of',
      score_df_predictions['Accuracy Score'].max(), 'which is (97.90%)')

# Use Hyper-Parameter Tuning model for Random Forest Classifier to see if we can increase its score

param_grid = [{'n_estimators': [100, 200, 300],
               'max_features': ['auto'],
               'max_depth': [10, 20, 30],
               'min_samples_leaf': [1, 2],
               'min_samples_split': [2]}]

grid = GridSearchCV(estimator=model[2], param_grid=param_grid, scoring='roc_auc', cv=10, n_jobs=-1)
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
print()
print()

print("Weather Forecast for the next 14 days")
"""
# This below program forecasts weather data for the next 14 days using Web Scraping
# This is to demonstrate web scraping, regular expressions and merging two dataframes.
"""
# getting 14 day forecast for New York.

# This gets new york weather forecast for the next 14 days
page = requests.get("https://www.timeanddate.com/weather/usa/new-york/ext")

soup = BeautifulSoup(page.content, "html.parser")

table = soup.find_all("table", {"class": "zebra tb-wt fw va-m tb-hover"})

weather_data_list = []
for i, items in enumerate(table):
    for j, row in enumerate(items.find_all("tr")):
        d = {}
        try:
            d['Temp'] = row.find_all("td", {"class": ""})[0].text
        except:
            d['Temp'] = np.nan

        try:
            d['Weather'] = row.find("td", {"class": "small"}).text
        except:
            d['Weather'] = np.nan

        try:
            d['Visibility'] = row.find_all("td", {"class": ""})[3].text
        except:
            d['Visibility'] = np.nan

        weather_data_list.append(d)

weather1 = pd.DataFrame(weather_data_list)
weather2 = weather1.dropna(how='all')
weather2 = weather2.reset_index()
weather2.pop('index')
weather2['Weather'] = weather2['Weather'].str.replace(" ", "")
weather2['Visibility'] = weather2['Visibility'].str.extract('(\d+)') + 'mi'
weather2['Temp'] = weather2['Temp'].str.extract('(\d+)') + u'\N{DEGREE SIGN}' + 'F'
weather2['Temp_in_number'] = weather2['Temp'].str.extract('(\d+)')

dayno = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
df = pd.DataFrame(dayno, columns=['Day'])

pd.set_option('display.max_columns', None)
# merge two data frames.
result = pd.concat([df, weather2], axis=1)

print(result)
