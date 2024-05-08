import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing

# Load the CSV file as a DataFrame
df = pd.read_csv('weather.csv')

# Display the size of the weather data
print('Size of the weather data:', df.shape)

# Display the first 5 rows of the DataFrame
print(df[0:5])



# Checking null values
print(df.count().sort_values())

# Drop some columns
df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Date', 'Location'], axis=1)

print(df.shape)

# Get rid of null values
df = df.dropna(how='any')
print(df.shape)

# Get rid of outliers
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df = df[(z < 3).all(axis=1)]
print(df.shape)

# Changing the value of RainToday and RainTomorrow
df['RainToday'] = df['RainToday'].replace({'No': 0, 'Yes': 1})
df['RainTomorrow'] = df['RainTomorrow'].replace({'No': 0, 'Yes': 1})

# See unique values and convert them to int
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))
#transform the categorical columns
df = pd.get_dummies(df,columns=categorical_columns)
print(df.iloc[4:9])

# Normalize the dataset
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

print(df.iloc[4:10])
print(df.columns)




# Perform feature selection
from sklearn.feature_selection import SelectKBest, chi2
X = df.loc[:,df.columns!='RainTomorrow']
y = df['RainTomorrow']
selector = SelectKBest(chi2, k=3)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)]) # top 3 columns


# Assign important features as x
df = df[['RainToday', 'Rainfall', 'Humidity3pm', 'RainTomorrow']]
x = df[['Humidity3pm']]
y = df[['RainTomorrow']]

# Data Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Calculating the accuracy and the time taken by the classifier
t0 = time.time()
# Data splicing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf_logreg = LogisticRegression(random_state=0)
clf_logreg.fit(X_train, y_train)

# Evaluating the model using testing data set
y_pred = clf_logreg.predict(X_test)
score = accuracy_score(y_test, y_pred)

# Print the score
print('accuracy is ', score)
print('time taken', time.time()-t0)

