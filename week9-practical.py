# Save Model Using Pickle
import pandas
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

dataframe = pandas.read_csv('diabetes(1).csv', index_col=True)
array = dataframe.values
x = array[:, 0:8]
y = array[:, 8]
test_size = 0.33
seed = 7
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size, random_state=seed)

# Fit the model on training set
model = LogisticRegression(max_iter=500)
model.fit(x_train, y_train)

# Save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the model
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

# Task 1: Think of a use case scenario where you can build a fully function app to present
# prediction results

# A web application to help individuals monitor their health

# Goal of the app is to predict how likely a user is to develop medical conditions e.g. diabetes or heart
# disease

# these would be based on input data such as weight, age, activity level tc.

# Task 2 Create another model with LR to predict results and serialize model with picklefiles (Air
# Quality dataset)

df = pd.read_csv('AirQualityUCI-CSV.csv', sep=';', decimal=',', index_col=False)
df = df.dropna(axis=1, how='all')
df = df.dropna()
df = df.drop(columns=['Date', 'Time'], errors='ignore')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
X = df.drop(columns=['CO(GT)'])
y = df['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))

with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'linear_regression_model.pkl'")
