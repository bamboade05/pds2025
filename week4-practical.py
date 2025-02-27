numeric_only=True

# --- Calculating Mean and Median ---

import pandas as pd

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
 'Lee','Chanchal','Gasper','Naviya','Andres']),
 'Age':pd.Series([25,26,25,23,30,25,23,34,40,30,25,46])}

#Create a DataFrame
df = pd.DataFrame(d)
print("Mean Values in the Distribution")
print(df.mean())
print("*******************************")
print("Median Values in the Distribution")
print(df.median())


# --- Calculating Mode ---

# Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
 'Lee','Chanchal','Gasper','Naviya','Andres']),
 'Age':pd.Series([25,26,25,23,30,25,23,34,40,30,25,46])}
#Create a DataFrame
df = pd.DataFrame(d)
print(df.mode())

# --- Measuring Standard Deviation ---

import pandas as pd
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
 'Lee','Chanchal','Gasper','Naviya','Andres']),
 'Age':pd.Series([25,26,25,23,30,25,23,34,40,30,25,46]),
'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,
4.10,3.65])}

#Create a DataFrame
df = pd.DataFrame(d)

# Calculate the standard deviation
print(df.std())

# --- Measuring Skewness ---

import pandas as pd
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
 'Lee','Chanchal','Gasper','Naviya','Andres']),
 'Age':pd.Series([25,26,25,23,30,25,23,34,40,30,25,46]),
'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,
4.10,3.65])}

#Create a DataFrame
df = pd.DataFrame(d)
print(df.skew())

# --- Python - Normal Distribution ---

import matplotlib.pyplot as plt
import numpy as np
mu, sigma = 0.5, 0.1
s = np.random.normal(mu, sigma, 1000)

# Create the bins and histogram
count, bins, ignored = plt.hist(s, 20, normed=True)

# Plot the distribution curve
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
 np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=3,
color='y')

plt.show()

# --- Python – Correlation ---

import matplotlib.pyplot as plt
import seaborn as sns
df = sns.load_dataset('iris')
#without regression
sns.pairplot(df, kind="scatter")
plt.show()

# --- Finding Relationships : ---

import pandas as pd
df = pd.read_csv('data.csv')
print(df.corr())