import pandas
import pandas as pd
import numpy as np
from pandas._config import dates

# --- Object creation ---

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s, '\n')

# creating a dataframe by passing dictionary of objects that can be converted into
# a series-like structure

df2 = pd.DataFrame({
 "A": 1.0,
 "B": pd.Timestamp("20130102"),
 "C": pd.Series(1, index=list(range(4)), dtype="float32"),
 "D": np.array([3] * 4, dtype="int32"),
 "E": pd.Categorical(["test", "train", "test", "train"]),
 "F": "foo",
 })

print('Dataframe: \n \n', df2, '\n')

print('Dataframe types: \n \n', df2.dtypes, '\n')

# --- Viewing data ---

print(df2.head(), '\n')
print(df2.tail(3), '\n')
print(df2.index, '\n') # display index
print(df2.columns, '\n') # display columns
print(df2.to_numpy(), '\n')
print(df2.describe(), '\n') # quick statistic summary of data

# -- Transposing your data --
print(df2.T, '\n')
print(df2.sort_index(axis=1, ascending=False), '\n') # sort by an axis
print(df2.sort_values(by="B"), '\n')
print(df2["A"], '\n')
print(df2[0:3], '\n')
# print(df2["20130102":"20130104"], '\n')

# -- Selection by label --

# print(df2.loc[dates[0]], '\n') # getting a cross section using a label
print(df2.loc[:, ["A", "B"]], '\n') # selecting a multi-axis by label
print(df2.loc["20130102":"20130104", ["A", "B"]], '\n') # label slicing, where both endpoints are included
# print(df2.loc["20130102", ["A", "B"]], '\n') # reduction in the dimensions of the returned object
# print(df2.loc[dates[0], "A"], '\n') # for getting a scalar value
# print(df2.at[dates[0], "A"], '\n') # for getting fat access to a scalar

# -- Selection by position --

print(df2.iloc[3], '\n') # select via the position of passed integers

# -- By integer slices, acting similar to NumPy/Python --

print(df2.iloc[3:5, 0:2], '\n')

# -- By lists of integer position locations, similar to NumPy/Python style --

# print(df2.iloc[[1, 2, 4], [0, 2]], '\n')

# -- For slicing rows explicitly --

print(df2.iloc[1:3, :], '\n')

# -- For slicing columns explicitly --

print(df2.iloc[:, 1:3], '\n')

# -- For getting a value explicitly --

print(df2.iloc[1, 1], '\n')

# -- For getting fast access to a scalar (equivalent to the prior method) --

print(df2.iat[1, 1], '\n')

# --- Boolean indexing ---

print(df2[df2["A"] > 0], '\n')

# print(df2[df2 > 0], '\n') # selecting values where a boolean condition is met

# -- Using the isin() method for filtering --

df2 = df2.copy() # supposed to be df.copy() but it doesn't exist
print(df2, '\n')

print(df2[df2["E"].isin(["two", "four"])], '\n')

# --- Settings ---

# - Setting a new column automatically aligns the data by the indexes

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
print(s1, '\n')

df2["F"] = s1
print(df2, '\n')

# -- Setting values by label: --

# df2.at[dates[0], "A"] = 0

# -- Setting values by position --

# df2.iat[0, 1] = 0

# -- Setting by assigning with a NumPy array --

# df2.loc[:, "D"] = np.array([5] * len(df2))

print(df2, '\n') # result of the prior setting operations

# -- A where operation with setting --

df2 = df2.copy()
# df2[df2 > 0] = -df2
print(df2, '\n')

# --- String Methods ---

s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
print(s, '\n')

s.str.lower()
print(s, '\n')

# --- Time series ---

rng = pd.date_range("1/1/2012", periods=100, freq="S")
print(rng, '\n')

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts, '\n')

print(ts.resample("5Min").sum(), '\n')

# -- Series.tz_localize() localizes a time series to a time zone --

rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts, '\n')

ts_utc = ts.tz_localize("UTC")
print(ts_utc, '\n')

# Series.tz_convert() converts a timezones aware time series to another time zone

print(ts_utc.tz_convert("US/Eastern"), '\n')