# Save Model Using Pickle
import pandas
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

airquality = pd.read_csv('AirQualityUCI-CSV.csv')
df = pd.DataFrame(airquality)
# print(df.head(5))


#
x = df[['NO2(GT)', 'PT08.S4(NO2)']]
y = df['C6H6(GT)']
print(x)
print(y)
# Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;

# res = list(airquality.columns)
# print(res)

# X = airquality.columns[columns]
# print(X)
# x = aq_array[:, 0:16]
# y = aq_array[:, 8]

test_size = 0.33
seed = 7
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size, random_state=seed)

# Fit the model on training set
aq_model = LogisticRegression(max_iter=500)
aq_model.fit(x_train, y_train)

# # Save the model to disk
# filename = 'finalized_aq_model.sav'
# pickle.dump(aq_model, open(filename, 'wb'))
#
# # Load the model
# loaded_aq_model = pickle.load(open(filename, 'rb'))
# result = loaded_aq_model.score(x_test, y_test)
# print(result)
