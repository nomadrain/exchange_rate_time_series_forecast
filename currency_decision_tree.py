import pandas
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

RATES = pandas.read_csv('/storage/bin/exrates.csv')
RATES['Date'] = pandas.to_datetime(RATES['time.time'], unit='s')
rates = RATES.copy()

from sklearn.model_selection import train_test_split

X = np.array(rates['Date']).reshape(-1, 1)
Y = np.array(rates['www.exchangerates.org.uk'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=1)
DT_model = DecisionTreeRegressor(max_depth=20).fit(X_train, Y_train)
DT_predict = DT_model.predict(X_test)

plt.plot(X, Y)
plt.plot(X_test, DT_predict, 'go', label='line 1',)

future_x = []
max_sample_date = max(X)
for i in range(1, 11):
    future_x.append( max_sample_date + i * 24 * 60 * 60 * 1000000000)
future_x = np.array(future_x).reshape(-1, 1)
future_y = DT_model.predict(future_x)
plt.plot(future_x, future_y, 'ro')
plt.xlabel('Date')
plt.ylabel('UAHs for 1 USD')
plt.show()
