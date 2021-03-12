import statsmodels
import csv
import pandas
from matplotlib import pyplot as plt
import numpy as np

data = []
with open('/storage/bin/exrates.csv') as acsv:
    exrtsreader = csv.reader(acsv, delimiter=',')
    for row in exrtsreader:
        if row:
            try:
                data.append((int(row[0]), float(row[2])))
            except ValueError:
                pass

RATES = pandas.read_csv('/storage/bin/exrates.csv')
X = np.array(RATES['time.time'])
XDATES = pandas.to_datetime(RATES['time.time'], unit='s')
Y = np.array(RATES['www.exchangerates.org.uk'])

from statsmodels.tsa.vector_ar.var_model import VAR
model = VAR(data)
model_fit = model.fit()
forecast_x_VAR = []
forecast_y_VAR = []
for astep in model_fit.forecast(model_fit.endog, steps=20):
    forecast_x_VAR.append(astep[0])
    forecast_y_VAR.append(astep[1])
forecast_xdates = pandas.to_datetime(forecast_x_VAR, unit='s')
plt.plot(XDATES, Y, label='www.exchangerates.org.uk')
plt.plot(forecast_xdates, forecast_y_VAR, 'g', label='VAR forcast')

from statsmodels.tsa.statespace.varmax import VARMAX
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
forecast_x_VARMAX = []
forecast_y_VARMAX = []
for astep in model_fit.forecast(20):
    forecast_x_VARMAX.append(astep[0])
    forecast_y_VARMAX.append(astep[1])
forecast_xdates = pandas.to_datetime(forecast_x_VARMAX, unit='s')
plt.plot(forecast_xdates, forecast_y_VARMAX, 'r', label='VARMAX smoothing')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
# fit model
model = ExponentialSmoothing(Y, seasonal_periods=5)
model_fit = model.fit()
yhat = model_fit.predict(len(Y), len(Y))
print(max(XDATES))
plt.plot(max(XDATES), yhat, 'ro', label='Holt Winter’s Exponential smoothing')


plt.xlabel('Date')
plt.ylabel('UAHs for 1 USD')
plt.legend()
plt.show()
