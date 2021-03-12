import pandas
from matplotlib import pyplot as plt
import fbprophet

print('Prophet %s' % fbprophet.__version__)

RATES = pandas.read_csv('/storage/bin/exrates.csv', header=0)
RATES['ds'] = pandas.to_datetime(RATES['time.time'], unit='s')
RATES['y'] = RATES['www.exchangerates.org.uk']
maxts = max(RATES['time.time'])
RATES = RATES.drop(['hryvna.today', 'time.time', 'www.exchangerates.org.uk'], axis=1)
#plt.plot(RATES['ds'], RATES['y'])
model = fbprophet.Prophet()
model.fit(RATES)
future = list()
for i in range(1,5): # 10 days
    future.append(maxts + 86400 * i)
future = pandas.DataFrame(future)
future.columns = ['ds']
future['ds'] = pandas.to_datetime(future['ds'], unit='s')
forecast = model.predict(future)
# sprint(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
model.plot(forecast)
plt.show()

