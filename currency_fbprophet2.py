import pandas
from matplotlib import pyplot as plt
import fbprophet

RATES = pandas.read_csv('/storage/bin/exrates.csv', header=0)
RATES['ds'] = pandas.to_datetime(RATES['time.time'], unit='s')
RATES['y'] = RATES['www.exchangerates.org.uk']
maxts = max(RATES['time.time'])
RATES = RATES.drop(['hryvna.today', 'time.time', 'www.exchangerates.org.uk'], axis=1)

model = fbprophet.Prophet()
model.fit(RATES)
future = list()
for i in range(1,7): # 10 days
    future.append(maxts + 86400 * i)

future = pandas.DataFrame(future)
current = pandas.DataFrame(RATES['ds'])

future.columns = ['ds']
future['ds'] = pandas.to_datetime(future['ds'], unit='s')

data = pandas.concat([current, future], axis=0)
forecast = model.predict(data)
model.plot(forecast, xlabel='Date', ylabel='UAHs for 1 USD')

plt.show()

