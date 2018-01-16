
# coding: utf-8

# In[1]:

# load dataset using read_csv()
from pandas import read_csv
series = read_csv('Energydata_spain.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(type(series))
print(series.head())


# In[2]:

# calculate descriptive statistics
from pandas import Series
series = Series.from_csv('Energydata_spain.csv', header=0)
print(series.describe())


# The describe() function creates a seven number summary of the loaded time series including mean, standard deviation, 
# median,minimum,and maximum of the observations. Descriptive statistics helps get an idea of the distribution and
# spread of values. This may help with ideas of data scaling and even data cleaning that we can perform later as part 
# of preparing our dataset for modeling.

# In[3]:

from matplotlib import pyplot
series.plot()
pyplot.show()


# Since the graph is very dense, I tried to make it less dense using same line plot with dots instead of the connected line.

# In[4]:

series.plot(style='k--.')
pyplot.show()


# It is not clearly evident that there is an overall increasing trend in the data along with some seasonal variations. So, more formally, we can check stationarity using the Dickey-Fuller Test

# In[5]:

series.hist()
pyplot.show()


# In[6]:

series.plot(kind='kde')
pyplot.show()


# Histogram shows a distribution that looks strongly Gaussian. The plotting function automatically selects the 
# size of the bins based on the spread of values in the data. Using KDE, we observe that perhaps the distribution is a little asymmetrical and perhaps a little pointy to be Guassian

# In[7]:

X = series.values
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:len(X)]
print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
pyplot.plot(train)
pyplot.plot([None for i in train] + [x for x in test])
pyplot.show()


# In[8]:

# calculate and plot monthly average
from pandas import Series
from matplotlib import pyplot
# create a boxplot of monthly data
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
from pandas import concat
resample = series.resample('M')
monthly_mean = resample.mean()
print(monthly_mean.head())
monthly_mean.plot()
pyplot.show()


# In[9]:

# create a boxplot of monthly data
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
from pandas import concat
one_year = series['2012']
groups = one_year.groupby(TimeGrouper('M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
months.boxplot()
pyplot.show()


# In[10]:

# create a heat map of monthly data
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
from pandas import concat
one_year = series['2012']
groups = one_year.groupby(TimeGrouper('M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
pyplot.matshow(months, interpolation=None, aspect='auto')
pyplot.show()


# In[11]:

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show()


# In[12]:

# autocorrelation plot of time series
from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=31)
pyplot.show()


# ACF captures the relationship of an observation with past observations in the same and opposite seasons 
# or times of year. Sine waves like this seen in this example are a strong sign of seasonality in the dataset.

# In[13]:

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series, lags=50)
pyplot.show()


# PCAF graphs also indicates seasonality.The PACF is better for AR models, and also shows the weekly and yearly seasons, although the correlation is lost faster with the lag.

# A simple way to correct for a seasonal component is to use differencing. 

# In[14]:

# deseasonalize a time series using differencing
from pandas import Series
from matplotlib import pyplot
X = series.values
diff = list()
days_in_year = 366
for i in range(days_in_year, len(X)):
    value = X[i] - X[i - days_in_year]
    diff.append(value)
print(diff[:10])
pyplot.plot(diff)
pyplot.show()


# In[15]:

# deseasonalize a time series using month-based differencing
from pandas import Series
from matplotlib import pyplot
X = series.values
diff = list()
days_in_year = 365
for i in range(days_in_year, len(X)):
	month_str = str(series.index[i].year)+'-'+str(series.index[i].month)
	month_mean_last_year = series[month_str].mean()
	value = X[i] - month_mean_last_year
	diff.append(value)
# calculate and plot monthly average
print(diff[:10])
pyplot.plot(diff)
pyplot.show()


# In[17]:

# deseasonalize by differencing with a polynomial model
from pandas import Series
from matplotlib import pyplot
from numpy import polyfit
# fit polynomial: x^2*b1 + x*b2 + ... + bn
X = [i%365 for i in range(0, len(series))]
y = series.values
degree = 4
coef = polyfit(X, y, degree)
# create curve
curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)
# create seasonally adjusted
values = series.values
diff = list()
for i in range(len(values)):
    value = values[i] - curve[i]
    diff.append(value)
pyplot.plot(diff)
pyplot.show()


# Dickey-Fuller Test-This is one of the statistical tests for checking stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Values for difference confidence levels. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary.

# In[16]:

from statsmodels.tsa.stattools import adfuller
X = series.values
result = adfuller(X)
print('Results of Dickey-Fuller Test:')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# Running the Dickey- fuller test gives the test statistic value of -5.9 The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset). As part of the output, I get a look-up table to help determine the ADF statistic ( statistic value of -5.9 is  less than the value of -3.4 at 1%. This suggests that we can reject the null hypothesis with a signifcance  level of less than 1%(i.e. a low probability that the result is a statistical  fluke). Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.

# In[17]:

from statsmodels.tsa.stattools import adfuller
Y = diff
result = adfuller(Y)
print('Results of Dickey-Fuller Test:')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[18]:

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(diff)
pyplot.show()

# autocorrelation plot of time series
from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(diff, lags=300)
pyplot.show()


# In[19]:

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(diff, lags=50)
pyplot.show()


# In[22]:

X = series.values
train_size = int(len(X) * 0.50)
train = X[0:train_size]
test=X[train_size:len(X)]
print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
pyplot.plot(train)
pyplot.plot([None for i in train] + [x for x in test])
pyplot.show()


# In[21]:

from math import sqrt
from statsmodels.tsa.ar_model import AR
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
        rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[23]:

# AutoRegressive Integrated Moving Average.
# fit an ARIMA model and plot residual errors
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
# load dataset

# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())


# In[24]:

# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
X = series.values
size = int(len(X) * 0.875)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[ ]:




# In[ ]:




# In[ ]:

# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.95)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
series = Series.from_csv('Energydata_spain.csv', header=0)
# evaluate parameters
p_values = [1, 2, 4, 6, 7, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)




