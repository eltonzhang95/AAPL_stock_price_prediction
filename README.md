# Time Series Forecasting on Apple Stock Price
The aim of this project is to make time series forecasting on Apple stock price. The prediction periods are 1 day, 7 days, 1 month, and 3 months. Code are written in both Python and Scala. Code structure is optimized to be excuted by Apache Spark.  
### Methodology
Both statistic and machine learning models are developed for TS forecasting. This repo contains the code for data pre-processing for both stats and ML models, as well as the tuning of ARIMA model.  
  
For the ML model, multiple indexes such as SP500, NASDAQ100, currency exchange rates, etc. are engineered as features, along with the stock prices of Apple's suppliers and competitors. For the ARIMA model, statistical tools such as adfuller test, ACF and PACF plot are utilized to identify the stability and seasonality of the data. 
