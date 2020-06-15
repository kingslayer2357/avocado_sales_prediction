# -*- coding: utf-8 -*-
"""
Created on Mon May 25 04:19:04 2020

@author: kingslayer
"""
### FOR ALL REGIONS ###
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet


avocado_df=pd.read_csv("avocado.csv")

avocado_df=avocado_df.sort_values("Date")
plt.plot(avocado_df["Date"],avocado_df["AveragePrice"])

sns.countplot(y="region",data=avocado_df)
sns.countplot(y="year",data=avocado_df)


avocado_prophet=avocado_df[["Date","AveragePrice"]]
avocado_prophet.columns=["ds","y"]

m=Prophet()
m.fit(avocado_prophet)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)

m.plot(forecast,xlabel="Date",ylabel="Average Price of Avocado",figsize=(10,10))
m.plot_components(forecast)

avocado_df["region"]

### REGION SPECIFIC ###

avocado_df=avocado_df[avocado_df["region"]=="California"]
avocado_df=avocado_df.sort_values("Date")
plt.plot(avocado_df["Date"],avocado_df["AveragePrice"])


sns.countplot(y="year",data=avocado_df)

avocado_prophet=avocado_df[["Date","AveragePrice"]]
avocado_prophet.columns=["ds","y"]

m=Prophet()
m.fit(avocado_prophet)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)

m.plot(forecast,xlabel="Date",ylabel="Average Price")
m.plot_components(forecast)