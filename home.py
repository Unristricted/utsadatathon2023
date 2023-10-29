import streamlit as st
from sqlalchemy import create_engine 
import psycopg2 
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

connection_string = 'postgresql://yaseen:grapemangomelon@utsadatathon2023.crpb4cubcwub.us-east-1.rds.amazonaws.com:5432/test'

def query_to_get_all_states():
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor()
    #cursor.execute("SELECT \"Estimated Population 5-17\",  \"Estimated number of relevant children 5 to 17 years old in pove\" FROM public.ussd17")# WHERE \"State Postal Code\" = 'TX'")
    cursor.execute("SELECT DISTINCT \"State Postal Code\" FROM public.ussd17")
    results = cursor.fetchall()
    connection.close()

    results_list = [x[0] for x in results]
    results_list.append('All')
    results_list.sort()
    return results_list

def query_povertydata_by_state(state):
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor()
    if state == 'All':
        cursor.execute("SELECT \"Estimated Population 5-17\",  \"Estimated number of relevant children 5 to 17 years old in pove\" FROM public.ussd17")
    else:
        cursor.execute("SELECT \"Estimated Population 5-17\",  \"Estimated number of relevant children 5 to 17 years old in pove\" FROM public.ussd17 WHERE \"State Postal Code\" = '{}'".format(state))
    results = cursor.fetchall()
    connection.close()

    x = [x[0] for x in results]
    y = [x[1] for x in results]

    #calculate and remove outliers
    x = np.array(x)
    y = np.array(y)

    #zscore
    x_z = np.abs(stats.zscore(x))
    y_z = np.abs(stats.zscore(y))

    #remove outliers from each array and same index from other array
    x_outliers = np.where(x_z > 3)
    y_outliers = np.where(y_z > 3)

    outliers = np.concatenate((x_outliers[0], y_outliers[0]))
    outliers = np.unique(outliers)

    x = np.delete(x, outliers)
    y = np.delete(y, outliers)

    results_df = pd.DataFrame({'x': x, 'y': y})
    
    return results_df

st.write("Hello world!")

#all 50 states
state = st.selectbox('Select a state', query_to_get_all_states())

#query data for selected state
df = query_povertydata_by_state(state)

#calculate metrics
st.write(df.describe())


#plot data
st.scatter_chart(df, x='x', y='y')

#plot log-log data with log-log trendline
st.write("Log-log plot")
filtered_df = df[(df['x'] > 0) & (df['y'] > 0)]

x = np.log(filtered_df[['x']])
y = np.log(filtered_df['y'])

model = LinearRegression()

model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_

x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
y_fitted = model.predict(x_range)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', alpha=0.5, s=10)
plt.xlabel('log(x)')
plt.ylabel('log(y)')
plt.legend()
plt.plot(x_range, y_fitted, color='red', label = 'log(y) = {}log(x) + {}'.format(round(slope, 2), round(intercept, 2)))
fig = plt.gcf()
st.pyplot(fig)