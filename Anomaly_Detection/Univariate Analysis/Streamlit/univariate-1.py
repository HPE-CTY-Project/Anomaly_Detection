import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from matplotlib import style
import statsmodels.api as sm
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score

st.header("Univariate Anomaly Detection ");

image = Image.open("univariate.jpg");
resized_img = image.resize((1000, 270))
st.image(resized_img);
dataset = st.sidebar.file_uploader("Choose your dataset: ");
st.sidebar.write("---");
if dataset is not None:
	data = pd.read_csv(dataset)
	method = st.sidebar.selectbox("Univariate Method: ", ["IQR", "Grubb's Test", "Z-Score", "Moving Average"]);
	st.sidebar.write("---");
	Time_Stamp = st.sidebar.selectbox("Time Stamp Column: ", data.columns);
	col = st.sidebar.selectbox("Select a feature", data.columns, index=1);
	st.sidebar.write("---");
	if(method != "Grubb's Test"):
		cont_par = st.sidebar.slider(
		    "Contamination Parameter(in %): ", min_value=1, max_value=10);
		cont_par = cont_par / 100

# --------------------------------------IQR---------------------------------------------------------------------------------------------------
	if method == "IQR":
		st.header("IQR")
		data.set_index(Time_Stamp, inplace=True)
		plt.style.use('ggplot')
		df = data.iloc[:, :]
		# st.write(df.columns)
		try:
			res = sm.tsa.seasonal_decompose(df[col], model='additive', period=24)
		except:
			st.subheader("Select a valid feature and time-stamp!")
			exit(1)
		residuals = pd.DataFrame({'residuals': res.resid}, index=df.index)
    # calculate  IQR
		q1 = residuals['residuals'].quantile(0.25)
		q3 = residuals['residuals'].quantile(0.75)
		iqr = q3 - q1

    # calculate upper and lower bounds based on the rolling IQR
		lower_bound = (q1 - 0.55 * iqr)
		upper_bound = (q3 + 0.55 * iqr)

    # find anomalies
		anomalies = residuals[(residuals['residuals'] < lower_bound) | (
		    residuals['residuals'] > upper_bound)]
		anomalies['anomalies_diff'] = abs(anomalies['residuals'].sub(upper_bound).where(
		    anomalies['residuals'] > upper_bound, other=anomalies['residuals'].sub(lower_bound)))
		top_n = int(np.ceil(len(df) * (cont_par)))
		top_n_outliers = anomalies.sort_values(
		    by='anomalies_diff', ascending=False).head(top_n)

    # Create a Plotly figure with a line plot of the normal data
		fig = go.Figure()
		fig.add_trace(go.Scatter(
		    x=df.index, y=df[col], name='Normal', line=dict(color='blue')))

    # Add scatter points for the anomalies
		anomaly_df = df.loc[top_n_outliers.index, [col]]
		fig.add_trace(go.Scatter(x=anomaly_df.index,
		              y=anomaly_df[col], name='Anomaly', mode='markers', marker=dict(color='red')))
		fig.update_layout(title=f'{col}', xaxis_title='Date', yaxis_title=col)
		st.plotly_chart(fig)
		st.write(f'Number of anomalies detected: {len(top_n_outliers)}')
		st.write(f'Contamination parmeter: {round(cont_par*100)}%')
		# st.write(len(outliers))
# -------------------------grub's---------------------------------------------------------------------------------------------------------------
	elif method == "Grubb's Test":
		st.header("Grubb's Test")
		if col is not None:
			# data = pd.read_csv(dataset, parse_dates=[Time_Stamp])
			data.set_index(Time_Stamp, inplace=True)
			plt.style.use('ggplot')
			try:
				stl = STL(data[col], period=24)
			except:
				st.subheader("Select a valid feature and time-stamp!")
				exit(1)
			stl = STL(data[col], period=24)
			result = stl.fit()
			seasonal, trend, resid = result.seasonal, result.trend, result.resid
    			# Apply Grubbs' test to detect anomalies in the residual component
			mean = np.mean(resid)
			std_dev = np.std(resid)
			alpha = 0.05
			outliers = []
			while True:
				n = len(resid)
				t_value = t.ppf(1 - alpha / (2 * n), n - 2)
				G_critical = (n - 1) / np.sqrt(n) * \
				              np.sqrt(t_value * 2 / (n - 2 + t_value * 2))
				G = np.max(np.abs(resid - mean)) / std_dev
				if G > G_critical:
					outlier_idx = np.argmax(np.abs(resid - mean))
					outliers.append(outlier_idx)
					resid = resid[resid != resid[outlier_idx]]
					n = len(resid)
					mean = np.mean(resid)
					std_dev = np.std(resid)
					G_critical = (n - 1) / np.sqrt(n) * \
					              np.sqrt(t_value * 2 / (n - 2 + t_value * 2))
				else:
					break

    			# Plot the original data with anomalies highlighted
			fig = go.Figure()
			fig.add_trace(go.Scatter(
			    x=data.index, y=data[col], mode='lines', name='Original Data', line=dict(color='blue')))
			fig.add_trace(go.Scatter(x=data.iloc[outliers].index, y=data.iloc[outliers]
			              [col], mode='markers', name='Anomalies', marker=dict(color='red', size=6)))
			fig.update_layout(title=f'{col}', xaxis_title='Date', yaxis_title=col)
			st.plotly_chart(fig)
			st.write(f'Number of anomalies detected: {len(outliers)}')
# --------------------------------------Z-Score-----------------------------------------------------------------------------------------------
	elif method == "Z-Score":
		st.header("Z-Score")
		try:
			data[Time_Stamp] = pd.to_datetime(data[Time_Stamp])
		except:
			st.subheader(f"{timestamp_col} is not a valid timestamp column")
		data.set_index(Time_Stamp, inplace=True)
		plt.style.use('ggplot')
		df = data.iloc[:, :]
		try:
			res = sm.tsa.seasonal_decompose(df[col], model='additive', period=24)
		except:
			st.subheader("Select a valid feature and  timestamp!")
			exit(1)
		residuals = pd.DataFrame({'residuals': res.resid}, index=data.index)

    		# calculate  zscore
		residuals['col_zscore'] = (
		    (residuals['residuals'] - residuals['residuals'].mean()) / residuals['residuals'].std(ddof=0))
    		# calculate upper and lower bounds
		lower_bound = -1
		upper_bound = 1
		anomalies = abs(residuals[(residuals['col_zscore'] < lower_bound) | (
		    residuals['col_zscore'] > upper_bound)])
    		# find anomalies
		top_n = int(np.ceil(len(data) * (cont_par)))
		top_n_outliers = anomalies.sort_values(
		    by='col_zscore', ascending=False).head(top_n)
		fig = go.Figure()
		fig.add_trace(go.Scatter(
		    x=data.index, y=data[col], mode='lines', name='Original Data', line=dict(color='blue')))
		fig.add_trace(go.Scatter(x=top_n_outliers.index,
		              y=df.loc[top_n_outliers.index, col], mode='markers', name='Anomalies', marker=dict(color='red', size=6)))
		fig.update_layout(title=f'{col}',
                xaxis_title='Date',
                yaxis_title='data values')
		st.plotly_chart(fig)
		st.write(f'Number of anomalies detected: {len(top_n_outliers)}')
		st.write(f'Contamination parmeter: {round(cont_par*100)}%')
# -----------------------------Moving Average---------------------------------------------------------------------------------------------------
	elif method == "Moving Average":
		#cont_par=cont_par*100
		st.header("Moving Average")
		#data = pd.read_csv(dataset)
		data[Time_Stamp] = pd.to_datetime(data[Time_Stamp])
		data.set_index(Time_Stamp, inplace=True)
		df = data.iloc[:, 0:-3:2]
		actual_anomalies = data.iloc[:, 1:-3:2]
		predicted_anomalies = pd.DataFrame(columns=df.columns, index=df.index)
		plt.style.use('ggplot')
		plt.rcParams['figure.figsize'] = [20, 5]
		fig = go.Figure()
		lambda_list = [1.5,1.75,2, 2.25, 2.5]
		window_size_list = [12, 24]

		len_lambda = len(lambda_list)
		len_window = len(window_size_list)
		count = len_lambda * len_window

		no_col = len(df.columns)
		no_rows = df.shape[0]

		df1 = pd.DataFrame({'Lambda': [], 'Window Size': [], 'p_anomalies': []})

		for l in lambda_list:
			for w in window_size_list:
				col_anomalies = 0
				for col1 in df.columns:
					stl = STL(df[col1])
					result = stl.fit()
					seasonal, trend, resid = result.seasonal, result.trend, result.resid
					rolling_mean = resid.rolling(window=w).mean()
					std = resid.rolling(window=w).std()
					threshold = l * std
					upper_limit = rolling_mean + threshold
					lower_limit = rolling_mean - threshold
					anomalies = (resid > upper_limit) | (resid < lower_limit)
					num_anomalies = anomalies.sum()
					col_anomalies = col_anomalies + num_anomalies
				avg_anomalies = col_anomalies / no_col
				new_row = {'Lambda': l, 'Window Size': w,'p_anomalies': (avg_anomalies * 100) / no_rows}
				df1 = df1.append(new_row, ignore_index=True)

		df1['Difference'] = abs(cont_par*100 - df1['p_anomalies'])
		df1_sorted = df1.sort_values(by=['Difference'])
		first_row = df1_sorted.iloc[0]
		min_l = first_row['Lambda']
		min_w = first_row['Window Size']
		p = first_row['p_anomalies']
		min_lamda=min_l
		min_window_size=min_w
		#'''st.write(f"Min lambda value: {min_l}")
		#st.write(f"Min window size: {min_w}")
		#st.write(df1_sorted)'''
		w=int(min_w)
		l=int(min_l)
		stl = STL(df[col])
		result = stl.fit()
		resid =  pd.DataFrame({'resid': result.resid}, index=df.index)
		rolling_mean = resid['resid'].rolling(window=w).mean()
		std = resid['resid'].rolling(window=w).std()
		threshold = l * std
		upper_bound = rolling_mean + threshold
		lower_bound = rolling_mean - threshold   
		anomalies = resid[(resid['resid'] < lower_bound) | (resid['resid'] > upper_bound)]
		anomalies['anomaly_diff'] = abs(anomalies['resid'].sub(upper_bound.loc[anomalies.index]).where(anomalies['resid'] > upper_bound.loc[anomalies.index], other=anomalies['resid'].sub(lower_bound.loc[anomalies.index])))
		#st.write(len(anomalies))
		predicted_anomalies[col] = 0
		top_n = int(np.ceil(len(data) * cont_par))
		#st.write(top_n)
		top_n_outliers = anomalies.sort_values(by='anomaly_diff', ascending=False).head(top_n)
		predicted_anomalies.loc[top_n_outliers.index, col] = 1
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=df.index, y=df[col], name='Normal', line=dict(color='blue')))
		anomaly_df = df.loc[top_n_outliers.index, [col]]
		fig.add_trace(go.Scatter(x=anomaly_df.index, y=anomaly_df[col], name='Anomaly', mode='markers', marker=dict(color='red')))
		fig.update_layout(title=col,xaxis_title='Date',yaxis_title='Value')
		st.plotly_chart(fig)
		st.write(f'Number of anomalies detected: {len(top_n_outliers)}')
		st.write(f'Contamination parmeter: {round(cont_par*100)}%')
# -----------------------------------------------------------------------------------------------------------------------------------------

else:
	pass
