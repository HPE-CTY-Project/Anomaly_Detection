import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from matplotlib import style
import statsmodels.api as sm
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score

st.header("Univariate Anomaly Detection using Z-score ");

image = Image.open("univariate.jpg");
resized_img = image.resize((1000, 270))
st.image(resized_img);
st.set_option('deprecation.showPyplotGlobalUse', False)
dataset = st.sidebar.file_uploader("**Choose your dataset:** ");
st.sidebar.write("---");
if dataset is not None:
	data = pd.read_csv(dataset)
	Time_Stamp = st.sidebar.selectbox("**Time Stamp Column:**", data.columns);
	col = st.sidebar.selectbox("**Select a feature**", data.columns, index=1);
	st.sidebar.write("---");
	cont_par = st.sidebar.slider("**Contamination Parameter(in %)**: ", min_value=1, max_value=10,help='Estimated percentage of anomalies present in the data');
	#st.sidebar.markdown("Estimated percentage of anomalies present in the data")
	cont_par = cont_par / 100
	try:
		data[Time_Stamp] = pd.to_datetime(data[Time_Stamp])
	except:
		st.subheader(f"{timestamp_col} is not a valid timestamp column")
	data.set_index(Time_Stamp, inplace=True)
	plt.style.use('ggplot')
	df = data.iloc[:, 0:-3:2]
	actual_anomalies = data.iloc[:, 1:-3:2]
	actual_anomalies.columns=actual_anomalies.columns.str.replace('_anomaly','')
	try:
		res = sm.tsa.seasonal_decompose(df[col], model='additive', period=24)
	except:
		st.subheader("Select a valid feature and  timestamp!")
		exit(1)
	predicted_anomalies_zscore = pd.DataFrame(index=data.index)
	residuals = pd.DataFrame({'residuals': res.resid}, index=data.index)

    		# calculate  zscore
	residuals['col_zscore'] = abs((residuals['residuals'] - residuals['residuals'].mean()) / residuals['residuals'].std(ddof=0))
	#threshold = 0.01
	#upper_bound = +(threshold)
	#lower_bound=-(threshold)
	#anomalies = abs(residuals[(residuals['col_zscore'] < lower_bound) | (residuals['col_zscore'] > upper_bound)])
    		# find anomalies
	top_n = int(np.ceil(len(data) * (cont_par)))
	top_n_outliers = residuals.sort_values(by='col_zscore', ascending=False).head(top_n)
	predicted_anomalies_zscore[col] = 0
	predicted_anomalies_zscore.loc[top_n_outliers.index, col] = 1
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df.index, y=df[col], name='Metric',mode='lines+markers',marker=dict(size=2.75, color="blue", symbol='circle', line={'width':1, 'color':"blue"})))
		 # Add scatter points for the anomalies
	anomaly_df = df.loc[top_n_outliers.index, [col]]
	z_scores = top_n_outliers['col_zscore']
	sizes = np.where(z_scores >8, 16, np.where(z_scores > 5, 13, 8))
	fig.add_trace(go.Scatter(
	x=anomaly_df.index,
        y=anomaly_df[col],
        name='Anomaly',
        mode='markers',
        marker=dict(color='red', symbol='triangle-up', size=sizes)
    	))
	#fig.add_trace(go.Scatter(
    	#x=df.index,
    	#y=df[col],
    	#name='',
    	#mode='markers',
    	#marker=dict(color='blue', size=3)  # Adjust the size value as desired
    	#))
	fig.update_layout(title=f'Feature: {col[0].upper()}{col[1:6]}-{col[7:]}',
	xaxis_title='Time-Stamps',
	yaxis_title='Data Values')
	st.plotly_chart(fig)
	st.write(f'**Number of anomalies detected**: {len(top_n_outliers)} ({round(cont_par*100)}%)')

	true_values = actual_anomalies[col].values
	predicted_values = predicted_anomalies_zscore[col].values
	precision = precision_score(true_values, predicted_values)
	accuracy = accuracy_score(true_values, predicted_values)
	f1 = f1_score(true_values, predicted_values)
	recall = recall_score(true_values, predicted_values)
	confusion=confusion_matrix(true_values, predicted_values)
    # Display the performance metrics


	def plot_confusion_matrix():
		sns.set(style='white')

		conf_mat = confusion_matrix(true_values, predicted_values)
		cm = np.array(conf_mat)
		fig, ax = plt.subplots(figsize=(5, 5))

		# Create a colormap with specified colors for the cells
		cmap = sns.color_palette(['red', 'palegreen'])

		sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 30},cmap=cmap, cbar=False,yticklabels=['0', '1'], xticklabels=['0', '1'], ax=ax)

# Define a list of colors for each cell border
		border_colors = ['blue', 'yellow', 'maroon', 'green']

# Calculate the adjusted rectangle size and position based on border width
		rect_width = 0.96
		rect_height = 0.96
		rect_x_offset = 0.02
		rect_y_offset = 0.02
# Iterate over each cell in the confusion matrix
		for i in range(2):
			for j in range(2):
        # Get the index of the border color based on the cell value
				border_index = i * 2 + j
        # Calculate the position of the rectangle
				rect_x = j + rect_x_offset
				rect_y = i + rect_y_offset
	# Add a border to each cell with the specified color
				rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,fill=False, edgecolor=border_colors[border_index], lw=4)
				ax.add_patch(rect)
		ax.xaxis.tick_top()
		ax.xaxis.set_label_position('top')
		ax.tick_params(labelsize=20, length=0)

#ax.set_title('Seaborn Confusion Matrix with Borders', size=24, pad=20)
		ax.set_xlabel('Predicted Values', size=20)
		ax.set_ylabel('Actual Values', size=20)
		plt.tight_layout()
		filename=f"cm_plots/{col}_cm.png"
		plt.savefig(filename)
		img=Image.open(filename)
		res1_img=img.resize((270, 250))
		st.image(res1_img)


	st.write("**Confusion Matrix:**")
	#st.write(confusion)
	plot_confusion_matrix()
	st.write("**Performance Metrics:**")
	data = {'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Value': [precision, recall, f1]}
	df = pd.DataFrame(data)
	hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
	# Inject CSS with Markdown
	st.markdown(hide_table_row_index, unsafe_allow_html=True)
	# Display a static table
	st.table(df)
