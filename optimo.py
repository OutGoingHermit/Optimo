import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet 
from prophet.plot import plot_plotly
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import copy
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from io import BytesIO

def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*10000
	fig = px.line(daily_cum_returns, title=title)
	return fig

def plot_efficient_frontier_and_max_sharpe(mu, S): 
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S)
	fig, ax = plt.subplots(figsize=(6,4))
	ef_max_sharpe = copy.deepcopy(ef)
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

#set time period
START = "1980-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#primary page title and add disclaimer
st.set_page_config(page_title = "Optimo", layout = "wide")
st.title("Portfolio Optimization Calculator")


#load in ticker symbols for query form
stocks = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/0.csv")["Symbol"].values
        

#add multiselect(from selectbox) for portfolio builder // integrate returns into output
selected_stocks = st.multiselect("Portfolio", stocks)
st.caption('Enter more than 5 stock TICKERS to be included in portfolio')

try:
	#load yahoo finance data
	@st.cache_data
	def load_data(tickers):
		data = yf.download(tickers, START, TODAY)["Adj Close"]
		data.reset_index(inplace=False)
		return data
				
	st.subheader('Portfolio Data')
	
	#data loading interactive component
	data_load_state = st.text("Data loading...")
	data = load_data(selected_stocks)
	data_load_state.text("Data loading... finished")
				
	#calculating covariance matrix
	sample_cov = risk_models.sample_cov(data, frequency=252)
	S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
	
	#calculate expected returns based on CAPM
	mu = expected_returns.capm_return(data)
	ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 0.2))
	
	#Calculate weights to maximize sharpe ratio and portfolio stats
	raw_weights = ef.max_sharpe()
	cleaned_weights = ef.clean_weights()
	expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
	
	#display weights as table
	col1, col2 = st.columns([1, 3])
	
	with col1:
		st.caption('Portfolio Allocation: Max Sharpe Ratio')
		weights_df = pd.DataFrame.from_dict(cleaned_weights, orient = 'index')
		weights_df.columns = ['weights']
		st.dataframe(
				cleaned_weights, 
				column_config={
				"value": "Weights",
				"": "Tickers"
				})
	
	#display pie chart of portfolio weights
	with col2:
		fig_pie = px.pie(weights_df, names=weights_df.index, values=(weights_df['weights']), title="Portfolio Weights: Pie Chart")
		fig_pie.update_layout(title_text="Portfolio Weights", title_x=0.3)
		st.plotly_chart(fig_pie)
		
	#display portfolio stats as header objects
	st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
	st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
	st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
		
	#plot efficient frontier curve
	fig = plot_efficient_frontier_and_max_sharpe(mu, S)
	fig_efficient_frontier = BytesIO()
	fig.savefig(fig_efficient_frontier, format="png")
	st.caption("Efficient Frontier: Max Sharpe Ratio")
	st.image(fig_efficient_frontier)
	
	#plot cumulative returns of optimized portfolio
	fig_cum_returns = plot_cum_returns(data, 'Cumulative Returns of Individual Stocks Starting with $10K')    
	
	data['Optimized Portfolio'] = 0
	for ticker, weight in cleaned_weights.items():
		data['Optimized Portfolio'] += data[ticker]*weight
	
	fig_cum_returns_optimized = plot_cum_returns(data['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $10K')
	
	st.plotly_chart(fig_cum_returns)
	st.plotly_chart(fig_cum_returns_optimized)
				
	#plot corelation matrix
	corr_df = S.corr().round(2)
	fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between stocks')
	fig_c = st.plotly_chart(fig_corr)
	
	#Forecasting 
	
	data.reset_index(inplace=True)
	
	df_train = data[['Date', 'Optimized Portfolio']]
	df_train = df_train.rename(columns={"Date": "ds", "Optimized Portfolio": "y"})
	
	#time period
	n_years = 2030 - 2023
	period = n_years * 365
	
	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)
	
	##st.subheader('Forecast data')
	##st.write(forecast.tail())
	
	st.write('Portfolio Forecast')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)
	
	st.write('forecast components')
	fig2 = m.plot_components(forecast)
	st.write(fig2)
	
except:
	st.markdown("Add more tickers!")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.warning("IMPORTANT: The projections or other information generated regarding the likelihood of various investment outcomes are hypothetical in nature, do not reflect actual investment results and are not guarantees of future results.")
st.warning("The optimization is based on adjusted daily return statistics of the selected portfolio of (S&P 500) companies for data going back to January 1, 1980. Raw data is sourced using Yahoo Finance API and calculations are produced using PyPortfolioOpt. Results may vary with each use and over time.")
st.warning("This calculator is based specifically on Mean-Variance-Optimization and uses the Capital Asset Pricing Model (CAPM) to determine expected returns.")
st.warning("The results do not constitute investment advice or recommendation, are provided solely for informational purposes, and are not an offer to buy or sell any securities.")
st.warning("Investing involves risk, including possible loss of principal. Past performance is not a guarantee of future results.")
st.warning("Asset allocation and diversification strategies do not guarantee a profit or protect against a loss.")
st.warning("Hypothetical returns do not reflect trading costs, transaction fees, commissions, or actual taxes due on investment returns.")

