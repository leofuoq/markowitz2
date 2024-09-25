import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Thiết lập cấu hình cho Streamlit
st.set_page_config(page_title="Interactive Portfolio Optimization", layout="wide")
st.title("Interactive Portfolio Optimization Dashboard")
st.markdown("""
This application provides an interactive way to optimize a stock portfolio using the Markowitz Model. Explore different portfolio configurations and view detailed financial metrics.
""")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
tickers_input = st.sidebar.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOG, AMZN):", "AAPL, MSFT, TSLA, GOOG, AMZN")
tickers = [ticker.strip() for ticker in tickers_input.split(",")]

# Date inputs
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Constraints inputs
risk_tolerance = st.sidebar.slider("Risk Tolerance (0: Low, 1: High)", 0.0, 1.0, 0.5)
min_weight = st.sidebar.slider("Minimum Allocation per Stock", 0.0, 0.5, 0.0)
max_weight = st.sidebar.slider("Maximum Allocation per Stock", 0.5, 1.0, 1.0)

# Lấy dữ liệu từ yfinance
@st.cache_data
def get_data(tickers, start, end):
data = yf.download(tickers, start=start, end=end)["Adj Close"]
return data

if len(tickers) > 0:
data = get_data(tickers, start_date, end_date)
daily_returns = data.pct_change().dropna()
st.subheader("Stock Price Data")
st.line_chart(data)

# Mô hình tối ưu hóa
def portfolio_statistics(weights, returns):
port_return = np.sum(weights * returns.mean()) * 252
port_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
port_volatility = np.sqrt(port_variance)
return port_return, port_volatility, port_return / port_volatility

def minimize_risk(weights, returns):
_, port_volatility, _ = portfolio_statistics(weights, returns)
return port_volatility

def negative_sharpe_ratio(weights, returns, risk_free_rate=0.01):
port_return, port_volatility, sharpe_ratio = portfolio_statistics(weights, returns)
return -(port_return - risk_free_rate) / port_volatility

def optimize_portfolio(returns, min_weight, max_weight, risk_tolerance):
num_assets = len(returns.columns)
args = (returns,)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Tổng trọng số phải bằng 1
bounds = tuple((min_weight, max_weight) for _ in range(num_assets))

# Tối ưu hóa dựa trên risk_tolerance
if risk_tolerance < 0.5:
    result = minimize(minimize_risk, num_assets * [1. / num_assets,], args=args, 
                      method='SLSQP', bounds=bounds, constraints=constraints)
else:
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args, 
                      method='SLSQP', bounds=bounds, constraints=constraints)
return result

# Tính toán và vẽ Markowitz Bullet
def calculate_portfolios(returns, num_portfolios=10000, risk_free_rate=0.01):
results = np.zeros((4, num_portfolios))
weights_record = []
num_assets = len(returns.columns)
for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    # Tính lợi nhuận và rủi ro của danh mục
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Tỷ lệ Sharpe
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
    
    # Lưu kết quả
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = sharpe_ratio
    results[3, i] = weights.max()  # Trọng số lớn nhất trong danh mục
    weights_record.append(weights)  # Lưu lại trọng số của danh mục
    
return results, weights_record

if len(tickers) > 0 and st.sidebar.button("Show Interactive Markowitz Bullet"):
portfolio_results, weights_record = calculate_portfolios(daily_returns)
max_sharpe_idx = np.argmax(portfolio_results[2])
max_sharpe_allocation = portfolio_results[:, max_sharpe_idx]

# Tạo DataFrame chứa thông tin các danh mục đầu tư
portfolios_df = pd.DataFrame(weights_record, columns=daily_returns.columns)
portfolios_df['Return'] = portfolio_results[0]
portfolios_df['Volatility'] = portfolio_results[1]
portfolios_df['Sharpe Ratio'] = portfolio_results[2]

# Vẽ biểu đồ Markowitz Bullet với Plotly
st.subheader("Interactive Markowitz Bullet")
fig = go.Figure()

# Thêm các điểm của danh mục vào biểu đồ
fig.add_trace(go.Scatter(
    x=portfolios_df['Volatility'],
    y=portfolios_df['Return'],
    mode='markers',
    marker=dict(color=portfolios_df['Sharpe Ratio'], colorscale='Viridis', size=5, showscale=True),
    text=portfolios_df.apply(lambda row: f"Weights: {dict(row[:len(tickers)])}", axis=1),
    hoverinfo='text'
))

# Thêm điểm của danh mục có Sharpe Ratio cao nhất
fig.add_trace(go.Scatter(
    x=[max_sharpe_allocation[1]],
    y=[max_sharpe_allocation[0]],
    mode='markers',
    marker=dict(color='red', size=10, symbol='star'),
    name='Max Sharpe Ratio',
    text=f"Max Sharpe Ratio Portfolio\nReturn: {max_sharpe_allocation[0]:.2%}\nVolatility: {max_sharpe_allocation[1]:.2%}",
    hoverinfo='text'
))

# Cấu hình cho biểu đồ
fig.update_layout(
    title='Markowitz Bullet',
    xaxis=dict(title='Volatility (Risk)'),
    yaxis=dict(title='Return'),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Tối ưu hóa danh mục đầu tư
if len(tickers) > 0 and st.sidebar.button("Optimize Portfolio"):
result = optimize_portfolio(daily_returns, min_weight, max_weight, risk_tolerance)
optimal_weights = result['x']
p_return, p_volatility, sharpe_ratio = portfolio_statistics(optimal_weights, daily_returns)

st.subheader("Optimized Portfolio")
st.write(f"Expected annual return: {p_return:.2%}")
st.write(f"Expected annual volatility: {p_volatility:.2%}")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

allocation = pd.DataFrame({
    'Stock': daily_returns.columns,
    'Optimal Allocation (%)': [f'{weight:.2%}' for weight in optimal_weights]
})
st.write(allocation)

# Biểu đồ phân bổ danh mục đầu tư
fig, ax = plt.subplots()
ax.pie(optimal_weights, labels=daily_returns.columns, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Hiển thị phân tích thêm
st.subheader("Detailed Analysis")
st.write("Correlation Matrix of Stock Returns")
st.dataframe(daily_returns.corr())

# Thông tin chi tiết về danh mục đầu tư tối ưu
st.subheader("Additional Portfolio Metrics")

# Tỷ suất sinh lợi hàng tháng
monthly_return = (np.sum(optimal_weights * daily_returns.mean()) * 21).round(4)
st.write("**Monthly Return:**", monthly_return)

# Tỷ suất sinh lợi hàng năm
annual_return = (np.sum(optimal_weights * daily_returns.mean()) * 252).round(4)
st.write("**Annualized Return:**", annual_return)

# Độ lệch chuẩn hàng năm
annual_volatility = (np.sqrt(np.dot(optimal_weights.T, np.dot(daily_returns.cov() * 252, optimal_weights)))).round(4)
st.write("**Annualized Volatility:**", annual_volatility)

# Sharpe Ratio
sharpe_ratio = (annual_return - 0.01) / annual_volatility

# Sharpe Ratio tiếp tục
st.write("**Sharpe Ratio:**", sharpe_ratio.round(4))

# Tỷ suất sinh lợi kỳ vọng hàng ngày
expected_daily_return = (np.sum(optimal_weights * daily_returns.mean())).round(4)
st.write("**Expected Daily Return:**", expected_daily_return)

# Tổng hợp tỷ trọng danh mục đầu tư
st.write("**Portfolio Allocation:**")
portfolio_allocation = pd.DataFrame({
    'Stock': daily_returns.columns,
    'Allocation (%)': [f'{weight * 100:.2f}%' for weight in optimal_weights]
})
st.write(portfolio_allocation)

# Biểu đồ tỷ trọng danh mục đầu tư
fig = go.Figure(data=[go.Pie(labels=portfolio_allocation['Stock'], values=[weight for weight in optimal_weights], hole=.3)])
fig.update_layout(title_text='Portfolio Allocation')
st.plotly_chart(fig)

# Phân tích thêm về chỉ số tương quan
st.subheader("Correlation Heatmap")
correlation = daily_returns.corr()
fig = go.Figure(data=go.Heatmap(
    z=correlation.values,
    x=correlation.columns,
    y=correlation.columns,
    colorscale='Viridis'))
fig.update_layout(title='Stock Return Correlation Heatmap')
st.plotly_chart(fig)

# Kết thúc đoạn mã 
