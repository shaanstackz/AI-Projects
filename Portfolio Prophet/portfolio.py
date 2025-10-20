import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os

sns.set_style("whitegrid")

openai.api_key = os.getenv("OPENAI_API_KEY") 

def fetch_stock_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)['Close'].reset_index()
    df.rename(columns={'Close':'Price'}, inplace=True)
    return df

def forecast_with_gpt(prices, ticker, days=5):
    prices_str = ", ".join([str(round(p,2)) for p in prices])
    prompt = (
        f"You are a financial analyst. Here are the last {len(prices)} closing prices "
        f"for {ticker}: {prices_str}. "
        f"Predict the next {days} days of closing prices in USD. "
        f"Give only numbers in a comma-separated list."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    forecast_text = response['choices'][0]['message']['content']
    try:
        forecast_prices = [float(p.strip()) for p in forecast_text.split(",")]
        return forecast_prices
    except:
        st.error(f"Error parsing GPT output for {ticker}. Raw output: {forecast_text}")
        return []

def calculate_expected_change(df, forecast_prices):
    last_price = df['Price'].iloc[-1]
    forecast_avg = sum(forecast_prices)/len(forecast_prices)
    percent_change = ((forecast_avg - last_price)/last_price) * 100
    return round(percent_change, 2)

def plot_forecasts(stock_data_dict, forecast_dict):
    plt.figure(figsize=(14,7))
    for ticker, df in stock_data_dict.items():
        plt.plot(df['Date'], df['Price'], label=f"{ticker} Historical")
        if ticker in forecast_dict and forecast_dict[ticker]:
            last_date = df['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_dict[ticker]))
            plt.plot(future_dates, forecast_dict[ticker], marker='o', label=f"{ticker} Forecast")
    plt.title("Stock Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price USD")
    plt.legend()
    st.pyplot(plt.gcf())

st.title("💹 AI Stock Portfolio Dashboard")
st.write("Enter multiple stock tickers and see AI forecasts, expected gains/losses, and rankings.")

tickers_input = st.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, TSLA, AMZN)").upper()
forecast_days = st.number_input("Forecast Horizon (days)", min_value=1, max_value=30, value=5)

if st.button("Run Portfolio Forecast") and tickers_input:
    tickers = [t.strip() for t in tickers_input.split(",")]
    stock_data_dict = {}
    forecast_dict = {}
    portfolio_table = []

    for ticker in tickers:
        with st.spinner(f"Processing {ticker}..."):
            try:
                df = fetch_stock_data(ticker, period="1y")
                stock_data_dict[ticker] = df
                recent_prices = df['Price'].tolist()[-60:]
                forecast_prices = forecast_with_gpt(recent_prices, ticker, days=forecast_days)
                forecast_dict[ticker] = forecast_prices
                
                if forecast_prices:
                    expected_change = calculate_expected_change(df, forecast_prices)
                    portfolio_table.append({
                        "Ticker": ticker,
                        "Last Price": df['Price'].iloc[-1],
                        f"Forecast Avg ({forecast_days}d)": round(sum(forecast_prices)/len(forecast_prices),2),
                        "Expected % Change": expected_change
                    })
            except Exception as e:
                st.error(f"Error with {ticker}: {e}")

    if portfolio_table:
        portfolio_df = pd.DataFrame(portfolio_table).sort_values(by="Expected % Change", ascending=False)
        st.subheader("📊 Portfolio Forecast Summary")
        
        def color_change(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'
        
        st.dataframe(portfolio_df.style.applymap(color_change, subset=["Expected % Change"]))

        st.subheader("📈 Forecast Comparison Chart")
        plot_forecasts(stock_data_dict, forecast_dict)
