# Portfolio Prophet 🚀💹

Portfolio Prophet is an AI-powered stock portfolio dashboard that fetches historical stock data, predicts future prices using AI, and provides actionable insights for multiple stocks at once. It is designed for portfolio analysis, trend visualization, and stock comparison.

## Features

- 📊 Multi-Stock Support: Forecast multiple tickers simultaneously.
- 🤖 AI-Powered Forecasting: Uses GPT-4 to predict future stock prices.
- 📈 Historical & Forecast Charts: Compare historical trends with predicted prices.
- 💰 Expected Gain/Loss Calculation: Calculates potential % change based on forecasts.
- 🏆 Portfolio Ranking: Sorts stocks by expected performance and highlights gains/losses in green/red.
- 🔧 Custom Forecast Horizon: Choose forecast period (1–30 days).
- 🌐 Interactive Dashboard: Built with Streamlit for easy use and visualization.

## Usage

1. Run the Streamlit app.
2. Enter one or multiple stock tickers separated by commas (e.g., AAPL, TSLA, AMZN).
3. Set the forecast horizon in days.
4. Click "Run Portfolio Forecast".
5. View:
   - Historical prices for each stock
   - AI-predicted future prices
   - Expected % change for each stock
   - Combined chart comparing all forecasts

## How It Works

- Fetches historical stock data from Yahoo Finance.
- Feeds recent prices to GPT-4 for AI forecasting.
- Calculates average predicted price and expected % change.
- Displays ranked portfolio table and forecast charts.

## Dependencies

- Python 3.10+
- Streamlit
- yfinance
- OpenAI API
- Pandas, Matplotlib, Seaborn
