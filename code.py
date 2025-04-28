import numpy as np
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5
from transformers import pipeline
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize MetaTrader 5
def initialize_mt5():
    if not mt5.initialize():
        logging.error("MetaTrader 5 initialization failed")
        quit()

# Load live or historical data
def get_data(symbol, timeframe, n_bars=1000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    data = pd.DataFrame(rates)
    data["time"] = pd.to_datetime(data["time"], unit="s")
    return data

# Data Preparation Function
def prepare_lstm_data(data, sequence_length=30):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# PyTorch Models for Ensemble
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last hidden state
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take the last hidden state
        return out

# Train ensemble models
def train_ensemble(data):
    X, y = prepare_lstm_data(data["close"].values)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32)

    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    epochs = 10
    batch_size = 32
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
    gru_model = GRUModel(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        lstm_model.train()
        gru_model.train()
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size].to(device)
            y_batch = y[i : i + batch_size].to(device)

            # LSTM Training
            lstm_optimizer.zero_grad()
            lstm_outputs = lstm_model(X_batch)
            lstm_loss = criterion(lstm_outputs.squeeze(), y_batch)
            lstm_loss.backward()
            lstm_optimizer.step()

            # GRU Training
            gru_optimizer.zero_grad()
            gru_outputs = gru_model(X_batch)
            gru_loss = criterion(gru_outputs.squeeze(), y_batch)
            gru_loss.backward()
            gru_optimizer.step()

        logging.info(f"Epoch {epoch + 1}/{epochs}, LSTM Loss: {lstm_loss.item()}, GRU Loss: {gru_loss.item()}")

    return lstm_model, gru_model

# Perform sentiment analysis on news headlines
def analyze_sentiment(news):
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    sentiments = sentiment_pipeline(news)
    scores = [s["score"] if s["label"] == "POSITIVE" else -s["score"] for s in sentiments]
    return sum(scores) / len(scores)

# Predict ensemble output
def predict_ensemble(models, data):
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    predictions = [model(data).item() for model in models]
    return sum(predictions) / len(predictions)

# Detect AMD phases (Accumulation, Markup, Distribution)
def detect_amd(data):
    kmeans = KMeans(n_clusters=3)
    data["phase"] = kmeans.fit_predict(data[["close"]])
    return data

# Calculate position size
def calculate_position_size(account_balance, risk_percent, pip_risk, pip_value):
    risk_amount = account_balance * (risk_percent / 100)
    position_size = risk_amount / (pip_risk * pip_value)
    return position_size

# Place Orders
def place_order(symbol, action, volume, price, sl, tp):
    order_type = mt5.ORDER_BUY if action == "BUY" else mt5.ORDER_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": f"{action} order by bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade failed: {result.comment}")
    else:
        logging.info(f"Trade successful: {action} {volume} lots at {price}")

# Trading Bot Execution
def trading_bot():
    initialize_mt5()
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M15
    account_balance = 10000  # Example balance
    risk_percent = 1
    pip_value = 10  # Example pip value

    # Load Historical Data
    data = get_data(symbol, timeframe, 1000)

    # Train Ensemble Models
    lstm_model, gru_model = train_ensemble(data)

    while True:
        # Fetch Live Data
        live_data = get_data(symbol, timeframe, 200)

        # Predict Prices
        latest_data = live_data["close"].values[-30:]
        predicted_price = predict_ensemble([lstm_model, gru_model], latest_data)

        # Detect AMD Phases
        live_data = detect_amd(live_data)
        latest_phase = live_data["phase"].iloc[-1]

        # Sentiment Analysis
        news = ["Gold prices rise amidst geopolitical tension"]  # Replace with live news
        sentiment_score = analyze_sentiment(news)

        # Decision Logic
        if latest_phase == 0 and sentiment_score > 0.5:
            price = live_data["close"].iloc[-1]
            stop_loss = price - 50  # Example SL
            take_profit = price + 100  # Example TP
            volume = calculate_position_size(account_balance, risk_percent, 50, pip_value)
            place_order(symbol, "BUY", volume, price, stop_loss, take_profit)
        elif latest_phase == 2 and sentiment_score < -0.5:
            price = live_data["close"].iloc[-1]
            stop_loss = price + 50  # Example SL
            take_profit = price - 100  # Example TP
            volume = calculate_position_size(account_balance, risk_percent, 50, pip_value)
            place_order(symbol, "SELL", volume, price, stop_loss, take_profit)

        mt5.sleep(60)

# Run the Bot
if __name__ == "__main__":
    trading_bot()
