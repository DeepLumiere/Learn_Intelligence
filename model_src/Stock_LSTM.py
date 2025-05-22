import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class StockLSTMDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X_sequences = []
        self.y_targets = []

        for i in range(len(X) - seq_len):
            self.X_sequences.append(X[i:i + seq_len])
            self.y_targets.append(y[i + seq_len])

        if not self.X_sequences:
            feature_dim = X.shape[1] if X.ndim > 1 else 1
            self.X_tensor = torch.empty(0, seq_len, feature_dim, dtype=torch.float32)
            self.y_tensor = torch.empty(0, 1, dtype=torch.float32)
        else:
            self.X_tensor = torch.tensor(np.array(self.X_sequences), dtype=torch.float32)
            self.y_tensor = torch.tensor(np.array(self.y_targets), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.2):  # Added dropout
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# 1. Fetch more historical data
def fetch_stock_data(ticker, n_years=5):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=n_years)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols]
    return df


def preprocess_data(df, n_lags=5):
    df_processed = df.copy()
    df_processed.set_index('Date', inplace=True)

    # Date features
    df_processed['Year'] = df_processed.index.year
    df_processed['Month_sin'] = np.sin(2 * np.pi * df_processed.index.month / 12)
    df_processed['Month_cos'] = np.cos(2 * np.pi * df_processed.index.month / 12)
    df_processed['Day_sin'] = np.sin(2 * np.pi * df_processed.index.day / 31)
    df_processed['Day_cos'] = np.cos(2 * np.pi * df_processed.index.day / 31)
    df_processed['DayOfWeek_sin'] = np.sin(2 * np.pi * df_processed.index.dayofweek / 7)
    df_processed['DayOfWeek_cos'] = np.cos(2 * np.pi * df_processed.index.dayofweek / 7)

    cols_to_lag = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_lag:
        if col in df_processed.columns:
            for lag in range(1, n_lags + 1):
                df_processed[f'{col}_lag_{lag}'] = df_processed[col].shift(lag)

    y_high = df_processed['High'].copy()
    y_low = df_processed['Low'].copy()
    y_close = df_processed['Close'].copy()

    feature_cols = [col for col in df_processed.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    X_df = df_processed[feature_cols].copy()

    combined_df = pd.concat([X_df, y_high, y_low, y_close], axis=1)
    combined_df.dropna(inplace=True)

    X_df = combined_df[X_df.columns].copy()
    y_high = combined_df['High'].copy()
    y_low = combined_df['Low'].copy()
    y_close = combined_df['Close'].copy()

    X_df.reset_index(inplace=True)  # 'Date' becomes a column

    return X_df, y_high, y_low, y_close


def train_and_evaluate_model(X_df_full,
                             y_series,
                             target_name,
                             X_scaler,
                             seq_len=10, epochs=50, batch_size=32, lr=0.001, patience=5):  # Added patience

    X_numerical = X_df_full.drop(columns=['Date']).values
    y_values = y_series.values.reshape(-1, 1)

    num_samples = len(X_numerical)
    if num_samples < seq_len + 2:
        print(
            f"Warning: Not enough data for {target_name} to form sequences with seq_len={seq_len}. Skipping training.")
        return None, None

    train_size = int(num_samples * 0.8)

    if train_size < seq_len + 1 or (num_samples - train_size) < seq_len + 1:
        print(f"Warning: Data split for {target_name} too small for seq_len={seq_len}. Skipping training.")
        return None, None

    X_train_raw = X_numerical[:train_size]
    X_test_raw = X_numerical[train_size:]
    y_train_raw = y_values[:train_size]
    y_test_raw = y_values[train_size:]

    X_train_scaled = X_scaler.transform(X_train_raw)
    X_test_scaled = X_scaler.transform(X_test_raw)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    y_test_scaled = y_scaler.transform(y_test_raw)

    train_dataset = StockLSTMDataset(X_train_scaled, y_train_scaled.flatten(), seq_len)
    test_dataset = StockLSTMDataset(X_test_scaled, y_test_scaled.flatten(), seq_len)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(
            f"Warning: Dataset for {target_name} is too small with current seq_len={seq_len} after split. Skipping training for {target_name}.")
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X_train_scaled.shape[1]
    model = LSTMModel(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    min_val_loss = np.inf
    epochs_o_improve = 0
    best_model_path = f"../models/{ticker}_{target_name}_best_model.pth"

    print(f"\nTraining model for {target_name}...")
    for epoch in tqdm(range(epochs), desc=f"Training {target_name}"):
        model.train()
        train_epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in test_loader:
                output_val = model(batch_X_val)
                val_loss = criterion(output_val, batch_y_val)
                val_epoch_loss += val_loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_val_loss = val_epoch_loss / len(test_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered for {target_name} at epoch {epoch + 1}. Loading best model.")
                model.load_state_dict(torch.load(best_model_path))
                break

    if epochs_no_improve < patience:
        try:
            model.load_state_dict(torch.load(best_model_path))
        except FileNotFoundError:
            print(f"Warning: Best model for {target_name} not found, using last epoch's model.")

    model.eval()
    y_true_list, y_pred_list = [], []
    # actual_prev_day_closes = []

    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(test_loader):
            output = model(batch_X)
            y_pred_list.extend(output.cpu().numpy())
            y_true_list.extend(batch_y.cpu().numpy())


    y_true_scaled_np = np.array(y_true_list)
    y_pred_scaled_np = np.array(y_pred_list)

    y_true_unscaled = y_scaler.inverse_transform(y_true_scaled_np)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled_np)

    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
    y_true_positive_mask = y_true_unscaled > 1e-6
    mape = mean_absolute_percentage_error(y_true_unscaled[y_true_positive_mask],
                                          y_pred_unscaled[y_true_positive_mask]) if np.sum(
        y_true_positive_mask) > 0 else float('nan')

    print(f"\nMetrics for {target_name} (unscaled test data):")
    print(f"  R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

    directional_accuracy = float('nan')
    if target_name == "Close":
        actual_prev_closes_for_test = y_test_raw[seq_len - 1: len(y_test_raw) - 1].flatten()

        # Ensure lengths match for comparison
        if len(y_pred_unscaled) != len(actual_prev_closes_for_test) or \
                len(y_true_unscaled) != len(actual_prev_closes_for_test):
            print(f"Warning: Length mismatch for directional accuracy calculation for {target_name}. "
                  f"Pred: {len(y_pred_unscaled)}, True: {len(y_true_unscaled)}, Prev Actual: {len(actual_prev_closes_for_test)}")
            directional_accuracy = float('nan')
        else:
            predicted_directions = (y_pred_unscaled > actual_prev_closes_for_test).astype(int)
            actual_directions = (y_true_unscaled > actual_prev_closes_for_test).astype(int)

            correct_predictions = np.sum(predicted_directions == actual_directions)
            directional_accuracy = correct_predictions / len(predicted_directions) * 100

        print(f"  Directional Accuracy: {directional_accuracy:.2f}%")

    return model, y_scaler


def predict_n_days_autoregressive(
        models,
        scalers,
        X_historical_df,
        historical_ohlcv_df,
        n_days_to_predict,
        seq_len,
        n_lags):
    print("\nStarting autoregressive prediction...")

    df_for_prediction = X_historical_df.copy()

    future_predictions = {'Date': []}
    for target_name in models.keys():
        future_predictions[target_name] = []

    last_known_volume_lags = {}
    for lag in range(1, n_lags + 1):
        col_name = f'Volume_lag_{lag}'
        if col_name in df_for_prediction.columns:
            last_known_volume_lags[col_name] = df_for_prediction[col_name].iloc[-1]
        else:
            last_known_volume_lags[col_name] = 0.0

    last_actual_close_for_open_approx = historical_ohlcv_df['Close'].iloc[-1]
    last_actual_high = historical_ohlcv_df['High'].iloc[-1]
    last_actual_low = historical_ohlcv_df['Low'].iloc[-1]
    last_actual_close = historical_ohlcv_df['Close'].iloc[-1]

    for day_step in tqdm(range(n_days_to_predict), desc="Predicting Future Days"):
        last_date_in_features_df = pd.to_datetime(df_for_prediction['Date'].iloc[-1])
        next_pred_date = last_date_in_features_df + pd.Timedelta(days=1)
        future_predictions['Date'].append(next_pred_date)

        current_sequence_unscaled_df = df_for_prediction.iloc[-seq_len:].drop(columns=['Date'])
        current_sequence_numerical_unscaled = current_sequence_unscaled_df.values

        if len(current_sequence_numerical_unscaled) < seq_len:
            print(
                f"Error: Not enough historical data to form a sequence of length {seq_len} for prediction step {day_step}.")
            break

        current_sequence_scaled = scalers['X'].transform(current_sequence_numerical_unscaled)
        input_tensor = torch.tensor(current_sequence_scaled, dtype=torch.float32).unsqueeze(0)

        predicted_values_unscaled_today = {}
        for target_name, model in models.items():
            model.eval()
            with torch.no_grad():
                pred_scaled = model(input_tensor)
            pred_unscaled = scalers[f'{target_name}_y'].inverse_transform(pred_scaled.cpu().numpy())[0, 0]
            predicted_values_unscaled_today[target_name] = pred_unscaled
            future_predictions[target_name].append(pred_unscaled)

        new_feature_row_dict = {}

        new_feature_row_dict['Year'] = next_pred_date.year
        new_feature_row_dict['Month_sin'] = np.sin(2 * np.pi * next_pred_date.month / 12)
        new_feature_row_dict['Month_cos'] = np.cos(2 * np.pi * next_pred_date.month / 12)
        new_feature_row_dict['Day_sin'] = np.sin(2 * np.pi * next_pred_date.day / 31)
        new_feature_row_dict['Day_cos'] = np.cos(2 * np.pi * next_pred_date.day / 31)
        new_feature_row_dict['DayOfWeek_sin'] = np.sin(2 * np.pi * next_pred_date.dayofweek / 7)
        new_feature_row_dict['DayOfWeek_cos'] = np.cos(2 * np.pi * next_pred_date.dayofweek / 7)

        if day_step == 0:
            new_feature_row_dict['Open_lag_1'] = last_actual_close_for_open_approx  # Common approximation
            new_feature_row_dict['High_lag_1'] = last_actual_high
            new_feature_row_dict['Low_lag_1'] = last_actual_low
            new_feature_row_dict['Close_lag_1'] = last_actual_close
        else:
            new_feature_row_dict['Open_lag_1'] = future_predictions['close'][-1]  # Close of previous predicted day
            new_feature_row_dict['High_lag_1'] = future_predictions['high'][-1]  # High of previous predicted day
            new_feature_row_dict['Low_lag_1'] = future_predictions['low'][-1]  # Low of previous predicted day
            new_feature_row_dict['Close_lag_1'] = future_predictions['close'][-1]  # Close of previous predicted day

        for lag in range(2, n_lags + 1):
            for col_base in ['Open', 'High', 'Low', 'Close']:
                prev_lag_col = f'{col_base}_lag_{lag - 1}'
                if prev_lag_col in df_for_prediction.columns:
                    new_feature_row_dict[f'{col_base}_lag_{lag}'] = df_for_prediction[prev_lag_col].iloc[-1]
                else:
                    new_feature_row_dict[f'{col_base}_lag_{lag}'] = 0.0

        for lag in range(1, n_lags + 1):
            new_feature_row_dict[f'Volume_lag_{lag}'] = last_known_volume_lags[f'Volume_lag_{lag}']

        feature_columns_order = [col for col in X_historical_df.columns if col != 'Date']
        new_row_values = [new_feature_row_dict.get(col, 0.0) for col in feature_columns_order]  # Fill missing with 0.0

        new_row_df_features = pd.DataFrame([new_row_values], columns=feature_columns_order)
        new_row_df_features['Date'] = next_pred_date  # Add date back

        new_row_df_features = new_row_df_features[df_for_prediction.columns]

        df_for_prediction = pd.concat([df_for_prediction, new_row_df_features], ignore_index=True)

    return pd.DataFrame(future_predictions)


if __name__ == '__main__':
    ticker = "MSFT"
    n_future_days = 100

    # Config
    config_seq_len = 25
    config_n_lags = 10
    config_epochs = 15
    config_batch_size = 32
    config_lr = 0.5
    config_patience = 10

    # 1. Fetch Data
    print(f"Fetching data for {ticker}...")
    historical_ohlcv_df = fetch_stock_data(ticker, n_years=5)
    print(f"Fetched {len(historical_ohlcv_df)} days of data.")

    if historical_ohlcv_df.empty:
        print("No historical data fetched. Exiting.")
        exit()

    # 2. Preprocess Data
    print("Preprocessing data...")
    X_df_processed, y_high_series, y_low_series, y_close_series = preprocess_data(historical_ohlcv_df.copy(), n_lags=config_n_lags)
    print(f"Processed X_df shape: {X_df_processed.shape}")

    min_samples_needed = config_seq_len + 1
    if len(X_df_processed) < min_samples_needed + (
            min_samples_needed * 0.2):
        print(
            f"Not enough data after preprocessing and lagging. Need at least {min_samples_needed + (min_samples_needed * 0.2)} samples. Have {len(X_df_processed)}. Exiting.")
        exit()

    X_numerical_df = X_df_processed.drop(columns=['Date'])

    total_samples = len(X_numerical_df)
    train_size_for_scaler = int(total_samples * 0.8)

    if train_size_for_scaler < config_seq_len + 1:
        print("Not enough training data to fit X_scaler or form sequences. Exiting.")
        exit()

    X_train_num_for_scaler = X_numerical_df.iloc[:train_size_for_scaler].values

    if len(X_train_num_for_scaler) == 0:
        print("Not enough training data to fit X_scaler. Exiting.")
        exit()

    X_feature_scaler = StandardScaler().fit(X_train_num_for_scaler)

    models_dict = {}
    scalers_dict = {'X': X_feature_scaler}

    targets_to_train = {
        "High": y_high_series,
        "Low": y_low_series,
        "Close": y_close_series
    }

    training_successful = True
    for target_name, y_target_series in targets_to_train.items():
        model, y_target_scaler = train_and_evaluate_model(
            X_df_processed.copy(),
            y_target_series,
            target_name,
            X_feature_scaler,
            seq_len=config_seq_len,
            epochs=config_epochs,
            batch_size=config_batch_size,
            lr=config_lr,
            patience=config_patience
        )
        if model is None or y_target_scaler is None:
            print(f"Failed to train model for {target_name}. Predictions will not be complete.")
            training_successful = False
            break
        models_dict[target_name.lower()] = model
        scalers_dict[f'{target_name.lower()}_y'] = y_target_scaler

    if not training_successful:
        print("Exiting due to training failure.")
        exit()

    future_df = predict_n_days_autoregressive(
        models_dict,
        scalers_dict,
        X_df_processed.copy(),
        historical_ohlcv_df.copy(),
        n_days_to_predict=n_future_days,
        seq_len=config_seq_len,
        n_lags=config_n_lags
    )
    print("\nFuture Predictions:")
    print(future_df.head())

    plt.figure(figsize=(15, 8))

    historical_plot_dates = pd.to_datetime(historical_ohlcv_df['Date'])
    plt.plot(historical_plot_dates, historical_ohlcv_df['Close'], label="Actual Historical Close", color='black')

    future_plot_dates = pd.to_datetime(future_df['Date'])
    plt.plot(future_plot_dates, future_df['high'], label="Predicted Future High", color='green', linestyle='--')
    plt.plot(future_plot_dates, future_df['low'], label="Predicted Future Low", color='red', linestyle='--')
    plt.plot(future_plot_dates, future_df['close'], label="Predicted Future Close", color='blue', linestyle='--')

    if not future_plot_dates.empty and not historical_plot_dates.empty:
        last_actual_date = historical_plot_dates.iloc[-1]
        last_actual_close_val = historical_ohlcv_df['Close'].iloc[-1]
        last_actual_high_val = historical_ohlcv_df['High'].iloc[-1]
        last_actual_low_val = historical_ohlcv_df['Low'].iloc[-1]

        plt.plot(last_actual_date, last_actual_close_val, 'o', color='black', markersize=5, label='Last Actual Close')
        # Mark start of predicted values
        plt.plot(future_plot_dates.iloc[0], future_df['close'].iloc[0], 'X', color='blue', markersize=7,
                 label='Start Predicted Close')
        plt.plot(future_plot_dates.iloc[0], future_df['high'].iloc[0], 'X', color='green', markersize=7,
                 label='Start Predicted High')
        plt.plot(future_plot_dates.iloc[0], future_df['low'].iloc[0], 'X', color='red', markersize=7,
                 label='Start Predicted Low')

        # Connect last actual point to first predicted point
        plt.plot([last_actual_date, future_plot_dates.iloc[0]], [last_actual_close_val, future_df['close'].iloc[0]],
                 color='blue', linestyle=':', linewidth=1)
        plt.plot([last_actual_date, future_plot_dates.iloc[0]], [last_actual_high_val, future_df['high'].iloc[0]],
                 color='green', linestyle=':', linewidth=1)
        plt.plot([last_actual_date, future_plot_dates.iloc[0]], [last_actual_low_val, future_df['low'].iloc[0]],
                 color='red', linestyle=':', linewidth=1)

    plt.title(f"{ticker} Stock Price: Actual Close & Predicted HLC for Next {n_future_days} Days (Autoregressive)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../graphs/{ticker}_Historical_plus_Predicted.png")
    plt.show()

    if not future_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(future_plot_dates, future_df['high'], label="Predicted Future High", color='green', linestyle='--')
        plt.plot(future_plot_dates, future_df['low'], label="Predicted Future Low", color='red', linestyle='--')
        plt.plot(future_plot_dates, future_df['close'], label="Predicted Future Close", color='blue', linestyle='--')

        # Connect to last actual points smoothly for the zoomed plot as well
        if not historical_ohlcv_df.empty:
            last_actual_high = historical_ohlcv_df['High'].iloc[-1]
            last_actual_low = historical_ohlcv_df['Low'].iloc[-1]
            last_actual_close = historical_ohlcv_df['Close'].iloc[-1]
            last_actual_date = pd.to_datetime(historical_ohlcv_df['Date'].iloc[-1])

            plt.plot([last_actual_date, future_plot_dates.iloc[0]], [last_actual_high, future_df['high'].iloc[0]],
                     color='green', linestyle=':')
            plt.plot([last_actual_date, future_plot_dates.iloc[0]], [last_actual_low, future_df['low'].iloc[0]],
                     color='red', linestyle=':')
            plt.plot([last_actual_date, future_plot_dates.iloc[0]], [last_actual_close, future_df['close'].iloc[0]],
                     color='blue', linestyle=':')

            plt.plot(last_actual_date, last_actual_high, 'o', color='darkgreen', label='Last Actual High')
            plt.plot(last_actual_date, last_actual_low, 'o', color='darkred', label='Last Actual Low')
            plt.plot(last_actual_date, last_actual_close, 'o', color='darkblue', label='Last Actual Close')

        plt.title(f"{ticker} Predicted HLC for Next {n_future_days} Days (Zoomed)")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"../graphs/{ticker}_Predicted_Zoomed.png")
        plt.show()