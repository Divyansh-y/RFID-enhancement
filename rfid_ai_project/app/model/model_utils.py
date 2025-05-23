# import os
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model
# import tensorflow.keras.losses as losses
# from scipy.interpolate import interp1d

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model_path = os.path.join(BASE_DIR, "lstm_model.h5")
# scaler_path = os.path.join(BASE_DIR, "scaler.save")

# # Load model once with custom loss 'mse'
# model = load_model(model_path, custom_objects={"mse": losses.MeanSquaredError()})

# # Load scaler once
# scaler = joblib.load(scaler_path)

# seq_length = 20  # same as training

# def create_sequences(data, seq_length):
#     sequences = []
#     for i in range(len(data) - seq_length):
#         sequences.append(data[i:i + seq_length])
#     return np.array(sequences)

# def enhance_rfid_signal(signal_str: str) -> str:
#     # Convert input string to float numpy array
#     signal = np.array([float(x) for x in signal_str.split(",")])

#     # Handle NaNs with linear interpolation
#     if np.isnan(signal).any():
#         x = np.arange(len(signal))
#         mask = ~np.isnan(signal)
#         f = interp1d(x[mask], signal[mask], kind='linear', fill_value="extrapolate")
#         signal = f(x)

#     # Scale signal (assumes scaler is already fitted and imported)
#     signal_scaled = scaler.transform(signal.reshape(-1, 1)).flatten()

#     # Create sequences (assumes create_sequences function and seq_length are defined)
#     sequences = create_sequences(signal_scaled, seq_length)

#     # Check if sequences is empty or not 2D, handle errors gracefully
#     if sequences.ndim != 2 or sequences.shape[0] == 0:
#         raise ValueError("Sequences could not be created from the input signal. Check input length and seq_length.")

#     # Reshape sequences to 3D for LSTM input: (samples, timesteps, features)
#     sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))

#     # Predict enhanced signal (assumes model is loaded and ready)
#     preds = model.predict(sequences)
#     preds = preds.flatten()

#     # Inverse scale back to original scale
#     preds_actual = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

#     # Return result as CSV string
#     return ",".join(map(str, preds_actual))
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import tensorflow as tf
import joblib
import os

# Load model and scaler when module is imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.save")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
seq_length = 3  # Make sure this matches your training setup

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

def enhance_rfid_signal(signal_str: str) -> str:
    try:
        signal = np.array([float(x) for x in signal_str.split(",")])
    except Exception as e:
        raise ValueError(f"Error converting input signal to float array: {e}")

    if len(signal) < seq_length:
        raise ValueError(f"Input signal length ({len(signal)}) is less than required seq_length ({seq_length}).")

    if np.isnan(signal).any():
        x = np.arange(len(signal))
        mask = ~np.isnan(signal)
        if mask.sum() < 2:
            raise ValueError("Not enough non-NaN points to interpolate.")
        f = interp1d(x[mask], signal[mask], kind='linear', fill_value="extrapolate")
        signal = f(x)

    signal_scaled = scaler.transform(signal.reshape(-1, 1)).flatten()

    sequences = create_sequences(signal_scaled, seq_length)
    if len(sequences) == 0 or sequences.ndim != 2:
        raise ValueError("Sequences could not be created from the input signal.")

    sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    preds = model.predict(sequences)
    preds = preds.flatten()

    preds_actual = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    return ",".join(map(str, preds_actual))

