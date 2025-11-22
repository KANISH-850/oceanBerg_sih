import streamlit as st
import numpy as np
from ml.train_cnn_lstm_fault import train_fault_cnn_lstm

st.title("Fault Classification – CNN + LSTM Trainer")

if st.button("Train Model"):
    st.write("Loading dataset...")

    # Load waveform data
    X = np.load("data/X_waveforms.npy")   # shape (N, T, C)
    y = np.load("data/y_waveforms.npy")   # labels

    st.write(f"Dataset loaded: X={X.shape}, y={len(y)}")

    with st.spinner("Training CNN + LSTM model..."):
        model, encoder, accuracy, history = train_fault_cnn_lstm(X, y)

    st.success(f"Training complete! Test Accuracy = {accuracy:.4f}")
    st.write("Model saved → models/cnn_lstm_fault.h5")
