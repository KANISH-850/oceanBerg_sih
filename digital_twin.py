import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import sys
import json
import time
import math
import shutil
import logging
import argparse
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except Exception:
    tf = None

try:
    
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet  
    except Exception:
        Prophet = None

import joblib

CONFIG = {
    "run_name": f"master_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
    "data_csv": "data/digital_twin_unified.csv",
    "simulate_data_if_missing": True,
    "simulate_rows": 5000,  
    "out_dir": "outputs",
    "models_dir": "models",
    "logs_dir": "logs",
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,
    "batch_size": 64,
    "epochs_lstm": 10,
    "epochs_cnn_lstm": 20,
    "lstm_window": 24,
    "prophet_periods": 24*7,  # 1 week ahead hourly
    "xgb_params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
    "catboost_params": {"iterations": 500, "learning_rate": 0.05, "depth": 6},
    "tf_lr": 1e-4,
    "clipnorm": 1.0
}

# Ensure directories
os.makedirs(CONFIG["out_dir"], exist_ok=True)
os.makedirs(CONFIG["models_dir"], exist_ok=True)
os.makedirs(CONFIG["logs_dir"], exist_ok=True)

# Logging
logger = logging.getLogger("digital_twin_master")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
# console
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)
# file
fh = logging.FileHandler(os.path.join(CONFIG["logs_dir"], f"{CONFIG['run_name']}.log"))
fh.setFormatter(fmt)
logger.addHandler(fh)

np.random.seed(CONFIG["random_seed"])

# --------------------
# Utility functions
# --------------------

SKIPPED_ROWS = []

def log_skip(identifier: str, reason: str):
    SKIPPED_ROWS.append({"id": identifier, "reason": reason})
    logger.warning(f"SKIPPED: {identifier} | reason: {reason}")

def save_json(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# --------------------
# Synthetic data generator (unified CSV)
# --------------------
def simulate_unified_dataset(path: str, rows: int = 5000):
    """
    Create a unified CSV containing combined columns required by all models.
    Columns:
      - timestamp
      - transformer WTI, OTI, oil_level, vibration, load_mw
      - breaker status/opcount/sf6
      - waveform (stringified list) and fault_type
      - failure (0/1)
      - health_index (float)
    """
    import math
    logger.info(f"Simulating unified dataset with {rows} rows -> {path}")
    timestamps = pd.date_range(end=datetime.utcnow(), periods=rows, freq="H")
    rng = np.random.RandomState(CONFIG["random_seed"])

    # base load pattern + noise
    hour = np.array([t.hour for t in timestamps])
    daily_pattern = 100 + 30 * np.sin((hour / 24) * 2 * np.pi)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "WTI": 60 + 5 * np.sin(np.linspace(0, 4*np.pi, rows)) + rng.normal(0, 1.0, rows),
        "OTI": 45 + 3 * np.sin(np.linspace(0, 4*np.pi, rows)) + rng.normal(0, 0.8, rows),
        "oil_level": 80 + rng.normal(0, 1.5, rows),
        "vibration": np.abs(rng.normal(0.5, 0.2, rows)),
        "load_mw": daily_pattern + rng.normal(0, 5.0, rows),
        "CB_status": rng.choice([0,1], size=rows, p=[0.05, 0.95]),
        "CB_opcount": rng.poisson(2, size=rows),
        "CB_SF6": 7.0 + rng.normal(0, 0.05, rows),
    })

    # failure label: rare events based on spike in vibration & temp
    failure_prob = np.clip( (df["vibration"] - 1.0) + (df["WTI"] - 75)/50, 0, 1 )
    df["failure"] = rng.binomial(1, 0.01 * (1 + failure_prob))

    # health index: synthetic function (lower = worse)
    df["health_index"] = 100 - ( (df["WTI"]-40).clip(0)/2 + df["vibration"]*15 + (df["CB_opcount"]/50)*20 )

    # Fault waveforms: randomly for some rows create waveform arrays (stringified)
    def make_waveform(idx):
        t = np.linspace(0,1,256)
        base = np.sin(2*np.pi*50*t)  # 50Hz nominal
        noise = rng.normal(0, 0.05, size=t.shape)
        # occasionally inject fault signature
        if rng.rand() < 0.02:
            fault = base * (1 + rng.normal(0.5, 0.2, size=t.shape)) + rng.normal(0,0.2,size=t.shape)
            return ",".join(map(str, fault.tolist())), "fault"
        else:
            return ",".join(map(str, (base+noise).tolist())), "normal"

    wf = [make_waveform(i) for i in range(rows)]
    df["waveform"], df["fault_type"] = zip(*wf)

    # shuffle slight and write
    df = df.reset_index(drop=True)
    df.to_csv(path, index=False)
    logger.info(f"Synthetic dataset written to {path}")

# --------------------
# Robust CSV loader
# --------------------
def load_unified_csv(path: str) -> pd.DataFrame:
    """
    Robustly loads the unified CSV. If file missing and simulation enabled, simulates.
    Performs basic cleaning:
      - removes fully-null rows
      - coerces numeric columns
      - logs corrupted rows
    """
    if not os.path.isfile(path):
        if CONFIG["simulate_data_if_missing"]:
            simulate_unified_dataset(path, rows=CONFIG["simulate_rows"])
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")

    df = None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Failed to read CSV directly: {e}. Attempting fallback read with engine='python'.")
        try:
            df = pd.read_csv(path, engine="python", error_bad_lines=False)
        except Exception as e2:
            logger.error(f"Failed to read CSV fallback: {e2}")
            raise

    # Basic cleaning
    initial_rows = len(df)
    # remove all null rows
    df.dropna(how="all", inplace=True)
    # ensure timestamp parse
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        log = f"timestamp_parse_error: {e}"
        log_skip = {"id": "timestamp_column", "reason": log}
        logger.warning(log)
        # attempt to coerce
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)

    # coerce numeric columns
    numeric_cols = ["WTI","OTI","oil_level","vibration","load_mw","CB_status","CB_opcount","CB_SF6","failure","health_index"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # log and drop rows that still have NaNs in essential cols
    essential = ["WTI","OTI","oil_level","vibration","load_mw"]
    mask_bad = df[essential].isnull().any(axis=1)
    bad_indices = df[mask_bad].index.tolist()
    for idx in bad_indices:
        log_skip(f"row_{idx}", "essential_missing")
    df = df[~mask_bad].reset_index(drop=True)

    logger.info(f"Loaded dataset: {path}. Rows: {initial_rows} -> {len(df)} after dropping corrupted essential rows.")
    return df

# --------------------
# Per-model dataset extractors
# --------------------
def build_isolation_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return X (numpy) and optional y for IsolationForest training/eval."""
    features = ["WTI","OTI","oil_level","vibration"]
    X = df[features].values.astype(float)
    y = df["failure"].values.astype(int) if "failure" in df.columns else None
    return X, y

def build_xgb_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = ["WTI","OTI","oil_level","vibration","load_mw","CB_opcount","CB_SF6"]
    df2 = df.copy()
    df2 = df2.dropna(subset=features + ["failure"])
    X = df2[features].values.astype(float)
    y = df2["failure"].astype(int).values
    return X, y

def build_lstm_dataset(df: pd.DataFrame, target_col: str = "load_mw", window: int = 24):
    """Return X (n_samples, window, 1), y (n_samples,) for forecasting."""
    series = df[[ "timestamp", target_col ]].sort_values("timestamp")
    vals = series[target_col].values.astype(float)
    # Create sliding window
    X, y = [], []
    for i in range(len(vals) - window):
        X.append(vals[i:i+window])
        y.append(vals[i+window])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y

def build_waveform_dataset(df: pd.DataFrame, max_len: int = 256):
    """
    Convert the 'waveform' string column into numpy arrays.
    Expects waveform column to be comma-separated floats.
    Returns X (n_samples, timesteps, 1), y (labels as ints)
    """
    rows = []
    labels = []
    for idx, row in df.iterrows():
        try:
            wav = row.get("waveform", "")
            if pd.isna(wav) or wav == "":
                log_skip(f"wave_{idx}", "waveform_missing")
                continue
            parts = wav.split(",")
            arr = np.array([float(x) for x in parts[:max_len]], dtype=float)
            if arr.size < max_len:
                # pad
                arr = np.pad(arr, (0, max_len - arr.size), mode="constant")
            rows.append(arr)
            labels.append(0 if row.get("fault_type","").lower()=="normal" else 1)
        except Exception as e:
            log_skip(f"wave_{idx}", f"waveform_parse_error:{e}")
            continue
    if not rows:
        return np.empty((0,max_len,1)), np.empty((0,))
    X = np.array(rows).reshape(-1, max_len, 1)
    y = np.array(labels, dtype=int)
    return X, y

def build_vibration_dataset(df: pd.DataFrame):
    """Return vibration values array for autoencoder training (n_samples, n_features)"""
    x = df["vibration"].values.astype(float).reshape(-1,1)
    return x

def build_catboost_health(df: pd.DataFrame):
    features = ["WTI","OTI","oil_level","vibration","load_mw","CB_opcount"]
    df2 = df.dropna(subset=features + ["health_index"])
    X = df2[features].values.astype(float)
    y = df2["health_index"].astype(float).values
    return X, y

# --------------------
# Model training functions
# --------------------
def train_isolation_forest(df: pd.DataFrame):
    logger.info("Training Isolation Forest...")
    X, y = build_isolation_dataset(df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=CONFIG["random_seed"])
    model.fit(Xs)
    outp = os.path.join(CONFIG["models_dir"], "isolation_forest.joblib")
    joblib.dump({"model": model, "scaler": scaler}, outp)
    logger.info(f"IsolationForest saved -> {outp}")
    # produce anomaly score column for dashboard
    scores = model.decision_function(Xs)
    anomalies = (model.predict(Xs) == -1).astype(int)
    df_out = df.copy()
    df_out["anomaly_score"] = scores
    df_out["anomaly_flag"] = anomalies
    df_out.to_csv(os.path.join(CONFIG["out_dir"], "isolation_predictions.csv"), index=False)

def train_xgb_failure(df: pd.DataFrame):
    if xgb is None:
        logger.error("xgboost not installed. Skipping XGBoost training.")
        return
    logger.info("Training XGBoost Failure Predictor...")
    X, y = build_xgb_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_seed"])
    model = xgb.XGBClassifier(**CONFIG["xgb_params"])
    model.fit(X_train, y_train, eval_set=[(X_test,y_test)], early_stopping_rounds=10, verbose=False)
    outp = os.path.join(CONFIG["models_dir"], "xgb_failure.joblib")
    joblib.dump(model, outp)
    logger.info(f"XGBoost saved -> {outp}")
    # Evaluate
    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test))>1 else None
    logger.info(f"XGB AUC (test): {auc}")
    # save predictions for dashboard
    preds = model.predict_proba(X)[:,1]
    df_pred = df.iloc[:len(preds)].copy()
    df_pred["failure_prob"] = preds
    df_pred.to_csv(os.path.join(CONFIG["out_dir"], "xgb_failure_predictions.csv"), index=False)

def train_lstm_forecast(df: pd.DataFrame):
    if tf is None:
        logger.error("TensorFlow not installed. Skipping LSTM training.")
        return
    logger.info("Training LSTM Forecast (load)...")
    X, y = build_lstm_dataset(df, target_col="load_mw", window=CONFIG["lstm_window"])
    if len(X) < 10:
        logger.warning("Not enough data for LSTM training. Skipping.")
        return
    split = int(len(X)*(1-CONFIG["test_size"]))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = models.Sequential([
        layers.Input(shape=(CONFIG["lstm_window"],1)),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=CONFIG["tf_lr"], clipnorm=CONFIG["clipnorm"])
    model.compile(optimizer=opt, loss="mse")
    cb = [callbacks.ModelCheckpoint(os.path.join(CONFIG["models_dir"], "lstm_forecast.h5"), save_best_only=True),
          callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=CONFIG["epochs_lstm"], batch_size=CONFIG["batch_size"], callbacks=cb, verbose=2)
    # Save model and predictions
    model.save(os.path.join(CONFIG["models_dir"], "lstm_forecast_full.h5"))
    preds = model.predict(X).flatten()
    df_pred = df.iloc[CONFIG["lstm_window"]:CONFIG["lstm_window"]+len(preds)].copy()
    df_pred["load_pred"] = preds
    df_pred.to_csv(os.path.join(CONFIG["out_dir"], "lstm_load_predictions.csv"), index=False)
    logger.info("LSTM forecast saved and predictions exported.")

def train_cnn_lstm_fault(df: pd.DataFrame):
    if tf is None:
        logger.error("TensorFlow not installed. Skipping CNN+LSTM training.")
        return
    logger.info("Preparing waveform dataset for CNN+LSTM...")
    X, y = build_waveform_dataset(df, max_len=256)
    if X.shape[0] < 10:
        logger.warning("Insufficient waveform samples for CNN+LSTM. Skipping.")
        return
    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_seed"])
    # build small model
    inp = layers.Input(shape=X_train.shape[1:])
    x = layers.Conv1D(32, 5, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=CONFIG["tf_lr"], clipnorm=CONFIG["clipnorm"])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    cb = [callbacks.ModelCheckpoint(os.path.join(CONFIG["models_dir"], "cnn_lstm_fault.h5"), save_best_only=True),
          callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=CONFIG["epochs_cnn_lstm"], batch_size=32, callbacks=cb, verbose=2)
    model.save(os.path.join(CONFIG["models_dir"], "cnn_lstm_fault_full.h5"))
    # predictions
    preds = model.predict(X).flatten()
    df_out = df.iloc[:len(preds)].copy()
    df_out["fault_prob"] = preds
    df_out.to_csv(os.path.join(CONFIG["out_dir"], "cnn_lstm_fault_predictions.csv"), index=False)
    logger.info("CNN+LSTM model trained and predictions saved.")

def train_autoencoder_vib(df: pd.DataFrame):
    if tf is None:
        logger.error("TensorFlow not installed. Skipping autoencoder training.")
        return
    logger.info("Training vibration autoencoder...")
    X = build_vibration_dataset(df)
    if len(X) < 10:
        logger.warning("Not enough vibration samples for autoencoder. Skipping.")
        return
    input_dim = X.shape[1]
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(input_dim)(x)
    ae = models.Model(inputs=inp, outputs=out)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["tf_lr"]), loss="mse")
    cb = [callbacks.ModelCheckpoint(os.path.join(CONFIG["models_dir"], "autoencoder_vib.h5"), save_best_only=True)]
    ae.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, callbacks=cb, verbose=2)
    ae.save(os.path.join(CONFIG["models_dir"], "autoencoder_vib_full.h5"))
    # compute recon error and threshold
    recon = ae.predict(X)
    mse = np.mean((X - recon)**2, axis=1)
    thr = np.percentile(mse, 99)
    df_out = df.iloc[:len(mse)].copy()
    df_out["vib_recon_mse"] = mse
    df_out["vib_anomaly"] = (mse > thr).astype(int)
    df_out.to_csv(os.path.join(CONFIG["out_dir"], "autoencoder_vib_predictions.csv"), index=False)
    logger.info(f"Autoencoder trained. Recon MSE threshold (99th percentile) = {thr:.6f}")

def train_prophet_load(df: pd.DataFrame):
    if Prophet is None:
        logger.error("Prophet not installed. Skipping Prophet training.")
        return
    logger.info("Training Prophet baseline on load series...")
    # prophet requires ds,y columns
    dfp = df[["timestamp","load_mw"]].rename(columns={"timestamp":"ds","load_mw":"y"})
    # drop missing
    dfp = dfp.dropna()
    model = Prophet()
    model.fit(dfp)
    future = model.make_future_dataframe(periods=CONFIG["prophet_periods"], freq="H")
    forecast = model.predict(future)
    outp = os.path.join(CONFIG["models_dir"], "prophet_model.json")
    try:
        model.save(outp)
    except Exception:
        # older prophet versions may not support save; instead pickle forecast
        save_json({"forecast": forecast.to_dict()}, os.path.join(CONFIG["out_dir"], "prophet_forecast.json"))
    # Save forecast
    forecast.to_csv(os.path.join(CONFIG["out_dir"], "prophet_forecast.csv"), index=False)
    logger.info("Prophet model trained and forecast exported.")

def train_catboost_health(df: pd.DataFrame):
    if CatBoostRegressor is None:
        logger.error("CatBoost not installed. Skipping CatBoost training.")
        return
    logger.info("Training CatBoost regression for health_index...")
    X, y = build_catboost_health(df)
    if len(X) < 20:
        logger.warning("Not enough data for CatBoost. Skipping.")
        return
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_seed"])
    model = CatBoostRegressor(**CONFIG["catboost_params"], verbose=50)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    outp = os.path.join(CONFIG["models_dir"], "catboost_health.cbm")
    model.save_model(outp)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logger.info(f"CatBoost health MAE (test): {mae}")
    # full preds for dashboard
    preds_full = model.predict(X)
    df_out = df.iloc[:len(preds_full)].copy()
    df_out["health_pred"] = preds_full
    df_out.to_csv(os.path.join(CONFIG["out_dir"], "catboost_health_predictions.csv"), index=False)

# --------------------
# Wrapper to run all training steps sequentially (with try/except per model)
# --------------------
def run_all_training(unified_df: pd.DataFrame):
    # create models dir if missing
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    steps = [
        ("IsolationForest", train_isolation_forest),
        ("XGBoost Failure", train_xgb_failure),
        ("LSTM Forecast", train_lstm_forecast),
        ("CNN+LSTM Fault", train_cnn_lstm_fault),
        ("Autoencoder Vibration", train_autoencoder_vib),
        ("Prophet Load", train_prophet_load),
        ("CatBoost Health", train_catboost_health)
    ]

    for name, func in steps:
        try:
            logger.info(f"--- Starting: {name} ---")
            func(unified_df)
            logger.info(f"--- Completed: {name} ---\n")
        except Exception as e:
            logger.exception(f"Error in step {name}: {e}")
            # continue to next model
            continue

    # Save skipped rows log
    if SKIPPED_ROWS:
        save_json(SKIPPED_ROWS, os.path.join(CONFIG["out_dir"], "skipped_rows.json"))
        logger.info(f"Saved skipped rows summary -> outputs/skipped_rows.json")

# --------------------
# CLI / Main
# --------------------
def main(args):
    if args.simulate:
        simulate_unified_dataset(CONFIG["data_csv"], rows=args.simulate_rows)
    df = load_unified_csv(CONFIG["data_csv"])
    run_all_training(df)
    logger.info("All training steps finished. Outputs placed in outputs/ and models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Digital Twin Master Pipeline")
    parser.add_argument("--simulate", action="store_true", help="Force simulation of dataset (overwrites file if exists)")
    parser.add_argument("--simulate-rows", type=int, default=CONFIG["simulate_rows"], help="Number of rows for simulated dataset")
    parser.add_argument("--data-csv", type=str, default=CONFIG["data_csv"], help="Path to unified CSV dataset")
    args = parser.parse_args()
    CONFIG["data_csv"] = args.data_csv
    if args.simulate:
        CONFIG["simulate_data_if_missing"] = True
        CONFIG["simulate_rows"] = args.simulate_rows
    main(args)

if __name__ == "__main__":
    logger.info("Loading unified dataset...")

    # Auto-load or auto-simulate dataset
    df = load_unified_csv(CONFIG["data_csv"])

    logger.info("Dataset loaded. Starting full training pipeline...")

    # Train all models sequentially
    run_all_training(df)

    logger.info("Training + Predictions completed for ALL models.")
