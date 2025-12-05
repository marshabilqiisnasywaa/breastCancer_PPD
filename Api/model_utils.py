import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


_model = None
_feature_columns = None


def load_model():
	global _model
	if _model is None:
		if not os.path.exists(MODEL_PATH):
			raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
		_model = joblib.load(MODEL_PATH)
	return _model


def load_feature_columns():
	global _feature_columns
	if _feature_columns is None:
		if not os.path.exists(COLUMNS_PATH):
			raise FileNotFoundError(f"Feature columns file not found at {COLUMNS_PATH}")
		_feature_columns = joblib.load(COLUMNS_PATH)
		if not isinstance(_feature_columns, (list, tuple)):
			raise ValueError("feature_columns.pkl must contain a list/tuple of feature names")
	return _feature_columns


def make_feature_frame(input_features: dict) -> pd.DataFrame:
	columns = load_feature_columns()
	# Initialize with zeros to allow missing keys; fill provided ones.
	data = {col: 0.0 for col in columns}
	for k, v in input_features.items():
		if k in data:
			try:
				data[k] = float(v)
			except (TypeError, ValueError):
				raise ValueError(f"Feature '{k}' must be numeric, got: {v}")
	df = pd.DataFrame([data], columns=columns)
	return df


def predict_proba(input_features: dict) -> float:
	model = load_model()
	X = make_feature_frame(input_features)
	# Prefer predict_proba if available; fallback to decision_function; else predict
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(X)
		# Assume positive class is index 1
		return float(np.asarray(proba)[0, 1])
	elif hasattr(model, "decision_function"):
		# Convert decision function to probability using logistic transform approximation
		score = float(np.asarray(model.decision_function(X))[0])
		return float(1.0 / (1.0 + np.exp(-score)))
	else:
		pred = int(np.asarray(model.predict(X))[0])
		return float(pred)


def predict_label(input_features: dict, threshold: float = 0.5) -> int:
	proba = predict_proba(input_features)
	return int(proba >= threshold)

