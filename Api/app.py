import os
import logging
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# Setup Logging agar kita bisa lihat error di Render
logging.basicConfig(level=logging.INFO)

# Import fungsi model
try:
    from .model_utils import predict_proba, predict_label, load_feature_columns
except ImportError:
    from model_utils import predict_proba, predict_label, load_feature_columns

app = Flask(__name__)

# --- FIX: Preload Model (Cara Baru & Aman) ---
# Kita load model saat aplikasi baru nyala, supaya request pertama nggak lemot/timeout
with app.app_context():
    try:
        load_feature_columns()
        app.logger.info("✓ Model & Fitur berhasil dimuat saat startup")
    except Exception as e:
        app.logger.error(f"✗ Gagal memuat model saat startup: {e}")

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Breast Cancer Predictor API is Running!", 
        "endpoints": ["/health", "/predict"]
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        # Load columns (karena sudah di-preload, ini harusnya cepat)
        cols = load_feature_columns()
        return jsonify({"status": "ok", "features": cols}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        raise BadRequest("Expected application/json body")
    
    payload = request.get_json()
    
    try:
        # Proses prediksi
        probability = predict_proba(payload)
        label = int(probability >= 0.5)
        
        return jsonify({
            "success": True,
            "probability": probability,
            "label": label
        }), 200
    except Exception as e:
        app.logger.error(f"Prediction Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    # --- FIX: PENGATURAN PORT YANG BENAR ---
    # Ambil PORT dari Render (Environment), kalau tidak ada pakai 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Matikan debug mode saat di Render agar lebih stabil
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    print(f"Starting Flask Server on Port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)