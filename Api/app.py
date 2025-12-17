from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

try:
	from .model_utils import predict_proba, predict_label, load_feature_columns
except ImportError:
	from model_utils import predict_proba, predict_label, load_feature_columns

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
	return jsonify({"message": "Breast Cancer Predictor API", 
				 "endpoints": ["/health", "/predict"]})

@app.route("/health", methods=["GET"])
def health():
	try:
		cols = load_feature_columns()
		return jsonify({"status": "ok", "features": cols}), 200
	except Exception as e:
		return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
	if not request.is_json:
		raise BadRequest("Expected application/json body")
	payload = request.get_json()
	if not isinstance(payload, dict):
		raise BadRequest("JSON body must be an object of feature_name: value")

	try:
		probability = predict_proba(payload)
		label = int(probability >= 0.5)
		return jsonify({
			"success": True,
			"probability": probability,
			"label": label
		}), 200
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 400


# if __name__ == "__main__":
# 	# Run Flask app for local testing
# 	print(app.url_map)
# 	app.run(host="0.0.0.0", port=8000, debug=True)

if __name__ == "__main__":
    # Gunakan PORT yang disediakan server, atau 8000 kalau di local
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

