from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("Crop_prediction.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        data = np.array([[N, P, K,
                          temperature,
                          humidity,
                          ph,
                          rainfall]])

        prediction = model.predict(data)[0]

        return render_template(
            "index.html",
            prediction_text=f"Recommended Crop : {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error : {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)
