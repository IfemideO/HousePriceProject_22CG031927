from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        overallqual = float(request.form["overallqual"])
        grlivarea = float(request.form["grlivarea"])
        totalbsmtsf = float(request.form["totalbsmtsf"])
        garagecars = float(request.form["garagecars"])
        fullbath = float(request.form["fullbath"])
        yearbuilt = float(request.form["yearbuilt"])

        features = np.array([[overallqual, grlivarea, totalbsmtsf,
                              garagecars, fullbath, yearbuilt]])

        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ₦{int(prediction):,}"
        )

    except:
        return render_template("index.html", prediction_text="Invalid input")

if __name__ == "__main__":
    app.run(debug=True)
