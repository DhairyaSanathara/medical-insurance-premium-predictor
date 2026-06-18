from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

params = np.load("model_parameters.npz")

w = params["w"]
b = params["b"]
mu = params["mu"]
sigma = params["sigma"]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        children = float(request.form["children"])
        discount = float(request.form["discount"])
        gender = float(request.form["gender"])
        region = float(request.form["region"])

        x = np.array([
            age,
            bmi,
            children,
            discount,
            gender,
            region
        ])

        x_scaled = (x - mu) / sigma

        prediction = np.dot(x_scaled, w) + b

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)