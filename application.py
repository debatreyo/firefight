# DEPENDENCIES
# for loading data processing pipeline & trained ml model
import pickle
# for dataframe manipulation
import pandas as pd
# for numerical computations
import numpy as np
# for working with local paths
from pathlib import Path
# for transforming data before model prediction
from sklearn.preprocessing import StandardScaler
# for creating web app
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template


# instantiate web app
application = Flask(__name__)
app = application # assign to variable with shorter name for easier working

# IMPORT MODEL & SCALAR OBJECT
scaler_path = Path(r"D:\Forest Fire ML Project\models\scaler.pkl")
model_path = Path(r"D:\Forest Fire ML Project\models\lin_reg.pkl")

scaler = pickle.load(open(scaler_path, "rb"))
model = pickle.load(open(model_path, "rb"))

# ROUTE FOR HOME PAGE
@app.route("/")
def index():
    return render_template("index.html")

# ROUTE FOR MAKING MODEL PREDICTIONS
@app.route("/model_predict", methods=["GET", "POST"])
def model_predict():
    if request.method == "POST":
        # fetch the input feature data entered by user
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        input_data = [[
            Temperature, RH, Ws,
            Rain, FFMC, DMC, ISI,
            Classes, Region
        ]]

        # scale input data
        input_data_scaled = scaler.transform(input_data)

        # predict FWI
        prediction = model.predict(input_data_scaled)

        return render_template("home.html", prediction=prediction[0])
    else:
        # show the page containing input fields to accept user data
        return render_template("home.html")


# RUN WEB APP
if __name__ == "__main__":
    app.run(host="0.0.0.0") # change flask default host to make
                            # app accessible from any IP address
                            # (i.e. any device which runs this web app)

"""
Label tag ->
`for` should be same as `id` of its Input tag

Input tag ->
`id` should match `for` of its corresponding Label
`name` will be used in application.py file to 
refer to the value entered by user in this input field.

request.form.get("...") ->
The "..." should match the `name` of its corresponding Input tag.

render_template("home.html", prediction=prediction[0]) ->
Parameter name `prediction=` is my choice. Just need to ensure
that in the HTML file too using same {{prediction}}.
And value it refers to is the variable used in the function (same name used).
Instead could have also done this:
result=prediction[0] and in home.html fil write
<h3>The predicted Forest Weather Index (FWI) is {{result}}</h3>

"""