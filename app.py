from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

from api_gemini import generate_diet_plan  # make sure this file exists

app = Flask(__name__)


model = joblib.load("diabetes_self.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1) Read values from form
        name = str(request.form["Name"])
        state = str(request.form["State"])
        country = str(request.form["Country"])
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        dpf = float(request.form["DiabetesPedigreeFunction"])
        age = float(request.form["Age"])


        data = [[pregnancies, glucose, blood_pressure,
                 skin_thickness, insulin, bmi, dpf, age]]

        columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        input_df = pd.DataFrame(data, columns=columns)


        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]  

       
        if pred == 1:
            prediction_label = "Diabetic"
            result_text = f"Diabetic - Confidence: {proba * 100:.2f} %"
        else:
            prediction_label = "Not Diabetic"
            
            confidence_not_diabetic = (1 - proba) * 100
            result_text = f"Not Diabetic - Confidence: {confidence_not_diabetic:.2f} %"

       
        patient_info = {
            "Name": name,
            "State": state,
            "Country": country,
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }


        diet_plan = generate_diet_plan(patient_info, prediction_label)

        
        return render_template(
            "result.html",
            prediction_text=result_text, 
            diet_plan=diet_plan
        )

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {e}", diet_plan=None)


if __name__ == "__main__":
    app.run(debug=True)
