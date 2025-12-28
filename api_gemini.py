# gemini_helper.py
import os
import google.generativeai as genai

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")


def generate_diet_plan(patient_info: dict, prediction_label: str) -> str:
    """
    patient_info: dict with stuff like name, state, country,age, bmi, glucose, etc.
    prediction_label: "Diabetic" or "Not Diabetic"
    """

    prompt = f"""
You are a helpful dietisian that provide dishes based on their state in their local names(for example in gujarati rice is called chokha). 
Based on the following patient information and diabetes prediction, 
create a simple, clear, beginner-friendly daily diet plan dont add any unecessary response only give what is asked for.

Patient info:
- Name: {patient_info.get("Name")}
- State: {patient_info.get("State")}
- Country: {patient_info.get("Country")}
- Age: {patient_info.get("Age")}
- BMI: {patient_info.get("BMI")}
- Glucose: {patient_info.get("Glucose")}
- BloodPressure: {patient_info.get("BloodPressure")}
- SkinThickness: {patient_info.get("SkinThickness")}
- Insulin: {patient_info.get("Insulin")}
- Diabetes Pedigree Function: {patient_info.get("DiabetesPedigreeFunction")}
- Pregnancies: {patient_info.get("Pregnancies")}

Prediction: {prediction_label}

Instructions:
- Explain in very simple language.
- Give a sample plan for: breakfast, lunch, dinner, and 2 snacks.
- Mention foods to avoid.
- Start new section by adding ------------------------ like design (for example between brekfast and lunch)
- Don't bold stuff or unecessary formating since i am using this as api data.
- Add a short disclaimer like "This is not medical advice, consult a doctor."

Use bullet points or clear sections.
"""

    response = model.generate_content(prompt)
    return response.text  # this is the diet plan as plain text
