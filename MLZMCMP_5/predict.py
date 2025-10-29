import pickle
import sklearn
from flask import Flask
from flask import request
from flask import jsonify

# Load model and vectorizer

with open("model_conv.pkl", 'rb') as f_in:
    model = pickle.load(f_in)

with open("dv_conv.pkl", 'rb') as f_in:
    dv = pickle.load(f_in)

# Vecrorize the customer input
customer = {
    "lead_source": "events",
    "industry": "other",
    "number_of_courses_viewed": 0,
    "annual_income": 0.0,
    "employment_status": "NA",
    "location": "asia",
    "interaction_count": 0,
    "lead_score": 0.03
}


app = Flask('Converted')

@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()

    vector = dv.transform([customer])
    prob = model.predict_proba(vector)[0,1]

    result = {
        "converted_pobability": prob,
        "converted" : bool(prob >= 0.5)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    #serve(app, host='0.0.0.0', port=9696)