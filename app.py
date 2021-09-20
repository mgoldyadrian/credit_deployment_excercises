from collections import UserString
from flask import Flask, jsonify, request, render_template
from logging import debug
import pickle
from predict import predict_data

app = Flask(__name__)

users = []

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route("/prediction", methods=['POST','GET'])
def make_predictions():
    if request.method == 'POST':
        data = request.get_json()
        result = predict_data(data)
        result = {
            "score_proba":str(result)
        }
        print(result)
        return jsonify(result)


@app.route('/result', methods=['POST','GET'])
def result_prediction():
    
    data = {}

    data["person_age"] = request.form.get("person_age")
    data["person_income"] = request.form.get("person_income")
    data["person_home_ownership"] = request.form.get("person_home_ownership")
    data["person_emp_length"] = request.form.get("person_emp_length")
    data["loan_intent"] = request.form.get("loan_intent")
    data["loan_grade"] = request.form.get("loan_grade")
    data["loan_amnt"] = request.form.get("loan_amnt")
    data["loan_int_rate"] = request.form.get("loan_int_rate")
    data["loan_percent_income"] = request.form.get("loan_percent_income")
    data["cb_person_default_on_file"] = request.form.get("cb_person_default_on_file")
    data["cb_person_cred_hist_length"] = request.form.get("cb_person_cred_hist_length")
    print(data)
    result = predict_data(data)
    hasil = {
        'score_proba':str(result)
    }
    print(hasil)
    prediction = str(result)
    return render_template('result.html', name=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=False)
