import pandas as pd
import pickle
from collections import defaultdict
import numpy as np

# Format Output
# {
#     "score_proba":{result}
# }

# raw_input = {'person_age': 26,
#   'person_income': 70000,
#   'person_home_ownership': 'RENT',
#   'person_emp_length': 2.0,
#   'loan_intent': 'DEBTCONSOLIDATION',
#   'loan_grade': 'B',
#   'loan_amnt': 10000,
#   'loan_int_rate': 9.45,
#   'loan_percent_income': 0.14,
#   'cb_person_default_on_file': 'N',
#   'cb_person_cred_hist_length': 3
# }

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

preprocess = CustomUnpickler(open("model/FE-SC-IMP-OHE-1.0.0.pkl", "rb")).load()

# with open("model/FE-SC-IMP-OHE-1.0.0.pkl", "rb") as f: #A
#     preprocess= pickle.load(f) #B

model = CustomUnpickler(open("model/M-LR-1.0.0.pkl", "rb")).load()

def formating_data(raw_input):
    # pandas dataframe
    # urutan colomn
    raw_input = pd.DataFrame.from_dict(raw_input, orient="index").T.replace({
        None: np.nan,
        "null":np.nan,
        "" : np.nan
    })
    return raw_input

def preprocess_data(raw_input):
    """
    acct dictionary format input
    return pandas / numpy with the same format as development
    """
    # validate data
    X = preprocess.transform(raw_input)
    return X

def predict_data(data):
    # load model
    # input dict
    # preprocess preprocess_data
    # return final predictions
    data = formating_data(data)
    data = preprocess_data(data)
    result = model.predict_proba(data)[:,1]
    return round(result[0],6)

# Cek hasil prediksi menggunakan raw_input

# if __name__ == "__main__":
#     result = predict_data(raw_input)
#     print(result)
