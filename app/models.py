import joblib
import numpy as np


class BinaryModels:
    def __init__(self, model_name=""):
        self.binary_scaler = joblib.load("./app/models/binary_standard_scaler.pkl")

        binary_model_path = f"./app/models/binary_cdpd_cda_{model_name}_best_model.pkl"

        self.binary_model = joblib.load(binary_model_path)

    def __call__(self, data):
        # 1 - normal, 0 - patology
        data_scaled = self.binary_scaler.transform(data)
        y_pred = self.binary_model.predict(data_scaled)
        if y_pred[0] == 1:
            return "normal"
        elif y_pred[0] == 0:
            return "patology"


class MultiModels:
    def __init__(self, model_name=""):
        self.multi_scaler = joblib.load("./app/models/multilabel_standard_scaler.pkl")

        multi_model_path = f"./app/models/multilabel_cdpd_cda_{model_name}_best_model.pkl"

        self.multi_model = joblib.load(multi_model_path)

    def __call__(self, data):
        # Norm - 0, MK patology - 1, AK patology - 2, Other patology - 3
        data_scaled = self.multi_scaler.transform(data)
        y_pred = self.multi_model.predict(data_scaled)

        patology_text = " patology(ies), be careful!"
        res = []
        if y_pred[0][0] == 1:
            return "norm"
        else:
            if y_pred[0][1] == 1:
                res.append("MK")
            if y_pred[0][2] == 1:
                res.append("AK")
            if y_pred[0][3] == 1:
                res.append("other")
            return " ".join(res) + patology_text


# streamlit run app/main.py
if __name__ == "__main__":
    bm = BinaryModels("GradientBoosting")
    mm = MultiModels("GradientBoosting")

    test_audio = np.random.rand(10, 20)
    res = bm(test_audio)
    res_multi = mm(test_audio)
    print(res)
    print("\n\n", res_multi)
