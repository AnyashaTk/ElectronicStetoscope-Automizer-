import joblib
import numpy as np


class BinaryModels:
    def __init__(self, model_name=""):
        self.binary_scaler = joblib.load("./electronicstetoscope-automizer-/models/binary_standard_scaler.pkl")

        binary_model_path = f"./electronicstetoscope-automizer-/models/binary_cdpd_cda_{model_name}_best_model.pkl"

        self.binary_model = joblib.load(binary_model_path)

    def __call__(self, data):
        # 1 - normal, 0 - patology
        data_scaled = self.binary_scaler.transform(data)
        y_pred = self.binary_model.predict(data_scaled)
        if y_pred[0] == 1:
            return "Проблем с сердцем не обнаружено"
        elif y_pred[0] == 0:
            return (
                "Обнаружена патология, рекомендуется провести дополнительные исследования для более точного диагноза"
            )


class MultiModels:
    def __init__(self, model_name=""):
        self.multi_scaler = joblib.load("./electronicstetoscope-automizer-/models/multilabel_standard_scaler.pkl")

        multi_model_path = f"./electronicstetoscope-automizer-/models/multilabel_cdpd_cda_{model_name}_best_model.pkl"

        self.multi_model = joblib.load(multi_model_path)

    def __call__(self, data):
        # Norm - 0, MK patology - 1, AK patology - 2, Other patology - 3
        data_scaled = self.multi_scaler.transform(data)
        y_pred = self.multi_model.predict(data_scaled)

        patology_text = ", рекомендуется провести дополнительные исследования для более точного диагноза"
        res = []
        if y_pred[0][0] == 1:
            return "Проблем с сердцем не обнаружено"
        else:
            if y_pred[0][1] == 1:
                res.append("Mитрального клапана")
            if y_pred[0][2] == 1:
                res.append("Aортального клапана")
            if y_pred[0][3] == 1:
                res.append("Трикуспидального или легочного клапанов")
            return "Обнаружена " + ", ".join(res) + patology_text


# streamlit run electronicstetoscope-automizer-/main.py
if __name__ == "__main__":
    bm = BinaryModels("GradientBoosting")
    mm = MultiModels("GradientBoosting")

    test_audio = np.random.rand(10, 20)
    res = bm(test_audio)
    res_multi = mm(test_audio)
    print(res)
    print("\n\n", res_multi)
