import os

import librosa
import numpy as np
import streamlit as st
from models import BinaryModels, MultiModels

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def main():
    bmodel_rf = BinaryModels("RandomForest")
    bmodel_gb = BinaryModels("GradientBoosting")
    bmodel_lr = BinaryModels("LogisticRegression")
    mmodel_rf = MultiModels("RandomForest")
    mmodel_gb = MultiModels("GradientBoosting")
    mmodel_lr = MultiModels("LogisticRegression")

    models = {
        "Наличие патологий": {
            "LogisticRegression": bmodel_lr,
            "RandomForest": bmodel_rf,
            "GradientBoosting": bmodel_gb,
        },
        "Определение наличия и локации патологии": {
            "LogisticRegression": mmodel_lr,
            "RandomForest": mmodel_rf,
            "GradientBoosting": mmodel_gb,
        },
    }

    st.title("Классификатор биения сердца")
    st.write("Загрузите записьс электронного фонендоскопа и выберите параметры анализа для получения рекомендаций")

    models_list = ["GradientBoosting", "RandomForest", "LogisticRegression"]

    # Инициализация сессии (если её нет)
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "task_type" not in st.session_state:
        st.session_state.task_type = list(models.keys())[0]
    if "model_name" not in st.session_state:
        st.session_state.model_name = models_list[0]
    if "result" not in st.session_state:
        st.session_state.result = None

    # Загрузка файла (сохраняется в сессии)
    st.session_state.uploaded_file = st.file_uploader(
        "Выберите аудио в формате wav",
        type=["wav"],
        key="file_uploader",  # Уникальный ключ для виджета
    )

    # Выбор задачи (сохраняется в сессии)
    st.session_state.task_type = st.selectbox("Выберите задачу", list(models.keys()), key="task_selectbox")

    # Выбор модели (сохраняется в сессии)
    st.session_state.model_name = st.selectbox("Выберите модель", models_list, key="model_selectbox")

    # Если файл загружен
    if st.session_state.uploaded_file is not None:
        # Сохраняем файл на сервер (в папку UPLOAD_FOLDER)
        file_path = os.path.join(UPLOAD_FOLDER, st.session_state.uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())

        st.success(f"Файл {st.session_state.uploaded_file.name} успешно загружен")

        # Кнопка "Получить результат" (работает только для текущего пользователя)
        if st.button("Получить результат", key="analyze_button"):
            # Загрузка аудио и обработка
            signal, sr = librosa.load(file_path, sr=None, mono=True)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)

            # Получаем результат и сохраняем в сессию
            st.session_state.result = models[st.session_state.task_type][st.session_state.model_name](features)

        # Выводим результат (если он есть в сессии)
        if st.session_state.result is not None:
            st.write(f"Результат анализа: **{st.session_state.result}**")

    # uploaded_file = st.file_uploader("Выберите аудио в формате wav", type=["wav"])
    #
    # task_type = st.selectbox("Выберите задачу", ["Наличие патологий", "Определение наличия и локации патологии"])
    #
    # model_name = st.selectbox("Выберите модель", models_list)
    #
    # if uploaded_file is not None:
    #     file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    #     with open(file_path, "wb") as f:
    #         f.write(uploaded_file.getbuffer())
    #
    #     st.success(f"Файл {uploaded_file.name} успешно загружен!")
    #
    #     if st.button("Получить результат"):
    #         # Загрузка аудио
    #         signal, sr = librosa.load(file_path, sr=None, mono=True)
    #         mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    #         features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    #
    #         result = models[task_type][model_name](features)
    #
    #         st.write(f"Результат анализа: **{result}**")


if __name__ == "__main__":
    main()
