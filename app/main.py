import os

import streamlit as st

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def classify_audio(file_path):
    if file_path[-5:] != "y.wav":
        return "Проблем с биением сердца не обнаружено, поздравляем!"
    else:
        return """Обнаружены шумы/патологии
        Рекомендуется провести УЗИ сердца, сделать кардиограмму и посетить кардиолога"""


def main():
    st.title("Классификатор биения сердца")
    st.write("Загрузите запись с электронного фонендоскопа для получения рекомендаций")

    uploaded_file = st.file_uploader("Выберите аудио в формате wav", type=["wav"])

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Файл {uploaded_file.name} успешно загружен!")

        result = classify_audio(file_path)
        st.write(f"Результат анализа: **{result}**")


if __name__ == "__main__":
    main()
