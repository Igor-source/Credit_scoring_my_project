# import requests
# import streamlit as st

# st.title("Кредитная карта Premium")
# st.write("Новая кредитная карта с мгновенным одобрением")

# with st.form("Подать заявку"):
#     age = st.number_input("Ваш возраст", min_value=18)
#     income = st.number_input("Ваш доход в тысячах рублей", min_value=0)
#     education = st.checkbox("У меня есть высшее образование")
#     work = st.checkbox("У меня есть стабильная работа")
#     car = st.checkbox("У меня есть автомобиль")
#     submit = st.form_submit_button("Подать заявку")
#     monthly_income = st.number_input("Ваш доход в тысячах рублей", min_value=0)
#     employment_type: 
#     months_at_job: int
#     education: str
#     loan_amount: float
#     loan_term_months: int
#     dependents: int
#     region: str
#     requested_product: str
    

# if submit:
#     data = {"age": age, "income": income, "education": education, "work": work, "car": car}
#     st.write(data)
#     response = requests.post("http://localhost:8000/score", json=data)
#     st.write(response.json())
    
    

import streamlit as st
import pandas as pd
import joblib

# Загружаем обученную модель
model = joblib.load("logistic_regression_credit_scoring.pkl")

st.set_page_config(
    page_title="Credit Scoring",
    page_icon="💳",
    layout="centered"
)

st.title("Кредитный скоринг")
st.write("Введите данные клиента для оценки вероятности дефолта.")

# Создаём форму
with st.form("credit_form"):

    st.subheader("Персональные данные")

    age = st.number_input(
        "Возраст",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    education = st.selectbox(
        "Образование",
        ["school", "college", "bachelor", "master"]
    )

    dependents = st.number_input(
        "Количество иждивенцев",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    region = st.selectbox(
        "Регион",
        ["north", "south", "east", "west", "central"]
    )

    st.subheader("Работа и доход")

    monthly_income = st.number_input(
        "Ежемесячный доход",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )

    employment_type = st.selectbox(
        "Тип занятости",
        ["employee", "self_employed", "contractor", "unemployed", "business"]
    )

    months_at_job = st.number_input(
        "Стаж работы (месяцев)",
        min_value=0,
        value=12,
        step=1
    )

    st.subheader("Параметры кредита")

    loan_amount = st.number_input(
        "Сумма кредита",
        min_value=0.0,
        value=100000.0,
        step=10000.0
    )

    loan_term_months = st.selectbox(
        "Срок кредита (месяцев)",
        [3, 6, 12, 18, 24, 36, 48, 60]
    )

    requested_product = st.selectbox(
        "Кредитный продукт",
        ["cash", "refinance", "card","pos", "auto"]
    )

    # Кнопка отправки формы
    submitted = st.form_submit_button("Оценить заявку")


# Обработка отправки формы
if submitted:

    # Формируем DataFrame с признаками
    features = pd.DataFrame([{
        "age": age,
        "monthly_income": monthly_income,
        "employment_type": employment_type,
        "months_at_job": months_at_job,
        "education": education,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "dependents": dependents,
        "region": region,
        "requested_product": requested_product
    }])

    # Получаем прогноз модели
    prediction = int(model.predict(features)[0])

    # Получаем вероятность дефолта
    probability = float(model.predict_proba(features)[0, 1])

    # Отображаем результат
    st.subheader("Результат оценки")

    st.metric(
        "Вероятность дефолта",
        f"{probability:.2%}"
    )

    if prediction == 0:
        st.success("Модель оценила заявку как низкорисковую.")
    else:
        st.error("Модель оценила заявку как высокорисковую.")

    st.write("Предсказанный класс:", prediction)
