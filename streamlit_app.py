import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title=" 🚢 Titanic Classifier", layout="wide")
st.title(' 🚢 Titanic Classifier - Предсказание выживаемости пассажиров Титаника')
st.write("## Работа с датасетом титаника")

df = pd.read_csv('titanic_final_processed.csv')

st.subheader("Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("Визуализация данных")
col1, col2 = st.columns(2)
with col1:
  fig1 = px.histogram(df, x="Survived", color="FamilySize", barmode="group", title="Выживаемость исходя от количество членов семьи")
  st.plotly_chart(fig1, use_container_width=True)
with col2:
  fig2 = px.scatter(df, x="Age", y="Fare", color="Survived", title="Возрасть vs Стоимость билета")
  st.plotly_chart(fig2, use_container_width=True)
X = df.drop(['Survived'], axis=1)
y = df['Survived']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Добавление метрик оценки модели ---
st.subheader("Оценка модели")

# Делаем предсказания для тестовой и тренировочной выборок
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Группа метрик для тестовой выборки
with st.expander("Метрики на тестовой выборке"):
    st.write("Эти метрики показывают производительность модели на данных, которые она не видела.")
    
    # Расчет метрик
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Отображение метрик в колонках для удобства
    col_acc, col_prec, col_rec, col_f1 = st.columns(4)
    with col_acc:
        st.metric(label="Точность (Accuracy)", value=f"{test_accuracy:.2f}")
    with col_prec:
        st.metric(label="Точность (Precision)", value=f"{test_precision:.2f}")
    with col_rec:
        st.metric(label="Полнота (Recall)", value=f"{test_recall:.2f}")
    with col_f1:
        st.metric(label="F1-score", value=f"{test_f1:.2f}")

    st.markdown("---")
    st.write("#### Отчет о классификации (Classification Report)")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)

    st.markdown("---")
    st.write("#### Матрица ошибок (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_test_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Не выжил', 'Выжил'],
                yticklabels=['Не выжил', 'Выжил'])
    ax.set_title('Матрица ошибок')
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Фактический класс')
    st.pyplot(fig)

# Группа метрик для тренировочной выборки
with st.expander("Метрики на тренировочной выборке (для сравнения)"):
    st.write("Эти метрики показывают производительность модели на данных, на которых она обучалась. Если метрики сильно лучше, чем на тестовой выборке, это может указывать на переобучение.")
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    col_acc, col_prec, col_rec, col_f1 = st.columns(4)
    with col_acc:
        st.metric(label="Точность (Accuracy)", value=f"{train_accuracy:.2f}")
    with col_prec:
        st.metric(label="Точность (Precision)", value=f"{train_precision:.2f}")
    with col_rec:
        st.metric(label="Полнота (Recall)", value=f"{train_recall:.2f}")
    with col_f1:
        st.metric(label="F1-score", value=f"{train_f1:.2f}")
    
    st.markdown("---")
    st.write("#### Матрица ошибок (Confusion Matrix) на тренировочной выборке")
    cm_train = confusion_matrix(y_train, y_train_pred)
    
    fig_train, ax_train = plt.subplots()
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=ax_train,
                xticklabels=['Не выжил', 'Выжил'],
                yticklabels=['Не выжил', 'Выжил'])
    ax_train.set_title('Матрица ошибок (тренировочная)')
    ax_train.set_xlabel('Предсказанный класс')
    ax_train.set_ylabel('Фактический класс')
    st.pyplot(fig_train)
  
