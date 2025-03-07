import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier

test = pd.read_csv('C:/Users/user/Desktop/ds_course/proj1/source/test.csv', sep=",")
train = pd.read_csv('C:/Users/user/Desktop/ds_course/proj1/source/train.csv', sep=",")
data = pd.concat([test, train])
data = data.sample(129880).reset_index().drop(['index', 'id'], axis=1)
data = data.drop(['Unnamed: 0', 'Arrival Delay in Minutes', 'Departure Delay in Minutes'], axis=1)

le = LabelEncoder()
data['satisfaction'] = le.fit_transform(data['satisfaction'])
data['Class'] = le.fit_transform(data['Class'])
data['Type of Travel'] = le.fit_transform(data['Type of Travel'])
data['Customer Type'] = le.fit_transform(data['Customer Type'])


X = data[[
    'Seat comfort',
    'On-board service',
    'Cleanliness',
    'Leg room service',
    'Inflight wifi service',
    'Baggage handling',
    'Checkin service',
    'Food and drink',
    'Type of Travel']]
y = data['satisfaction']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



logistic = LogisticRegression(max_iter=565)
logistic.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


des_tree = DecisionTreeClassifier(max_depth=5)
des_tree.fit(X_train, y_train)

lgbm_model = LGBMClassifier(verbose=-200)
lgbm_model.fit(X_train, y_train)


# Интерфейс Streamlit
st.title("✈️ Прогноз уровня комфорта пассажиров")
st.write('''На этой странице представлены модели которые предсказывают, будет ли пассажир удовлетворен **Satisfied** или нейтрален/неудовлетворен **Neutral or Dissatisfied**''')

# Отображение исходных данных
with st.expander('Data'):
    st.write("X")
    st.dataframe(X)
    st.write("y")
    st.dataframe(y)

# Боковая панель для ввода данных
with st.sidebar:
    st.header("Введите признаки: ")
    
    # Используем st.number_input для числового ввода
    seat_comfort = st.number_input('Seat comfort (0-5)', min_value=0, max_value=5, value=3)
    on_board_service = st.number_input('On-board service (0-5)', min_value=0, max_value=5, value=3)
    cleanliness = st.number_input('Cleanliness (0-5)', min_value=0, max_value=5, value=3)
    leg_room_service = st.number_input('Leg room service (0-5)', min_value=0, max_value=5, value=3)
    inflight_wifi_service = st.number_input('Inflight wifi service (0-5)', min_value=0, max_value=5, value=3)
    baggage_handling = st.number_input('Baggage handling (0-5)', min_value=0, max_value=5, value=3)
    checkin_service = st.number_input('Checkin service (0-5)', min_value=0, max_value=5, value=3)
    food_and_drink = st.number_input('Food and drink (0-5)', min_value=0, max_value=5, value=3)
    
    # Используем st.selectbox для выбора типа путешествия
    type_of_travel = st.selectbox('Type of Travel', ['Business travel', 'Personal Travel'])


# Преобразование типа путешествия в числовой формат
type_of_travel_mapping = {'Business travel': 1, 'Personal Travel': 0}
type_of_travel_encoded = type_of_travel_mapping[type_of_travel]

# Создание массива для предсказания
new = np.array([[
    seat_comfort,
    on_board_service,
    cleanliness,
    leg_room_service,
    inflight_wifi_service,
    baggage_handling,
    checkin_service,
    food_and_drink,
    type_of_travel_encoded
]])

# Масштабирование данных
new_scaled = scaler.transform(new)

# Предсказания моделей
pred_logistic = logistic.predict(new_scaled)[0]
pred_knn = knn.predict(new_scaled)[0]
pred_des_tree = des_tree.predict(new_scaled)[0]
pred_lgbm = lgbm_model.predict(new_scaled)[0]

# Функция для отображения результата
def mapping(pred):
    return "Satisfied 😊" if pred == 1 else "Neutral or Dissatisfied 😶‍🌫️"

# Отображение предсказаний
st.subheader("🔮 Предсказания моделей:")
st.write(f"**Logistic Regression:** {mapping(pred_logistic)}")
st.write(f"**K-Nearest Neighbors:** {mapping(pred_knn)}")
st.write(f"**Decision Tree:** {mapping(pred_des_tree)}")
st.write(f"**LightGBM:** {mapping(pred_lgbm)}")
