import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier

st.title('🔗Важность :violet-background[признаков]')
st.write('Графики важности признаков показывают, какие факторы наиболее значимы для предсказания удовлетворенности пассажиров в различных моделях машинного обучения')

test = pd.read_csv('C:\Users\user\Desktop\ds_course\proj1\source\test.csv', sep=",")
train = pd.read_csv('C:\Users\user\Desktop\ds_course\proj1\source\train.csv', sep=",")
data = pd.concat([test, train])
data = data.sample(129880).reset_index().drop(['index', 'id'], axis=1)
data = data.drop(['Unnamed: 0', 'Arrival Delay in Minutes', 'Departure Delay in Minutes'], axis=1)

le = LabelEncoder()
data['satisfaction'] = le.fit_transform(data['satisfaction'])
data['Class'] = le.fit_transform(data['Class'])
data['Type of Travel'] = le.fit_transform(data['Type of Travel'])
data['Customer Type'] = le.fit_transform(data['Customer Type'])


X = data[[
    'Inflight entertainment',
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

feature_names = [
'Inflight entertainment',
'Seat comfort',
'On-board service',
'Cleanliness',
'Leg room service',
'Inflight wifi service',
'Baggage handling',
'Checkin service',
'Food and drink','Ease of Online booking']

st.write('''---
##### 🔹Важность признаков модели Logistic Regression''')
coefs = logistic.coef_[0]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefs)})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=importance_df, 
    x='Importance', 
    y='Feature', 
    palette='PRGn', 
    ax=ax
)
ax.set_xlabel('Важность')
ax.set_ylabel(None)
st.pyplot(fig)


importances_LGBM = lgbm_model.feature_importances_
importances_normalized_LGBM = importances_LGBM / importances_LGBM.sum()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances_normalized_LGBM})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.write('''---
##### 🔹Важность признаков модели LGBM''')
fig1, ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=importance_df, 
    x='Importance', 
    y='Feature', 
    palette='PRGn', 
    ax=ax
)
ax.set_xlabel('Важность')
ax.set_ylabel(None)
st.pyplot(fig1)


importances = des_tree.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
st.write('''---
##### 🔹Важность признаков модели Decision Tree''')
fig2, ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=importance_df, 
    x='Importance', 
    y='Feature', 
    palette='PRGn', 
    ax=ax
)
ax.set_xlabel('Важность')
ax.set_ylabel(None)
st.pyplot(fig2)

st.write('''
- Wi-Fi на борту и удобство онлайн-бронирования являются ключевыми факторами, влияющими на удовлетворенность пассажиров. Это подчеркивает важность удобства и доступности услуг.
- Обслуживание на борту и комфорт сидений также играют значительную роль, что указывает на важность комфорта во время полета.
- Обработка багажа и развлечения на борту также важны, но в меньшей степени, чем Wi-Fi и удобство онлайн-бронирования.
''')
