import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.title('🔍Краткий анализ по :green-background[целевому признаку] ')
st.write('Анализ ключевых факторов, влияющих на удовлетворенность пассажиров, для выявления областей улучшения и повышения общего уровня удовлетворенности')

test = pd.read_csv('source/test.csv')
train = pd.read_csv("source/train.csv")
data = pd.concat([test, train])
data = data.sample(129880).reset_index().drop(['index', 'id'], axis=1)
data = data.drop(['Unnamed: 0', 'Arrival Delay in Minutes', 'Departure Delay in Minutes'], axis=1)

satisfaction_counts = data['satisfaction'].value_counts(normalize=True) * 100
satisfaction_df = satisfaction_counts.reset_index()
satisfaction_df.columns = ['satisfaction', 'percentage']

st.write(''' ---
##### 🔹Распределение удовлетворенности клиентов (%)''')
fig = plt.figure(figsize=(5, 3))
ax = sns.barplot(data=satisfaction_df, x='satisfaction', y='percentage', palette='PRGn')
for p in ax.containers:
    ax.bar_label(p, fmt='%.1f%%', label_type='edge', fontsize=10)

plt.xticks(rotation=0)
plt.xlabel(None)
plt.ylabel(None)
plt.ylim(0, 100)  # Ось Y в процентах
plt.tight_layout()
st.pyplot(fig)
st.write(' - Большинство клиентов остаются нейтральными или неудовлетворенными, что указывает на необходимость улучшения качества обслуживания')

st.write(''' ---
##### 🔹Распределение возраста в разрезе таргета''')
fig2 = plt.figure(figsize=(7,4))
sns.histplot(x=data['Age'], hue='satisfaction', data=data, palette='PRGn')
plt.tight_layout()
plt.ylabel(None)
st.pyplot(fig2) 

need_cols = ['Inflight service', 'Inflight wifi service', 'Ease of Online booking', 'Baggage handling']


st.write(''' ---
##### 🔹Влияние обслуживания на борту и качества Wi-Fi на борту на удовлетворенность пассажиров''')
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.kdeplot(data=data, x=need_cols[0], hue='satisfaction', ax=ax1, palette='PRGn_r', shade=True)
    plt.tight_layout()
    plt.ylabel(None)
    st.pyplot(fig1)


st.write('''
##### 🔹Влияние удобства онлайн-бронирования и обработки багажа на удовлетворенность пассажиров на удовлетворенность пассажиров''')
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.kdeplot(data=data, x=need_cols[1], hue='satisfaction', ax=ax2, palette='PRGn_r', shade=True)
    plt.tight_layout()
    plt.ylabel(None)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.kdeplot(data=data, x=need_cols[2], hue='satisfaction', ax=ax3, palette='PRGn_r', shade=True)
    plt.tight_layout()
    plt.ylabel(None)
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    sns.kdeplot(data=data, x=need_cols[3], hue='satisfaction', ax=ax4, palette='PRGn_r', shade=True)
    plt.tight_layout()
    plt.ylabel(None)
    st.pyplot(fig4)

st.write('''
- Пассажиры высоко ценят качественное обслуживание, и его улучшение может значительно повысить уровень удовлетворенности.
- Низкое качество Wi-Fi может вызвать недовольство. Улучшение Wi-Fi может помочь снизить уровень неудовлетворенности.
- Упростить процесс онлайн-бронирования, сделав его более удобным и интуитивно понятным
- Улучшить процесс обработки багажа, чтобы минимизировать задержки и потери.
         ''')
st.write('---')

vig_cols = ['Type of Travel', 'Class']
for col in vig_cols:
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Группировка и расчет процентов
    category_counts = data.groupby([col, 'satisfaction']).size().unstack()
    category_percents = category_counts.div(category_counts.sum(axis=1), axis=0).reset_index().melt(id_vars=col)

    # Построение графика
    sns.barplot(
        data=category_percents, 
        x=col, 
        y='value', 
        hue='satisfaction', 
        palette='PRGn', 
        ax=ax
    )
    
    # Добавление аннотаций
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Исключаем подписи на нулевых значениях
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.), 
                        ha='center', va='center', 
                        fontsize=12, color='white', fontweight='bold')

    # Настройка осей и легенды
    ax.set_ylabel('Процент')
    ax.set_xlabel('')
    ax.legend(title='Satisfaction')

    # Добавление заголовка с помощью st.write()
    st.write(f"##### 🔹 Влияние {col} на удовлетворенность пассажиров")
    
    # Отображение графика
    st.pyplot(fig)

st.write(''' 
- Деловые путешественники более удовлетворены, чем личные, что может быть связано с более высоким уровнем обслуживания.
-Пассажиры бизнес-класса наиболее удовлетворены, что подчеркивает важность высокого уровня обслуживания.
                 ''')
