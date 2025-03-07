import streamlit as st
import pandas as pd

pages = [
    st.Page('about.py', title = 'О проекте 🫡'), 
    st.Page('eda.py', title = 'Обзор 🔍'),
    st.Page('ml.py', title = 'Модели 💃'),
    st.Page('feature_imp.py', title = 'Важность признаков 🔗'),
    st.Page('metrics2.py', title = 'Результаты ⚖️')
]
pg_h = st.navigation(pages)
# pg_h.run()
