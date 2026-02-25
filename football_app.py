import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

st.title("⚽ Football AI Ultimate Predictor")

# رفع ملف HTML
uploaded_file = st.file_uploader("ارفع ملف HTML للإحصائيات", type="html")

def extract_tables(html_content):
    soup = BeautifulSoup(html_content, 'html5lib')  # استخدام html5lib لتجنب مشاكل lxml
    tables = soup.find_all('table')
    dfs = []
    for table in tables:
        try:
            df = pd.read_html(str(table))[0]
            dfs.append(df)
        except:
            continue
    return dfs

def analyze_matches(df):
    result = {}
    # التحقق من الأعمدة الشائعة
    if 'HomeScore' in df.columns and 'AwayScore' in df.columns:
        home_goals = df['HomeScore'].sum()
        away_goals = df['AwayScore'].sum()
        result['HomeGoals'] = home_goals
        result['AwayGoals'] = away_goals
        result['TotalGoals'] = home_goals + away_goals
    return result

def predict_outcome(df):
    # تحليل مبسط بناءً على آخر 6 مباريات لكل فريق
    last_games = df.tail(6)
    home_avg = last_games['HomeScore'].mean() if 'HomeScore' in df.columns else 0
    away_avg = last_games['AwayScore'].mean() if 'AwayScore' in df.columns else 0
    prediction = ""
    if home_avg > away_avg:
        prediction = "الفريق الأول مرشح للفوز"
    elif away_avg > home_avg:
        prediction = "الفريق الثاني مرشح للفوز"
    else:
        prediction = "نتيجة متقاربة، التعادل محتمل"
    return prediction

if uploaded_file:
    html_content = uploaded_file.read()
    dfs = extract_tables(html_content)
    if dfs:
        st.success(f"تم العثور على {len(dfs)} جدول في الملف.")
        for i, df in enumerate(dfs):
            st.subheader(f"جدول {i+1}")
            st.dataframe(df)

            analysis = analyze_matches(df)
            if analysis:
                st.write(f"مجموع أهداف الفريق الأول: {analysis.get('HomeGoals',0)}")
                st.write(f"مجموع أهداف الفريق الثاني: {analysis.get('AwayGoals',0)}")
                st.write(f"مجموع أهداف المباراة: {analysis.get('TotalGoals',0)}")
                st.write("التوقع:", predict_outcome(df))
            else:
                st.warning("لا يمكن تحليل هذا الجدول. تأكد من أن الأعمدة موجودة (HomeScore, AwayScore).")
    else:
        st.warning("لم يتم العثور على أي جداول في الملف.")
