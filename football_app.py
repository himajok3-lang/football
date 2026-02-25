import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Football Analyzer", layout="wide")
st.title("محلل مباريات كرة القدم - RedScores")
st.markdown("الصق نص RedScores كاملاً لتحليل المباريات والإحصائيات")

# --- إدخال النص ---
input_text = st.text_area("Paste RedScores Text Here", height=600)

def extract_matches(text):
    """
    استخراج المباريات من النص مع الفريقين والوقت والنتيجة.
    """
    # نمط للتعرف على المباريات: فريق1 - فريق2
    matches_pattern = re.compile(r"(\d{1,2}:\d{2})\s+([A-Za-z\s&]+)\s+([A-Za-z\s&]+)")
    matches = matches_pattern.findall(text)
    
    # قائمة المباريات المستخرجة
    data = []
    for match in matches:
        time, team1, team2 = match
        data.append({"Time": time, "Team1": team1.strip(), "Team2": team2.strip()})
    return pd.DataFrame(data)

def extract_results(text, team_name):
    """
    استخراج آخر نتائج فريق محدد (آخر 6 مباريات).
    """
    pattern = re.compile(
        rf"(\d{{1,2}}\.\d{{1,2}}).*?\n{team_name}\n([A-Za-z\s&]+)\n([0-9\-: ]+)", 
        re.MULTILINE
    )
    matches = pattern.findall(text)
    results = []
    for date, opponent, score in matches:
        results.append({"Date": date, "Opponent": opponent.strip(), "Score": score.strip()})
    return pd.DataFrame(results[-6:])  # آخر 6 مباريات

def extract_standings(text):
    """
    استخراج جدول الدوري و إحصائيات الفريق.
    """
    pattern = re.compile(r"#\s*(\d+)\.\s*([A-Za-z\s&]+)\s+[A-Za-z\s&]+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d:]+)\s+(\d+)")
    matches = pattern.findall(text)
    data = []
    for m in matches:
        rank, team, mp, w, d, l, gd, pts = m
        data.append({
            "Rank": int(rank), "Team": team.strip(), "MP": int(mp), "W": int(w),
            "D": int(d), "L": int(l), "GD": gd, "Pts": int(pts)
        })
    return pd.DataFrame(data)

if input_text:
    st.subheader("المباريات المكتشفة")
    matches_df = extract_matches(input_text)
    st.dataframe(matches_df)

    # اختيار الفريق للتحليل
    team_selected = st.selectbox("اختر الفريق لتحليل آخر 6 مباريات", matches_df["Team1"].unique())
    last_results = extract_results(input_text, team_selected)
    
    st.subheader(f"آخر 6 مباريات لفريق {team_selected}")
    st.dataframe(last_results)

    st.subheader("جدول ترتيب الدوري (إن وجد)")
    standings_df = extract_standings(input_text)
    if not standings_df.empty:
        st.dataframe(standings_df)
    
    # --- تحليل الرسوم البيانية ---
    st.subheader("تحليل الأهداف آخر 6 مباريات")
    if not last_results.empty:
        goals_for = []
        goals_against = []
        for s in last_results["Score"]:
            try:
                t1, t2 = re.findall(r'\d+', s)
                goals_for.append(int(t1))
                goals_against.append(int(t2))
            except:
                goals_for.append(0)
                goals_against.append(0)
        fig, ax = plt.subplots()
        ax.plot(last_results["Date"], goals_for, label="أهداف الفريق")
        ax.plot(last_results["Date"], goals_against, label="أهداف الخصم")
        ax.set_ylabel("عدد الأهداف")
        ax.set_xlabel("التاريخ")
        ax.legend()
        st.pyplot(fig)

    st.subheader("تحليل Over/Under و BTTS (تقريبي من النتائج الأخيرة)")
    over_1_5 = sum((gf+ga)>=2 for gf, ga in zip(goals_for, goals_against))/len(goals_for)*100
    over_2_5 = sum((gf+ga)>=3 for gf, ga in zip(goals_for, goals_against))/len(goals_for)*100
    btts = sum((gf>0 and ga>0) for gf, ga in zip(goals_for, goals_against))/len(goals_for)*100
    st.write(f"احتمالية Over 1.5 أهداف: {over_1_5:.1f}%")
    st.write(f"احتمالية Over 2.5 أهداف: {over_2_5:.1f}%")
    st.write(f"احتمالية BTTS (كلا الفريقين يسجل): {btts:.1f}%")
