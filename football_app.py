# app.py
import streamlit as st
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="محلل مباريات كرة القدم", layout="wide")
st.title("محلل مباريات كرة القدم مع الرسوم البيانية")

# رفع ملف HTML
uploaded_file = st.file_uploader("ارفع ملف HTML يحتوي على إحصائيات الفريقين", type="html")

if uploaded_file:
    soup = BeautifulSoup(uploaded_file, "html.parser")

    # ---- استخراج أسماء الفريقين ----
    teams = soup.find_all("div", class_="team-name")  # مثال، عدل حسب HTML
    if len(teams) >= 2:
        team1_name = teams[0].text.strip()
        team2_name = teams[1].text.strip()
    else:
        st.error("لم يتم العثور على أسماء الفريقين في الملف")
        st.stop()

    st.subheader(f"المباراة: {team1_name} vs {team2_name}")

    # ---- استخراج إحصائيات الفريقين ----
    def extract_stats(team_div):
        stats = {}
        try:
            stats["Avg Goals For"] = float(team_div.find("span", class_="avg-goals-for").text)
            stats["Avg Goals Against"] = float(team_div.find("span", class_="avg-goals-against").text)
            stats["BTTS"] = float(team_div.find("span", class_="btts").text.strip('%'))
            stats["Over2.5"] = float(team_div.find("span", class_="over25").text.strip('%'))
            stats["Cards"] = float(team_div.find("span", class_="cards").text)
            stats["Corners"] = float(team_div.find("span", class_="corners").text)
        except:
            st.warning(f"بعض الإحصائيات غير موجودة للفريق {team_div}")
        return stats

    team_divs = soup.find_all("div", class_="team-stats")
    analysis_team1 = extract_stats(team_divs[0])
    analysis_team2 = extract_stats(team_divs[1])

    # ---- جدول الإحصائيات ----
    st.subheader("إحصائيات الفريقين")
    stats_df = pd.DataFrame([analysis_team1, analysis_team2], index=[team1_name, team2_name])
    st.dataframe(stats_df)

    # ---- الرسوم البيانية للمقارنة ----
    st.subheader("مخططات مقارنة الإحصائيات")

    categories = ["Avg Goals For", "Avg Goals Against", "BTTS", "Over2.5", "Cards", "Corners"]
    values_team1 = [analysis_team1.get(cat, 0) for cat in categories]
    values_team2 = [analysis_team2.get(cat, 0) for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, values_team1, width, label=team1_name, color="skyblue")
    ax.bar(x + width/2, values_team2, width, label=team2_name, color="salmon")
    ax.set_ylabel("القيم")
    ax.set_title("مقارنة الإحصائيات بين الفريقين")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    st.pyplot(fig)

    # ---- توقع المباراة ----
    st.subheader("توقع المباراة")

    def calculate_match_prob(team1, team2):
        attack_diff = team1["Avg Goals For"] - team2["Avg Goals Against"]
        defense_diff = team2["Avg Goals For"] - team1["Avg Goals Against"]
        team1_win_prob = max(0, min(100, 50 + attack_diff*10 - defense_diff*5))
        team2_win_prob = max(0, min(100, 50 + defense_diff*10 - attack_diff*5))
        total = team1_win_prob + team2_win_prob
        if total > 100:
            factor = 100 / total
            team1_win_prob *= factor
            team2_win_prob *= factor
        draw_prob = max(0, 100 - (team1_win_prob + team2_win_prob))
        return team1_win_prob, draw_prob, team2_win_prob

    team1_prob, draw_prob, team2_prob = calculate_match_prob(analysis_team1, analysis_team2)

    # عرض النتائج كنص
    st.markdown(f"- احتمالية فوز {team1_name}: {team1_prob:.1f}%")
    st.markdown(f"- احتمالية التعادل: {draw_prob:.1f}%")
    st.markdown(f"- احتمالية فوز {team2_name}: {team2_prob:.1f}%")

    # رسم شريط توقعات المباراة
    st.subheader("رسم شريط توقعات المباراة")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar([team1_name, "تعادل", team2_name], [team1_prob, draw_prob, team2_prob], color=["skyblue", "grey", "salmon"])
    ax2.set_ylabel("احتمالية (%)")
    ax2.set_ylim(0, 100)
    st.pyplot(fig2)

    # ---- تنزيل CSV ----
    result_df = stats_df.copy()
    result_df["Win Prob (%)"] = [team1_prob, team2_prob]
    result_df["Draw Prob (%)"] = [draw_prob, draw_prob]

    csv = result_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="تنزيل التحليل كملف CSV",
        data=csv,
        file_name="match_analysis.csv",
        mime="text/csv"
    )
