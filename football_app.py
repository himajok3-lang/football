import streamlit as st
import torch
import torch.nn as nn
import re

st.title("⚽ Football AI Ultimate Predictor")
st.write("توقعات شاملة لكل مباراة باستخدام بيانات BetExplorer.")

# -----------------------------
class FullNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=20, output_size=3):
        super(FullNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out, dim=1)
        return out

model = FullNN()

# -----------------------------
# Parsing BetExplorer text
def parse_betexplorer_text(text):
    lines = text.splitlines()
    teams = []
    h2h_stats = []
    odds = []
    last6_team1 = []
    last6_team2 = []

    # Extract teams
    for line in lines:
        if "vs" in line.lower():
            parts = line.split("vs")
            teams.append((parts[0].strip(), parts[1].strip()))
    
    # Extract H2H
    h2h_pattern = re.compile(r"(\d+)\s+wins.*?(\d+)\s+draws.*?(\d+)\s+wins", re.IGNORECASE)
    for line in lines:
        match = h2h_pattern.search(line)
        if match:
            h2h_stats = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    
    # Extract Odds
    odds_pattern = re.compile(r"1[\s]*X[\s]*2.*?([\d.]+).*?([\d.]+).*?([\d.]+)")
    for line in lines:
        match = odds_pattern.search(line)
        if match:
            odds = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            break

    # Last 6 results
    score_pattern = re.compile(r"(\d+)\s*:\s*(\d+)")
    for line in lines:
        score_match = score_pattern.search(line)
        if score_match:
            t1_score = int(score_match.group(1))
            t2_score = int(score_match.group(2))
            if len(last6_team1) < 6:
                last6_team1.append(t1_score - t2_score)
            if len(last6_team2) < 6:
                last6_team2.append(t2_score - t1_score)
    
    if not teams:
        st.error("❌ لم يتم العثور على فرق في النص.")
        return None, None, None, None, None

    return teams, h2h_stats, odds, last6_team1, last6_team2

# -----------------------------
def convert_results_to_features(results):
    features = []
    for diff in results:
        if diff > 0:
            features.append(1.0)
        elif diff == 0:
            features.append(0.5)
        else:
            features.append(0.0)
    while len(features) < 6:
        features.append(0.5)
    return features

def create_features(h2h, odds, last6_t1, last6_t2):
    features = []
    total = sum(h2h) if sum(h2h) > 0 else 1
    features.extend([h2h[0]/total, h2h[1]/total, h2h[2]/total])
    features.extend(odds if odds else [0.33,0.34,0.33])
    features.extend(convert_results_to_features(last6_t1))
    features.extend(convert_results_to_features(last6_t2))
    return torch.tensor([features], dtype=torch.float32)

# -----------------------------
def compute_btts(last6_t1, last6_t2):
    t1_goals = sum(1 for g in last6_t1 if g>0)
    t2_goals = sum(1 for g in last6_t2 if g>0)
    return (t1_goals>=3 and t2_goals>=3)

def compute_over_under(last6_t1, last6_t2, threshold=2.5):
    avg_goals = (sum([max(0,g) for g in last6_t1]) + sum([max(0,g) for g in last6_t2]))/6
    return "Over" if avg_goals>threshold else "Under"

def compute_double_chance(pred):
    return {
        "1X": pred[0]+pred[1],
        "12": pred[0]+pred[2],
        "X2": pred[1]+pred[2]
    }

def get_label(score1, score2):
    if score1 > score2:
        return 0
    elif score1 == score2:
        return 1
    else:
        return 2

# -----------------------------
# Training section
st.subheader("📊 تدريب النموذج على بيانات سابقة (اختياري)")

train_data_input = st.text_area("ألصق بيانات مباريات سابقة هنا:")
if st.button("تدريب النموذج"):
    if train_data_input:
        # هنا يمكن استخراج المباريات التاريخية بنفس parse_betexplorer_text
        # وتحويلها إلى features وlabels
        # مثال افتراضي: مجرد إظهار أن التدريب جاهز
        st.write("✅ النموذج جاهز للتدريب على البيانات التاريخية.")
        # يمكن إضافة حلقة تدريب PyTorch هنا كما في المثال السابق

# -----------------------------
# التوقعات
text_input = st.text_area("ألصق بيانات المباريات الجديدة هنا:")
if text_input:
    matches, h2h, odds, last6_t1, last6_t2 = parse_betexplorer_text(text_input)
    if matches:
        for team1, team2 in matches:
            features = create_features(h2h, odds, last6_t1, last6_t2)
            prediction = model(features).detach().numpy()[0]
            
            st.subheader(f"🏟 {team1} vs {team2}")
            # 1X2
            st.write("🔮 1X2 احتمالات:")
            st.write({
                f"{team1} يفوز": f"{prediction[0]*100:.2f}%",
                "تعادل": f"{prediction[1]*100:.2f}%",
                f"{team2} يفوز": f"{prediction[2]*100:.2f}%"
            })
            
            # BTTS
            st.write("⚽ كلا الفريقين يسجل؟", "نعم ✅" if compute_btts(last6_t1, last6_t2) else "لا ❌")
            
            # Over/Under
            st.write("📊 مجموع الأهداف:", compute_over_under(last6_t1, last6_t2))
            
            # Double Chance
            dc = compute_double_chance(prediction)
            st.write("🎯 فرص مزدوجة:", {
                "1X": f"{dc['1X']*100:.2f}%",
                "12": f"{dc['12']*100:.2f}%",
                "X2": f"{dc['X2']*100:.2f}%"
            })

st.write("💡 يمكنك تطوير التطبيق لاحقًا لإضافة توقع الشوط الأول إذا كانت بيانات الشوط الأول متاحة.")
