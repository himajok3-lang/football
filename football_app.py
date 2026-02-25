import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

st.title("⚽ Football AI Predictor")
st.write("تحليل المباريات القادمة باستخدام بيانات الفريقين.")

# -----------------------------
# إدخال النص من المستخدم
text_input = st.text_area("ألصق بيانات الفريقين هنا:")

# -----------------------------
# نموذج PyTorch بسيط لتوضيح الفكرة
class SimpleNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

model = SimpleNN()

# -----------------------------
# معالجة البيانات من النص (مثال: استخراج فرق، نتائج)
def parse_text(text):
    lines = text.splitlines()
    team_names = []
    for line in lines:
        if "vs" in line.lower():
            parts = line.split("vs")
            team_names.append(parts[0].strip())
            team_names.append(parts[1].strip())
    if not team_names:
        st.error("❌ لم يتم العثور على فرق في النص.")
        return None
    return team_names[:2]

teams = parse_text(text_input)

# -----------------------------
# توليد بيانات عشوائية للتجربة
def generate_features():
    return torch.rand((1,10))  # 10 ميزات عشوائية كمثال

# -----------------------------
# التوقع
if teams:
    features = generate_features()
    prediction = model(features).detach().numpy()[0]
    st.write(f"🏟 الفريقين: {teams[0]} vs {teams[1]}")
    st.write("🔮 احتمالات الفوز / التعادل / الخسارة:")
    st.write({
        f"{teams[0]} يفوز": f"{prediction[0]*100:.2f}%",
        "تعادل": f"{prediction[1]*100:.2f}%",
        f"{teams[1]} يفوز": f"{prediction[2]*100:.2f}%"
    })