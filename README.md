# 🤖 Robotic Arm Predictive Maintenance

This project uses machine learning to monitor the condition of an industrial robotic arm (joints and gearbox), detect early wear, and estimate potential downtime cost savings.

---

## 📌 Project Overview

The app takes sensor inputs:

- Joint temperature (°C)
- Gearbox vibration (mm/s)
- Motor current (A)
- Payload weight (kg)
- Cycle time (seconds per cycle)

It classifies the robot condition into:

- 🟢 Normal operation
- 🟡 Early wear / Warning
- 🔴 Fault likely / Critical

When wear or a fault is detected, the dashboard provides:

- Maintenance recommendations
- Estimated downtime savings

---

### 🚀 How to Run

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Open in your browser:
http://localhost:8501





Install dependencies:

