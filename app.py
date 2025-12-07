import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# -----------------------------
# Data + Model
# -----------------------------
def generate_synthetic_data(n_samples: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    joint_temp = rng.normal(55, 10, n_samples)          # °C
    vibration = rng.normal(1.5, 0.6, n_samples)          # mm/s
    motor_current = rng.normal(10, 3, n_samples)         # A
    payload_weight = rng.normal(4, 1.2, n_samples)       # kg
    cycle_time = rng.normal(8, 1.5, n_samples)           # sec/cycle

    labels = []
    for jt, vib, cur, load, cyc in zip(
        joint_temp, vibration, motor_current, payload_weight, cycle_time
    ):
        score = 0

        if jt > 65:
            score += 2
        elif jt > 58:
            score += 1

        if vib > 2.0:
            score += 2
        elif vib > 1.4:
            score += 1

        if cur > 14:
            score += 1

        if load > 5.5:
            score += 1

        if cyc > 10.0:
            score += 1

        if score >= 4:
            labels.append("Fault")
        elif score >= 2:
            labels.append("Warning")
        else:
            labels.append("Normal")

    df = pd.DataFrame(
        {
            "joint_temperature": joint_temp,
            "gearbox_vibration": vibration,
            "motor_current": motor_current,
            "payload_weight": payload_weight,
            "cycle_time": cycle_time,
            "status": labels,
        }
    )
    return df


def train_model(df: pd.DataFrame):
    X = df[["joint_temperature", "gearbox_vibration", "motor_current",
            "payload_weight", "cycle_time"]]
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Normal", "Warning", "Fault"])

    return model, acc, cm, X.columns, ["Normal", "Warning", "Fault"]


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Robotic Arm Predictive Maintenance",
        layout="wide",
    )

    df = generate_synthetic_data()
    model, acc, cm, feature_names, cm_labels = train_model(df)

    st.title("🤖 Robotic Arm Predictive Maintenance – Joint & Gearbox Health")
    st.write(
        "Use sensor readings from a robotic arm to estimate health state, "
        "support maintenance decisions, and quantify potential downtime savings."
    )

    # ---- Layout
    col_sidebar, col_main = st.columns([1.1, 2.4])

    with col_sidebar:
        st.subheader("Input Robotic Arm Measurements")

        jt = st.slider("Joint Temperature (°C)", 40.0, 90.0, 60.0, 0.5)
        vib = st.slider("Gearbox Vibration (mm/s)", 0.2, 4.0, 1.2, 0.05)
        cur = st.slider("Motor Current (A)", 4.0, 20.0, 12.0, 0.5)
        load = st.slider("Payload Weight (kg)", 1.0, 8.0, 3.5, 0.1)
        cyc = st.slider("Average Cycle Time (sec per cycle)", 5.0, 14.0, 8.5, 0.1)

        st.markdown("---")
        cost_per_hour = st.number_input(
            "Estimated cost of robot downtime per hour (USD)",
            min_value=1000.0,
            max_value=50000.0,
            value=15000.0,
            step=500.0,
        )

        run_btn = st.button("Run Prediction")

    with col_main:
        placeholder_status = st.empty()
        st.subheader("📋 Recommendation for Maintenance Team")
        recommendation_placeholder = st.empty()

        st.subheader("💰 Potential Downtime Impact")
        cost_placeholder = st.empty()

        st.subheader("📊 Current Sensor Snapshot")
        snapshot_placeholder = st.empty()

        st.subheader("📈 Model Performance")
        perf_col1, perf_col2 = st.columns([1, 1])
        with perf_col1:
            st.metric("Validation accuracy", f"{acc * 100:.2f}%")
        cm_placeholder = perf_col2

        st.subheader("🔍 Feature Importance")
        fi_placeholder = st.empty()

    if run_btn:
        new_sample = pd.DataFrame(
            [
                {
                    "joint_temperature": jt,
                    "gearbox_vibration": vib,
                    "motor_current": cur,
                    "payload_weight": load,
                    "cycle_time": cyc,
                }
            ]
        )

        pred_label = model.predict(new_sample)[0]

        if pred_label == "Normal":
            status_text = "✅ NORMAL – Joint & gearbox health OK"
            status_color = "#d4edda"
            status_font = "#155724"
            hours_avoided = 0
            rec_lines = [
                "System is operating within normal limits.",
                "Continue routine inspections as per schedule.",
                "Monitor trends for any gradual increase in vibration or temperature.",
            ]
        elif pred_label == "Warning":
            status_text = "⚠️ WARNING – Early joint/gearbox wear"
            status_color = "#fff3cd"
            status_font = "#856404"
            hours_avoided = 1
            rec_lines = [
                "Inspect lubrication of critical joints at the next planned stop.",
                "Check gearbox clearances and mounting bolts.",
                "Plan maintenance during the next scheduled downtime window.",
            ]
        else:
            status_text = "🛑 CRITICAL – Gearbox / joint fault likely"
            status_color = "#f8d7da"
            status_font = "#721c24"
            hours_avoided = 4
            rec_lines = [
                "Inspect gearbox and joints immediately (noise, backlash, play).",
                "Limit production speed or heavy payloads until inspected.",
                "Consider taking the robot out of service for corrective maintenance.",
            ]

        status_html = f"""
        <div style="
            background-color:{status_color};
            color:{status_font};
            padding:0.9rem 1.0rem;
            border-radius:0.5rem;
            border:1px solid rgba(0,0,0,0.05);
            font-weight:600;">
            {status_text}
        </div>
        """
        placeholder_status.markdown(status_html, unsafe_allow_html=True)

        recommendation_placeholder.markdown(
            "\n".join([f"- {line}" for line in rec_lines])
        )

        savings = hours_avoided * cost_per_hour
        cost_placeholder.write(
            f"- Estimated unplanned downtime avoided: **~{hours_avoided} hours**"
        )
        cost_placeholder.write(
            f"- Approximate cost saving: **~${savings:,.0f} USD** "
            f"(at ${cost_per_hour:,.0f}/hour)"
        )

        snapshot_df = pd.DataFrame(
            {
                "Measurement": [
                    "Joint Temperature (°C)",
                    "Gearbox Vibration (mm/s)",
                    "Motor Current (A)",
                    "Payload Weight (kg)",
                    "Cycle Time (sec/cycle)",
                ],
                "Value": [jt, vib, cur, load, cyc],
            }
        )
        snapshot_placeholder.dataframe(snapshot_df, hide_index=True)

        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm, interpolation="nearest")
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xticks(np.arange(len(cm_labels)))
        ax_cm.set_yticks(np.arange(len(cm_labels)))
        ax_cm.set_xticklabels(cm_labels)
        ax_cm.set_yticklabels(cm_labels)
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
        cm_placeholder.pyplot(fig_cm)

        importances = model.feature_importances_
        fig_fi, ax_fi = plt.subplots()
        y_pos = np.arange(len(feature_names))
        ax_fi.barh(y_pos, importances)
        ax_fi.set_yticks(y_pos)
        ax_fi.set_yticklabels(
            [
                "Joint Temp",
                "Gearbox Vib",
                "Motor Current",
                "Payload",
                "Cycle Time",
            ]
        )
        ax_fi.invert_yaxis()
        ax_fi.set_xlabel("Importance")
        ax_fi.set_title("Feature Importance")
        fi_placeholder.pyplot(fig_fi)


if __name__ == "__main__":
    main()
