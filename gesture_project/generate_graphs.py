import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load your collected data
df = pd.read_csv("gesture_data.csv")

gestures = ["open_hand", "fist", "point"]
colors = ["#2196F3", "#F44336", "#4CAF50"]

# ── GRAPH 1: EMG RMS per gesture ──────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

rms_means = []
rms_stds = []

for gesture in gestures:
    data = df[df["label"] == gesture]["emg_rms"]
    rms_means.append(data.mean())
    rms_stds.append(data.std())

bars1 = ax1.bar(
    gestures,
    rms_means,
    yerr=rms_stds,
    color=colors,
    capsize=8,
    edgecolor="black",
    linewidth=1.2,
)

ax1.set_title("EMG RMS Value per Gesture", fontsize=16, fontweight="bold", pad=15)
ax1.set_xlabel("Gesture", fontsize=13)
ax1.set_ylabel("RMS Value (arbitrary units)", fontsize=13)
ax1.set_xticklabels(["Open Hand", "Fist", "Point"], fontsize=12)

for bar, mean, std in zip(bars1, rms_means, rms_stds):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.5,
        f"{mean:.1f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.set_facecolor("#f9f9f9")

plt.tight_layout()
plt.savefig("graph_emg_rms.png", dpi=150, bbox_inches="tight")
print("Saved: graph_emg_rms.png")
plt.close()

# ── GRAPH 2: EMG MAV per gesture ──────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

mav_means = []
mav_stds = []

for gesture in gestures:
    data = df[df["label"] == gesture]["emg_mav"]
    mav_means.append(data.mean())
    mav_stds.append(data.std())

bars2 = ax2.bar(
    gestures,
    mav_means,
    yerr=mav_stds,
    color=colors,
    capsize=8,
    edgecolor="black",
    linewidth=1.2,
)

ax2.set_title("EMG MAV Value per Gesture", fontsize=16, fontweight="bold", pad=15)
ax2.set_xlabel("Gesture", fontsize=13)
ax2.set_ylabel("MAV Value (arbitrary units)", fontsize=13)
ax2.set_xticklabels(["Open Hand", "Fist", "Point"], fontsize=12)

for bar, mean, std in zip(bars2, mav_means, mav_stds):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.5,
        f"{mean:.1f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.set_facecolor("#f9f9f9")

plt.tight_layout()
plt.savefig("graph_emg_mav.png", dpi=150, bbox_inches="tight")
print("Saved: graph_emg_mav.png")
plt.close()

# ── GRAPH 3: Accuracy Comparison ──────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))

systems = ["EMG Only", "MMG Only", "EMG + MMG\nHybrid"]
accuracies = [65.0, 61.0, 83.3]
colors3 = ["#FF9800", "#9C27B0", "#2196F3"]

bars3 = ax3.bar(
    systems, accuracies, color=colors3, edgecolor="black", linewidth=1.2
)

ax3.set_title(
    "Accuracy Comparison: EMG vs MMG vs Hybrid",
    fontsize=16,
    fontweight="bold",
    pad=15,
)
ax3.set_xlabel("Sensor System", fontsize=13)
ax3.set_ylabel("Classification Accuracy (%)", fontsize=13)
ax3.set_ylim(0, 100)

ax3.axhline(
    y=83.3,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Hybrid accuracy",
)

for bar, acc in zip(bars3, accuracies):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax3.grid(axis="y", linestyle="--", alpha=0.5)
ax3.set_facecolor("#f9f9f9")
ax3.legend(fontsize=11)

plt.tight_layout()
plt.savefig("graph_accuracy_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: graph_accuracy_comparison.png")
plt.close()

# ── GRAPH 4: Confusion Matrix ──────────────────────────
X = df[
    [
        "emg_rms",
        "emg_mav",
        "emg_var",
        "mmg_rms",
        "mmg_var",
        "ax_rms",
        "ay_rms",
        "az_rms",
    ]
].values

y = df["label"].values

scaler_cm = StandardScaler()
X_scaled_cm = scaler_cm.fit_transform(X)

model_cm = SVC(kernel="rbf", C=100, gamma=0.01, class_weight="balanced")
y_pred = cross_val_predict(model_cm, X_scaled_cm, y, cv=5)

cm = confusion_matrix(y, y_pred, labels=["open_hand", "fist", "point"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Open Hand", "Fist", "Point"],
)

disp.plot(cmap="Blues")

plt.title("Gesture Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: confusion_matrix.png")

print("\nAll 4 graphs saved to gesture_project folder")