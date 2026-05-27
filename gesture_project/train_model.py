import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

import joblib


def train():

    df = pd.read_csv('gesture_data.csv')

    # Full hybrid features (EMG + MMG combined)
    X_full = df[
        [
            'emg_rms', 'emg_mav', 'emg_var',
            'mmg_rms', 'mmg_var', 'ax_rms',
            'ay_rms', 'az_rms'
        ]
    ].values

    # EMG only features
    X_emg = df[
        ['emg_rms', 'emg_mav', 'emg_var']
    ].values

    # MMG only features
    X_mmg = df[
        [
            'mmg_rms', 'mmg_var',
            'ax_rms', 'ay_rms', 'az_rms'
        ]
    ].values

    y = df['label'].values

    # Scale each feature set separately
    scaler_full = StandardScaler()
    scaler_emg  = StandardScaler()
    scaler_mmg  = StandardScaler()

    X_full_scaled = scaler_full.fit_transform(X_full)
    X_emg_scaled  = scaler_emg.fit_transform(X_emg)
    X_mmg_scaled  = scaler_mmg.fit_transform(X_mmg)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Same classifier for fair comparison
    model_full = SVC(
        kernel='rbf',
        C=100,
        gamma='scale',
        probability=True,
        class_weight='balanced'
    )

    model_emg = SVC(
        kernel='rbf',
        C=100,
        gamma=0.01,
        probability=True,
        class_weight='balanced'
    )

    model_mmg = SVC(
        kernel='rbf',
        C=100,
        gamma=0.01,
        probability=True,
        class_weight='balanced'
    )

    scores_full = cross_val_score(model_full, X_full_scaled, y, cv=cv)
    scores_emg  = cross_val_score(model_emg,  X_emg_scaled,  y, cv=cv)
    scores_mmg  = cross_val_score(model_mmg,  X_mmg_scaled,  y, cv=cv)

    print("\n=== Accuracy Comparison ===")
    print(f"EMG only:         {scores_emg.mean() * 100:.1f}%")
    print(f"MMG only:         {scores_mmg.mean() * 100:.1f}%")
    print(f"EMG + MMG hybrid: {scores_full.mean() * 100:.1f}%")

    print(
        f"\nHybrid improvement over EMG alone: "
        f"{(scores_full.mean() - scores_emg.mean()) * 100:.1f} percentage points"
    )

    print(
        f"Hybrid improvement over MMG alone: "
        f"{(scores_full.mean() - scores_mmg.mean()) * 100:.1f} percentage points"
    )

    # Save hybrid model for realtime display
    model_full.fit(X_full_scaled, y)

    joblib.dump(model_full,  'gesture_model.pkl')
    joblib.dump(scaler_full, 'gesture_scaler.pkl')

    print("\nHybrid model saved!")

    # Show samples per gesture
    print("\n=== Samples per gesture ===")

    for gesture in ['open_hand', 'fist', 'point']:
        count = list(y).count(gesture)
        print(f"{gesture}: {count} samples")


if __name__ == '__main__':
    train()