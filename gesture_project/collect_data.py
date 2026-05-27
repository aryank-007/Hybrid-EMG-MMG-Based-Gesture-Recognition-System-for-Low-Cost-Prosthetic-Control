import serial
import numpy as np
import csv
import time

PORT        = '/dev/cu.usbmodem21301'
BAUD_RATE   = 115200
WINDOW_SIZE = 100
GESTURES    = ['open_hand', 'fist', 'point']
SAMPLES     = 50
OUTPUT_FILE = 'gesture_data.csv'


def extract_features(emg, ax, ay, az):
    emg = np.array(emg, dtype=float)
    ax  = np.array(ax,  dtype=float)
    ay  = np.array(ay,  dtype=float)
    az  = np.array(az,  dtype=float)

    emg_rms  = np.sqrt(np.mean(emg**2))
    emg_mav  = np.mean(np.abs(emg))
    emg_var  = np.var(emg)
    mag      = np.sqrt(ax**2 + ay**2 + az**2)
    mag_ac   = mag - np.mean(mag)
    mmg_rms  = np.sqrt(np.mean(mag_ac**2))
    mmg_var  = np.var(mag_ac)
    ax_rms   = np.sqrt(np.mean(ax**2))
    ay_rms   = np.sqrt(np.mean(ay**2))
    az_rms   = np.sqrt(np.mean(az**2))

    return [emg_rms, emg_mav, emg_var, mmg_rms, mmg_var, ax_rms, ay_rms, az_rms]


def collect():
    ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    ser.flushInput()

    rows = []
    print("\n=== EMG + MMG Gesture Data Collection ===\n")

    for gesture in GESTURES:
        print(f"\n>>> Gesture: {gesture.upper()}")
        input(f"    Press ENTER when ready to record '{gesture}'...")

        for i in range(SAMPLES):
            print(f"  Sample {i+1}/{SAMPLES}  — Get ready (2s)...", end='\r')
            time.sleep(2)
            print(f"  Sample {i+1}/{SAMPLES}  — HOLD THE GESTURE NOW!     ")

            emg_buf, ax_buf, ay_buf, az_buf = [], [], [], []
            while len(emg_buf) < WINDOW_SIZE:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    parts = line.split(',')
                    if len(parts) == 4:
                        emg_buf.append(int(parts[0]))
                        ax_buf.append(int(parts[1]))
                        ay_buf.append(int(parts[2]))
                        az_buf.append(int(parts[3]))
                except (ValueError, UnicodeDecodeError):
                    continue

            feats = extract_features(emg_buf, ax_buf, ay_buf, az_buf)
            rows.append(feats + [gesture])
            print(f"    EMG_RMS={feats[0]:.1f}  MMG_RMS={feats[3]:.1f}  recorded")
            time.sleep(0.3)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['emg_rms','emg_mav','emg_var',
                         'mmg_rms','mmg_var','ax_rms','ay_rms','az_rms','label'])
        writer.writerows(rows)

    ser.close()
    print(f"\nData saved to '{OUTPUT_FILE}'")
    print(f"Total samples: {len(rows)}")


if __name__ == '__main__':
    collect()
