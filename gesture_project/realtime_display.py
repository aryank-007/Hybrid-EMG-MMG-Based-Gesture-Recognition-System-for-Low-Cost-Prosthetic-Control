import serial
import numpy as np
import joblib
import threading
import time
import os
import tkinter as tk
from PIL import Image, ImageTk

PORT        = '/dev/cu.usbmodem21301'
BAUD_RATE   = 115200
WINDOW_SIZE = 100
CONFIDENCE_THRESHOLD = 0.40

GESTURE_IMAGES = {
    'open_hand': 'images/open_hand.jpg',
    'fist':      'images/fist.jpg',
    'point':     'images/point.jpg',
}


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

    return [[emg_rms, emg_mav, emg_var, mmg_rms, mmg_var, ax_rms, ay_rms, az_rms]]


class GestureApp:
    def __init__(self):
        self.model  = joblib.load('gesture_model.pkl')
        self.scaler = joblib.load('gesture_scaler.pkl')

        self.root = tk.Tk()
        self.root.title("Gesture Recognition — EMG + MMG")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry('640x620')
        self.root.resizable(False, False)

        # Countdown label
        self.countdown_label = tk.Label(
            self.root, text="Get Ready...",
            font=('Helvetica', 22), fg='#ffcc00', bg='#1a1a2e'
        )
        self.countdown_label.pack(pady=5)

        # Gesture name label
        self.name_label = tk.Label(
            self.root, text="",
            font=('Helvetica', 28, 'bold'), fg='white', bg='#1a1a2e'
        )
        self.name_label.pack(pady=5)

        # Image display
        self.img_label = tk.Label(self.root, bg='#1a1a2e')
        self.img_label.pack()

        # Confidence label
        self.conf_label = tk.Label(
            self.root, text="", font=('Helvetica', 16), bg='#1a1a2e', fg='#00ff88'
        )
        self.conf_label.pack(pady=5)

        # Load images
        self.tk_images = {}
        for name, path in GESTURE_IMAGES.items():
            if os.path.exists(path):
                img = Image.open(path).resize((480, 300), Image.LANCZOS)
                self.tk_images[name] = ImageTk.PhotoImage(img)
            else:
                print(f"WARNING: Image not found: {path}")

        # Serial connection
        self.ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        self.ser.flushInput()

        self.emg_buf, self.ax_buf, self.ay_buf, self.az_buf = [], [], [], []
        self.running      = True
        self.collecting   = False
        self.countdown_on = False

        # Start serial reading thread
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

        # Start countdown cycle
        self.root.after(1000, self._start_countdown)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _start_countdown(self):
        self.collecting = False
        self.emg_buf, self.ax_buf, self.ay_buf, self.az_buf = [], [], [], []
        self._do_countdown(3)

    def _do_countdown(self, count):
        if count > 0:
            self.countdown_label.config(
                text=f"Get ready... {count}",
                fg='#ffcc00'
            )
            self.name_label.config(text="")
            self.conf_label.config(text="")
            self.root.after(1000, self._do_countdown, count - 1)
        else:
            self.countdown_label.config(
                text="HOLD YOUR GESTURE NOW!",
                fg='#00ff88'
            )
            self.collecting = True
            # Collect for 2 seconds then predict
            self.root.after(2000, self._predict_gesture)

    def _predict_gesture(self):
        self.collecting = False
        self.countdown_label.config(text="Predicting...", fg='#ffffff')

        if len(self.emg_buf) >= WINDOW_SIZE:
            feats        = extract_features(
                self.emg_buf[-WINDOW_SIZE:],
                self.ax_buf[-WINDOW_SIZE:],
                self.ay_buf[-WINDOW_SIZE:],
                self.az_buf[-WINDOW_SIZE:]
            )
            feats_scaled = self.scaler.transform(feats)
            gesture      = self.model.predict(feats_scaled)[0]
            probas       = self.model.predict_proba(feats_scaled)[0]
            confidence   = float(max(probas))

            self._update_ui(gesture, confidence)
        else:
            self.countdown_label.config(text="Not enough data, retrying...", fg='#ff6644')

        # Wait 3 seconds showing result then start again
        self.root.after(3000, self._start_countdown)

    def _read_loop(self):
        while self.running:
            try:
                line  = self.ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                if len(parts) != 4:
                    continue

                if self.collecting:
                    self.emg_buf.append(int(parts[0]))
                    self.ax_buf.append(int(parts[1]))
                    self.ay_buf.append(int(parts[2]))
                    self.az_buf.append(int(parts[3]))

            except (ValueError, serial.SerialException):
                continue

    def _update_ui(self, gesture, confidence):
        if confidence < CONFIDENCE_THRESHOLD:
            self.name_label.config(text="Not sure...", fg='#888888')
            self.conf_label.config(text=f"Confidence: {confidence:.0%}")
            return

        display_name = gesture.replace('_', ' ').upper()
        self.name_label.config(text=display_name, fg='white')

        if gesture in self.tk_images:
            self.img_label.config(image=self.tk_images[gesture])

        bar_color = '#00ff88' if confidence > 0.85 else '#ffcc00' if confidence > 0.70 else '#ff6644'
        self.conf_label.config(text=f"Confidence: {confidence:.0%}", fg=bar_color)

    def _on_close(self):
        self.running = False
        self.ser.close()
        self.root.destroy()


if __name__ == '__main__':
    GestureApp()