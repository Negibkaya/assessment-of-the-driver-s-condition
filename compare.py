"""
Сравнение гибридного метода (с калибровкой) vs Baseline CNN (без калибровки)
Запись результатов в CSV для последующего анализа.
Управление:
  [q] - выход
  [1] - показать только гибрид
  [2] - показать только baseline
  [b] - показать оба (side-by-side)
  [s] - пауза/продолжить
  [r] - перекалибровать гибридную систему
"""

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import time
import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

# ============================================================
# ⚙️ НАСТРОЙКИ
# ============================================================
MODEL_HYBRID = "mobilenet_bio_fatigue.onnx"
MODEL_BASELINE = "baseline_cnn.onnx"

CALIBRATION_DURATION = 5.0
MIN_CALIBRATION_SAMPLES = 60

# Базовые пороги (fallback)
EAR_THRESH_DEFAULT = 0.21
MAR_THRESH_DEFAULT = 0.45

HEAD_PITCH_THRESH_REL = 12.0
HEAD_YAW_THRESH_REL = 15.0

MICROSLEEP_TIME_THR = 1.5
NODDING_TIME_THR = 1.0
LONG_YAWN_TIME_THR = 2.5
NO_FACE_TIME_THR = 3.0

SMOOTH_RISE = 0.35
SMOOTH_FALL = 0.08

# Пороги baseline
THRESHOLD_MILD = 0.30
THRESHOLD_DROWSY = 0.50
THRESHOLD_ALERT = 0.85

# CSV
CSV_PATH = "comparison_results.csv"
LOG_INTERVAL = 0.1  # секунд между записями в CSV

# ============================================================
# 🛠 УТИЛИТЫ И СТРУКТУРЫ (общие для гибридной системы)
# ============================================================


class AsymmetricSmoother:
    def __init__(self, rise, fall, start=0.0):
        self.rise = rise
        self.fall = fall
        self.val = start

    def update(self, target):
        if target > self.val:
            self.val = self.rise * target + (1 - self.rise) * self.val
        else:
            self.val = self.fall * target + (1 - self.fall) * self.val
        return max(0.0, min(1.0, self.val))

    def reset(self, start=0.0):
        self.val = start


class SystemState(Enum):
    CALIBRATING = "CALIBRATING"
    ACTIVE = "ACTIVE"
    MILD_FATIGUE = "TIRED"
    DROWSY = "DROWSY"
    ALERT = "!!! WAKE UP !!!"
    DISTRACTED = "NO FACE / DISTRACTED"
    NO_MODEL = "MODEL ERROR"


@dataclass
class CalibrationData:
    ear_samples: List[float] = field(default_factory=list)
    mar_samples: List[float] = field(default_factory=list)
    yaw_samples: List[float] = field(default_factory=list)
    pitch_samples: List[float] = field(default_factory=list)
    face_w_samples: List[float] = field(default_factory=list)
    nn_samples: List[float] = field(default_factory=list)

    ear_thr: float = EAR_THRESH_DEFAULT
    mar_thr: float = MAR_THRESH_DEFAULT
    yaw_base: float = 0.0
    pitch_base: float = 0.0
    nn_offset: float = 0.0
    base_face_w: float = 100.0

    start_time: float = 0.0
    is_done: bool = False

    def add(self, ear, mar, yaw, pitch, fw, nn_p=None):
        if ear > 0.08:
            self.ear_samples.append(ear)
        if mar > 0.01:
            self.mar_samples.append(mar)
        self.yaw_samples.append(yaw)
        self.pitch_samples.append(pitch)
        if fw > 0:
            self.face_w_samples.append(fw)
        if nn_p is not None:
            self.nn_samples.append(nn_p)

    def compute(self):
        if len(self.ear_samples) < MIN_CALIBRATION_SAMPLES:
            return False

        ear_arr = np.array(self.ear_samples)
        ear_med = np.median(ear_arr)
        ear_std = 1.4826 * np.median(np.abs(ear_arr - ear_med))

        candidate1 = ear_med * 0.65
        candidate2 = ear_med - 2.5 * ear_std
        self.ear_thr = max(0.12, min(candidate1, candidate2))

        if self.mar_samples:
            mar_arr = np.array(self.mar_samples)
            mar_med = np.median(mar_arr)
            mar_std = 1.4826 * np.median(np.abs(mar_arr - mar_med))
            self.mar_thr = max(0.25, mar_med + 1.8 * mar_std)
        else:
            self.mar_thr = MAR_THRESH_DEFAULT

        self.yaw_base = float(np.mean(self.yaw_samples))
        self.pitch_base = float(np.mean(self.pitch_samples))

        if self.nn_samples:
            self.nn_offset = float(np.percentile(self.nn_samples, 90))

        if self.face_w_samples:
            self.base_face_w = float(np.median(self.face_w_samples))

        self.is_done = True
        return True

    def get_progress(self):
        t_prog = (time.time() - self.start_time) / CALIBRATION_DURATION
        s_prog = len(self.ear_samples) / MIN_CALIBRATION_SAMPLES
        return min(1.0, min(t_prog, s_prog))

    def get_adapted_ear_thr(self, curr_w):
        if self.base_face_w == 0 or curr_w == 0:
            return self.ear_thr
        ratio = curr_w / self.base_face_w
        if ratio < 0.7:
            return self.ear_thr * 0.85
        return self.ear_thr


@dataclass
class TemporalState:
    close_start: Optional[float] = None
    yawn_start: Optional[float] = None
    nod_start: Optional[float] = None
    no_face_start: Optional[float] = None
    ear_hist: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update_face_status(self, has_face, ts):
        if not has_face:
            if self.no_face_start is None:
                self.no_face_start = ts
            return ts - self.no_face_start
        else:
            self.no_face_start = None
            return 0.0

    def update_events(self, is_closed, is_yawning, is_nodding, ts):
        c_dur = 0.0
        if is_closed:
            if self.close_start is None:
                self.close_start = ts
            c_dur = ts - self.close_start
        else:
            self.close_start = None

        y_dur = 0.0
        if is_yawning:
            if self.yawn_start is None:
                self.yawn_start = ts
            y_dur = ts - self.yawn_start
        else:
            self.yawn_start = None

        n_dur = 0.0
        if is_nodding:
            if self.nod_start is None:
                self.nod_start = ts
            n_dur = ts - self.nod_start
        else:
            self.nod_start = None

        self.ear_hist.append((ts, 1 if is_closed else 0))
        return c_dur, y_dur, n_dur

    def get_perclos(self, window_sec=60):
        if not self.ear_hist:
            return 0.0
        now = self.ear_hist[-1][0]
        valid = [v for t, v in self.ear_hist if now - t <= window_sec]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)


# ============================================================
# 📐 BIO EXTRACTOR (общий для гибридной системы)
# ============================================================

class RobustBiometricExtractor:
    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.R_EYE_V_PAIRS = [(159, 145), (160, 144), (158, 153)]
        self.R_EYE_H = (33, 133)
        self.L_EYE_V_PAIRS = [(386, 374), (385, 380), (387, 373)]
        self.L_EYE_H = (263, 362)

        self.MOUTH_V_PAIRS = [(13, 14), (81, 178),
                              (82, 87), (311, 402), (312, 317)]
        self.FACE_WIDTH_PTS = (234, 454)

    def _calc_ear(self, pts, v_pairs, h_pair):
        vert_distances = []
        for upper_idx, lower_idx in v_pairs:
            dist = np.linalg.norm(pts[upper_idx] - pts[lower_idx])
            vert_distances.append(dist)
        avg_vert = np.mean(vert_distances)
        horiz = np.linalg.norm(pts[h_pair[0]] - pts[h_pair[1]])
        return avg_vert / (horiz + 1e-6)

    def _calc_mar(self, pts, face_width):
        vert_distances = []
        for upper_idx, lower_idx in self.MOUTH_V_PAIRS:
            dist = np.linalg.norm(pts[upper_idx] - pts[lower_idx])
            vert_distances.append(dist)
        avg_vert = np.mean(vert_distances)
        reference = face_width * 0.22
        return avg_vert / (reference + 1e-6)

    def process(self, img):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)

        feats = np.zeros(8, dtype=np.float32)
        if not res.multi_face_landmarks:
            return feats, False, None

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm])

        face_width = np.linalg.norm(
            pts[self.FACE_WIDTH_PTS[0]] - pts[self.FACE_WIDTH_PTS[1]]
        ) + 1e-6

        l_ear = self._calc_ear(pts, self.L_EYE_V_PAIRS, self.L_EYE_H)
        r_ear = self._calc_ear(pts, self.R_EYE_V_PAIRS, self.R_EYE_H)
        ear = (l_ear + r_ear) / 2.0

        mar = self._calc_mar(pts, face_width)

        nose = pts[1]
        eye_mid = (pts[33] + pts[263]) / 2.0
        dx = nose[0] - eye_mid[0]
        dy = nose[1] - eye_mid[1]
        yaw = (dx / face_width) * 100
        pitch = (dy / face_width) * 100

        feats[:] = [l_ear, r_ear, ear, mar, yaw, pitch, 0, 1]

        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        pad_x, pad_y = (xmax - xmin) * 0.2, (ymax - ymin) * 0.2
        bbox = (
            int(max(0, xmin - pad_x)), int(max(0, ymin - pad_y)),
            int(min(w, xmax + pad_x)), int(min(h, ymax + pad_y))
        )

        return feats, True, bbox


# ============================================================
# 🧠 ГИБРИДНАЯ СИСТЕМА (с калибровкой)
# ============================================================

class HybridFatigueSystem:
    def __init__(self, model_path):
        self.calib = CalibrationData()
        self.state = SystemState.CALIBRATING
        self.temporal = TemporalState()
        self.extractor = RobustBiometricExtractor()
        self.smoother = AsymmetricSmoother(SMOOTH_RISE, SMOOTH_FALL)
        self.score_hist = deque(maxlen=200)

        self.use_nn = False
        self.sess = None
        try:
            self.sess = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])
            self.use_nn = True
            print(f"✅ [HYBRID] Model loaded: {model_path}")
        except Exception as e:
            print(f"⚠️ [HYBRID] NN not found: {e}")

    def start_calibration(self):
        self.calib = CalibrationData()
        self.calib.start_time = time.time()
        self.state = SystemState.CALIBRATING
        self.temporal = TemporalState()
        self.smoother.reset()
        print("\n🔄 [HYBRID] Starting calibration...")

    def preprocess_nn(self, crop):
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 -
               [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)

    def process(self, frame):
        ts = time.time()
        feats, has_face, bbox = self.extractor.process(frame)

        res = {
            'has_face': has_face,
            'bbox': bbox,
            'state': self.state,
            'score': 0.0,
            'raw_score': 0.0,
            'metrics': {},
            'calib_prog': 0.0,
            'history': [],
        }

        no_face_dur = self.temporal.update_face_status(has_face, ts)
        if no_face_dur > NO_FACE_TIME_THR and self.state != SystemState.CALIBRATING:
            self.state = SystemState.DISTRACTED
            return res

        if not has_face:
            return res

        ear = float(feats[2])
        mar = float(feats[3])
        yaw = float(feats[4])
        pitch = float(feats[5])
        face_w = bbox[2] - bbox[0] if bbox else 0

        # ========== CALIBRATION ==========
        if self.state == SystemState.CALIBRATING:
            nn_p = 0.0
            if self.use_nn:
                try:
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if crop.size > 0:
                        inp = self.preprocess_nn(crop)
                        bio = np.expand_dims(feats, axis=0).astype(np.float32)
                        out = self.sess.run(None, {
                            self.sess.get_inputs()[0].name: inp,
                            self.sess.get_inputs()[1].name: bio
                        })
                        logits = out[0][0]
                        probs = np.exp(logits - np.max(logits)) / \
                            np.sum(np.exp(logits - np.max(logits)))
                        nn_p = float(probs[1])
                except Exception:
                    pass

            self.calib.add(ear, mar, yaw, pitch, face_w, nn_p)
            prog = self.calib.get_progress()
            res['calib_prog'] = prog

            if prog >= 1.0 and self.calib.compute():
                self.state = SystemState.ACTIVE
                print(f"\n✅ [HYBRID] Calibration complete!")
                print(f"   EAR threshold: {self.calib.ear_thr:.3f}")
                print(f"   MAR threshold: {self.calib.mar_thr:.3f}")
                print(f"   Pitch base: {self.calib.pitch_base:.1f}")

            res['metrics'] = {'ear': ear, 'mar': mar, 'pitch': pitch}
            return res

        # ========== MONITORING ==========
        curr_ear_thr = self.calib.get_adapted_ear_thr(face_w)
        is_closed = ear < curr_ear_thr
        is_yawning = mar > self.calib.mar_thr
        pitch_delta = pitch - self.calib.pitch_base
        is_nodding = pitch_delta > HEAD_PITCH_THRESH_REL

        c_dur, y_dur, n_dur = self.temporal.update_events(
            is_closed, is_yawning, is_nodding, ts)

        # ========== NN SCORE ==========
        nn_score = 0.0
        if self.use_nn:
            try:
                crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if crop.size > 0:
                    inp = self.preprocess_nn(crop)
                    bio = np.expand_dims(feats, axis=0).astype(np.float32)
                    out = self.sess.run(None, {
                        self.sess.get_inputs()[0].name: inp,
                        self.sess.get_inputs()[1].name: bio
                    })
                    logits = out[0][0]
                    probs = np.exp(logits - np.max(logits)) / \
                        np.sum(np.exp(logits - np.max(logits)))
                    raw = float(probs[1])
                    nn_score = max(0.0, (raw - self.calib.nn_offset) / (
                        self.calib.nn_offset + 1e-6))
            except Exception:
                pass

        # ========== BIO SCORE ==========
        bio_score = 0.0
        if is_closed:
            bio_score += 0.5
        if is_yawning:
            bio_score += 0.4
        if is_nodding:
            bio_score += 0.4
        if y_dur > LONG_YAWN_TIME_THR:
            bio_score += 0.4

        perclos = self.temporal.get_perclos()
        temp_score = 0.5 if perclos > 0.15 else 0.0

        final_raw = 0.35 * nn_score + 0.40 * bio_score + 0.25 * temp_score

        override_active = False
        if c_dur > MICROSLEEP_TIME_THR:
            final_raw = 1.0
            override_active = True
        if n_dur > NODDING_TIME_THR:
            final_raw = 1.0
            override_active = True

        if override_active:
            self.smoother.val = max(self.smoother.val, 0.95)
            smooth_score = 1.0
        else:
            smooth_score = self.smoother.update(final_raw)

        self.score_hist.append(smooth_score)
        res['raw_score'] = final_raw
        res['score'] = smooth_score

        if smooth_score > 0.85:
            self.state = SystemState.ALERT
        elif smooth_score > 0.50:
            self.state = SystemState.DROWSY
        elif smooth_score > 0.30:
            self.state = SystemState.MILD_FATIGUE
        else:
            self.state = SystemState.ACTIVE

        res['state'] = self.state
        res['metrics'] = {
            'ear': ear, 'ear_thr': curr_ear_thr,
            'mar': mar, 'mar_thr': self.calib.mar_thr,
            'pitch': pitch, 'pitch_base': self.calib.pitch_base,
            'perclos': perclos,
            'c_dur': c_dur, 'y_dur': y_dur, 'n_dur': n_dur,
            'is_closed': is_closed, 'is_yawning': is_yawning,
            'nn_score': nn_score, 'bio_score': bio_score,
        }
        res['history'] = list(self.score_hist)
        return res


# ============================================================
# 🧠 BASELINE СИСТЕМА (только CNN, без калибровки)
# ============================================================

class BaselineFatigueSystem:
    def __init__(self, model_path):
        self.model_loaded = False
        self.session = None
        self.input_name = None

        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.model_loaded = True
            print(f"✅ [BASELINE] Model loaded: {model_path}")
            print(f"   Input: {self.input_name}")
            print(f"   Provider: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"❌ [BASELINE] Failed to load model: {e}")

        self.smoother = AsymmetricSmoother(SMOOTH_RISE, SMOOTH_FALL)
        self.score_history = deque(maxlen=200)
        self.prev_time = time.time()
        self.fps = 0.0

    def preprocess(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)

    def process(self, frame):
        result = {
            'state': SystemState.NO_MODEL,
            'score': 0.0,
            'raw_score': 0.0,
            'confidence': 0.0,
            'fps': 0.0,
            'history': [],
        }

        curr_time = time.time()
        self.fps = 0.9 * self.fps + 0.1 * \
            (1.0 / (curr_time - self.prev_time + 1e-6))
        self.prev_time = curr_time
        result['fps'] = self.fps

        if not self.model_loaded:
            return result

        try:
            input_tensor = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0][0]

            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            raw_fatigue_prob = float(probs[1])
            result['raw_score'] = raw_fatigue_prob
            result['confidence'] = float(np.max(probs))

            smooth_score = self.smoother.update(raw_fatigue_prob)
            result['score'] = smooth_score

            self.score_history.append(smooth_score)
            result['history'] = list(self.score_history)

            if smooth_score > THRESHOLD_ALERT:
                result['state'] = SystemState.ALERT
            elif smooth_score > THRESHOLD_DROWSY:
                result['state'] = SystemState.DROWSY
            elif smooth_score > THRESHOLD_MILD:
                result['state'] = SystemState.MILD_FATIGUE
            else:
                result['state'] = SystemState.ACTIVE

        except Exception as e:
            print(f"⚠️ [BASELINE] Inference error: {e}")
            result['state'] = SystemState.NO_MODEL

        return result


# ============================================================
# 🎨 UI
# ============================================================

class ComparisonUI:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.alert_phase = 0.0

    def get_state_color(self, state):
        colors = {
            SystemState.ACTIVE: (0, 255, 0),
            SystemState.MILD_FATIGUE: (0, 255, 255),
            SystemState.DROWSY: (0, 165, 255),
            SystemState.ALERT: (0, 0, 255),
            SystemState.DISTRACTED: (0, 165, 255),
            SystemState.CALIBRATING: (255, 255, 0),
            SystemState.NO_MODEL: (128, 128, 128),
        }
        return colors.get(state, (255, 255, 255))

    def draw_hybrid(self, frame, res, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        state = res['state']
        score = res['score']
        color = self.get_state_color(state)
        met = res.get('metrics', {})

        # Alert overlay
        if state in [SystemState.ALERT, SystemState.DISTRACTED]:
            self.alert_phase += 0.15
            alpha = 0.3 + 0.2 * abs(np.sin(self.alert_phase))
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Header
        cv2.rectangle(overlay, (0, 0), (w, 55), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        cv2.putText(frame, "HYBRID (Calibrated)", (10, 20),
                    self.font, 0.55, (100, 200, 100), 1)
        cv2.putText(frame, state.value, (10, 45), self.font, 0.75, color, 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 80, 20),
                    self.font, 0.45, (150, 150, 150), 1)
        cv2.putText(frame, f"Score: {score:.0%}",
                    (w - 130, 45), self.font, 0.5, color, 1)

        # Calibration bar
        if state == SystemState.CALIBRATING:
            prog = res.get('calib_prog', 0)
            cx, cy = w // 2, h // 2
            bw = 300
            cv2.rectangle(frame, (cx - bw // 2, cy),
                          (cx + bw // 2, cy + 20), (100, 100, 100), 2)
            cv2.rectangle(frame, (cx - bw // 2, cy),
                          (int(cx - bw // 2 + bw * prog), cy + 20), (0, 255, 0), -1)
            cv2.putText(frame, f"CALIBRATING... {prog:.0%}",
                        (cx - 140, cy - 10), self.font, 0.7, (255, 255, 255), 1)
            cv2.putText(frame,
                        f"EAR: {met.get('ear', 0):.3f}  MAR: {met.get('mar', 0):.3f}",
                        (cx - 120, cy + 40), self.font, 0.45, (150, 150, 150), 1)
            return frame

        # Bbox
        bbox = res.get('bbox')
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{state.value} {int(score * 100)}%",
                        (bbox[0], bbox[1] - 8), self.font, 0.5, color, 1)

        # Metrics panel
        ph, pw = 200, 280
        px, py = 10, h - ph - 10
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (80, 80, 80), 1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        ear_thr = met.get('ear_thr', EAR_THRESH_DEFAULT)
        mar_thr = met.get('mar_thr', MAR_THRESH_DEFAULT)
        ear = met.get('ear', 0)
        mar = met.get('mar', 0)
        perclos = met.get('perclos', 0)

        # EAR bar
        ear_c = (0, 255, 0) if ear > ear_thr else (0, 0, 255)
        self._draw_bar(frame, px + 10, py + 25, 200, 10, ear, 0.45, ear_c,
                       f"EAR ({ear_thr:.2f})", ear_thr)

        # MAR bar
        mar_c = (0, 255, 0) if mar < mar_thr else (0, 0, 255)
        self._draw_bar(frame, px + 10, py + 60, 200, 10, mar, 0.80, mar_c,
                       f"MAR ({mar_thr:.2f})", mar_thr)

        # Pitch bar
        pitch = met.get('pitch', 0) - met.get('pitch_base', 0)
        pit_c = (0, 255, 0) if abs(
            pitch) < HEAD_PITCH_THRESH_REL else (0, 0, 255)
        self._draw_bar(frame, px + 10, py + 95, 200, 10, abs(pitch), 25.0, pit_c,
                       f"Pitch Δ ({HEAD_PITCH_THRESH_REL:.0f})", HEAD_PITCH_THRESH_REL)

        # Debug lines
        lines = [
            f"Closed: {met.get('c_dur', 0):.1f}s",
            f"Yawn:   {met.get('y_dur', 0):.1f}s",
            f"Nod:    {met.get('n_dur', 0):.1f}s",
            f"PERCLOS:{perclos * 100:.0f}%",
            f"NN: {met.get('nn_score', 0):.2f}  Bio: {met.get('bio_score', 0):.2f}",
        ]
        for i, l in enumerate(lines):
            cv2.putText(frame, l, (px + 10, py + 130 + i * 16),
                        self.font, 0.4, (180, 180, 180), 1)

        # History graph
        hist = res.get('history', [])
        if len(hist) > 1:
            gx, gy = w - 230, h - 120
            gw, gh = 220, 100
            cv2.rectangle(overlay, (gx, gy),
                          (gx + gw, gy + gh), (10, 10, 10), -1)
            cv2.rectangle(overlay, (gx, gy),
                          (gx + gw, gy + gh), (60, 60, 60), 1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            for thresh, c in [(THRESHOLD_MILD, (100, 255, 100)),
                              (THRESHOLD_DROWSY, (100, 165, 100)),
                              (THRESHOLD_ALERT, (100, 100, 255))]:
                ty = int(gy + gh * (1 - thresh))
                cv2.line(frame, (gx, ty), (gx + gw, ty), c, 1)
            pts = []
            for i, v in enumerate(hist):
                px_ = int(gx + (i / 200) * gw)
                py_ = int(gy + gh * (1 - v))
                pts.append([px_, py_])
            if len(pts) > 1:
                cv2.polylines(
                    frame, [np.array(pts, np.int32)], False, (0, 140, 255), 2)
            cv2.putText(frame, "FATIGUE LEVEL", (gx, gy - 8),
                        self.font, 0.4, (200, 200, 200), 1)

        return frame

    def draw_baseline(self, frame, res, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        state = res['state']
        score = res['score']
        color = self.get_state_color(state)

        # Alert overlay
        if state == SystemState.ALERT:
            self.alert_phase += 0.15
            alpha = 0.3 + 0.2 * abs(np.sin(self.alert_phase))
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Header
        cv2.rectangle(overlay, (0, 0), (w, 55), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        cv2.putText(frame, "BASELINE (CNN Only)", (10, 20),
                    self.font, 0.55, (200, 100, 100), 1)
        cv2.putText(frame, state.value, (10, 45), self.font, 0.75, color, 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 80, 20),
                    self.font, 0.45, (150, 150, 150), 1)
        cv2.putText(frame, f"Score: {score:.0%}",
                    (w - 130, 45), self.font, 0.5, color, 1)

        # Score bar
        bar_y = h - 45
        bar_h = 22
        bw = w - 40
        bx = 20
        cv2.rectangle(overlay, (bx, bar_y),
                      (bx + bw, bar_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(overlay, (bx, bar_y),
                      (bx + bw, bar_y + bar_h), (80, 80, 80), 1)
        fill_w = int(bw * score)
        if fill_w > 0:
            if score < 0.3:
                bc = (0, 255, 0)
            elif score < 0.5:
                bc = (0, 255, 255)
            elif score < 0.85:
                bc = (0, 165, 255)
            else:
                bc = (0, 0, 255)
            cv2.rectangle(overlay, (bx, bar_y),
                          (bx + fill_w, bar_y + bar_h), bc, -1)
        for thresh in [THRESHOLD_MILD, THRESHOLD_DROWSY, THRESHOLD_ALERT]:
            tx = bx + int(bw * thresh)
            cv2.line(frame, (tx, bar_y),
                     (tx, bar_y + bar_h), (255, 255, 255), 1)
        cv2.putText(frame, f"{int(score * 100)}%", (bx + bw + 8, bar_y + bar_h - 3),
                    self.font, 0.5, (255, 255, 255), 1)

        # History graph
        hist = res.get('history', [])
        if len(hist) > 1:
            gw, gh = 220, 90
            gx, gy = w - gw - 20, h - gh - 60
            cv2.rectangle(overlay, (gx, gy),
                          (gx + gw, gy + gh), (20, 20, 20), -1)
            cv2.rectangle(overlay, (gx, gy),
                          (gx + gw, gy + gh), (60, 60, 60), 1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            for thresh in [THRESHOLD_MILD, THRESHOLD_DROWSY, THRESHOLD_ALERT]:
                ty = int(gy + gh * (1 - thresh))
                cv2.line(frame, (gx, ty), (gx + gw, ty), (60, 60, 60), 1)
            pts = []
            for i, v in enumerate(hist):
                px_ = int(gx + (i / 200) * gw)
                py_ = int(gy + gh * (1 - v))
                pts.append([px_, py_])
            if len(pts) > 1:
                cv2.polylines(
                    frame, [np.array(pts, np.int32)], False, (0, 140, 255), 2)
            cv2.putText(frame, "FATIGUE HISTORY", (gx, gy - 8),
                        self.font, 0.4, (150, 150, 150), 1)

        # Raw vs Smooth
        cv2.putText(frame, f"Raw: {res.get('raw_score', 0):.2f}  Smooth: {score:.2f}",
                    (20, h - 55), self.font, 0.4, (150, 150, 150), 1)

        return frame

    def _draw_bar(self, frame, x, y, w, h, val, max_val, color, title, thresh=None):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
        ratio = min(1.0, max(0.0, val / max_val))
        cv2.rectangle(frame, (x, y), (x + int(w * ratio), y + h), color, -1)
        if thresh is not None:
            tx = int((thresh / max_val) * w)
            if 0 < tx < w:
                cv2.line(frame, (x + tx, y - 2),
                         (x + tx, y + h + 2), (255, 255, 255), 2)
        cv2.putText(frame, title, (x, y - 5),
                    self.font, 0.4, (200, 200, 200), 1)


# ============================================================
# 📊 CSV ЛОГГЕР
# ============================================================

class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.file = None
        self.writer = None
        self.last_log_time = 0.0
        self._init()

    def _init(self):
        self.file = open(self.path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp_s',
            'frame_idx',
            # Hybrid
            'hybrid_state', 'hybrid_score', 'hybrid_raw_score',
            'hybrid_ear', 'hybrid_ear_thr', 'hybrid_mar', 'hybrid_mar_thr',
            'hybrid_perclos', 'hybrid_c_dur', 'hybrid_y_dur', 'hybrid_n_dur',
            'hybrid_nn_score', 'hybrid_bio_score',
            'hybrid_has_face', 'hybrid_calib_prog',
            # Baseline
            'baseline_state', 'baseline_score', 'baseline_raw_score',
            'baseline_confidence', 'baseline_has_face',
        ])
        self.file.flush()

    def log(self, frame_idx, hybrid_res, baseline_res):
        now = time.time()
        if now - self.last_log_time < LOG_INTERVAL:
            return
        self.last_log_time = now

        h = hybrid_res
        b = baseline_res

        def state_str(s):
            try:
                return s.value
            except AttributeError:
                return str(s)

        self.writer.writerow([
            f"{now:.3f}",
            frame_idx,
            # Hybrid
            state_str(h.get('state', '')),
            f"{h.get('score', 0):.4f}",
            f"{h.get('raw_score', 0):.4f}",
            f"{h.get('metrics', {}).get('ear', 0):.4f}",
            f"{h.get('metrics', {}).get('ear_thr', 0):.4f}",
            f"{h.get('metrics', {}).get('mar', 0):.4f}",
            f"{h.get('metrics', {}).get('mar_thr', 0):.4f}",
            f"{h.get('metrics', {}).get('perclos', 0):.4f}",
            f"{h.get('metrics', {}).get('c_dur', 0):.2f}",
            f"{h.get('metrics', {}).get('y_dur', 0):.2f}",
            f"{h.get('metrics', {}).get('n_dur', 0):.2f}",
            f"{h.get('metrics', {}).get('nn_score', 0):.4f}",
            f"{h.get('metrics', {}).get('bio_score', 0):.4f}",
            '1' if h.get('has_face', False) else '0',
            f"{h.get('calib_prog', 0):.3f}",
            # Baseline
            state_str(b.get('state', '')),
            f"{b.get('score', 0):.4f}",
            f"{b.get('raw_score', 0):.4f}",
            f"{b.get('confidence', 0):.4f}",
            '1' if b.get('has_face', True) else '0',
        ])
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            print(f"\n📊 CSV saved: {os.path.abspath(self.path)}")


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    print("=" * 55)
    print(" COMPARISON: Hybrid (Calibrated) vs Baseline (CNN Only)")
    print("=" * 55)
    print(" Controls:")
    print("  [q] Quit")
    print("  [1] Show Hybrid only")
    print("  [2] Show Baseline only")
    print("  [b] Show Both (side-by-side)")
    print("  [s] Pause / Resume")
    print("  [r] Recalibrate Hybrid")
    print("=" * 55)

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Systems
    hybrid = HybridFatigueSystem(MODEL_HYBRID)
    baseline = BaselineFatigueSystem(MODEL_BASELINE)
    ui = ComparisonUI()
    logger = CSVLogger(CSV_PATH)

    # State
    paused = False
    show_mode = 'both'  # 'hybrid', 'baseline', 'both'
    frame_idx = 0
    hybrid_fps = 0.0
    baseline_fps = 0.0

    hybrid.start_calibration()

    print("\n🚀 Starting comparison...\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
            frame = cv2.flip(frame, 1)

            # Process both systems
            hybrid_res = hybrid.process(frame)
            baseline_res = baseline.process(frame)

            hybrid_fps = hybrid_fps * 0.9 + \
                (1.0 / max(time.time() - baseline.prev_time + 1e-6, 1e-6)) * 0.1
            baseline_fps = baseline.fps

            # Log to CSV
            logger.log(frame_idx, hybrid_res, baseline_res)
            frame_idx += 1

        # Determine what to show
        if show_mode == 'hybrid':
            view = ui.draw_hybrid(frame.copy(), hybrid_res, hybrid_fps)
            label = "HYBRID ONLY"
        elif show_mode == 'baseline':
            view = ui.draw_baseline(frame.copy(), baseline_res, baseline_fps)
            label = "BASELINE ONLY"
        else:
            # Side-by-side
            h_img = ui.draw_hybrid(frame.copy(), hybrid_res, hybrid_fps)
            b_img = ui.draw_baseline(frame.copy(), baseline_res, baseline_fps)
            # Resize to same height
            target_h = 480
            h_img = cv2.resize(
                h_img, (int(target_h * h_img.shape[1] / h_img.shape[0]), target_h))
            b_img = cv2.resize(
                b_img, (int(target_h * b_img.shape[1] / b_img.shape[0]), target_h))
            # Pad to same width
            max_w = max(h_img.shape[1], b_img.shape[1])
            if h_img.shape[1] < max_w:
                pad = np.zeros(
                    (h_img.shape[0], max_w - h_img.shape[1], 3), dtype=np.uint8)
                h_img = np.hstack([h_img, pad])
            if b_img.shape[1] < max_w:
                pad = np.zeros(
                    (b_img.shape[0], max_w - b_img.shape[1], 3), dtype=np.uint8)
                b_img = np.hstack([b_img, pad])
            view = np.hstack([h_img, b_img])
            # Divider line
            cv2.line(view, (view.shape[1] // 2, 0),
                     (view.shape[1] // 2, view.shape[0]), (255, 255, 255), 2)
            # Labels
            cv2.putText(view, "HYBRID (Calibrated)", (10, 25),
                        ui.font, 0.6, (100, 255, 100), 1)
            cv2.putText(view, "BASELINE (CNN Only)",
                        (view.shape[1] // 2 + 10, 25), ui.font, 0.6, (255, 100, 100), 1)

        # Pause indicator
        if paused:
            cv2.putText(view, "PAUSED", (view.shape[1] // 2 - 60, view.shape[0] // 2),
                        ui.font, 1.2, (0, 255, 255), 2)

        # Mode indicator
        mode_txt = f"Mode: {show_mode.upper()}  |  Frame: {frame_idx}"
        cv2.putText(view, mode_txt, (10, view.shape[0] - 10),
                    ui.font, 0.45, (200, 200, 200), 1)

        cv2.imshow("Comparison: Hybrid vs Baseline", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            show_mode = 'hybrid'
            print("  → Showing HYBRID only")
        elif key == ord('2'):
            show_mode = 'baseline'
            print("  → Showing BASELINE only")
        elif key == ord('b'):
            show_mode = 'both'
            print("  → Showing BOTH (side-by-side)")
        elif key == ord('s'):
            paused = not paused
            print(f"  → {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('r'):
            hybrid.start_calibration()
            print("  → Hybrid recalibration started")

    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print("\n👋 Done!")


if __name__ == "__main__":
    main()
