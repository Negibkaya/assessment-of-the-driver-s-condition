import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import time
import winsound
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# НАСТРОЙКИ (CONFIG)
# ==========================================
MODEL_PATH = "mobilenet_bio_fatigue.onnx"

CALIBRATION_DURATION = 5.0
MIN_CALIBRATION_SAMPLES = 60

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

FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"


# ==========================================
# ЗВУКОВОЙ АЛЕРТ
# ==========================================

def play_alert_sound():
    """Неблокирующий звуковой сигнал (1000 Гц, 500 мс)"""
    try:
        winsound.Beep(1000, 500)
    except Exception:
        pass


class AlertSoundManager:
    """Управление повторяющимся звуковым сигналом при ALERT"""

    def __init__(self, interval=2.0):
        self.interval = interval
        self._was_alert = False
        self._timer = None
        self._stop = False

    def _beep_loop(self):
        if self._stop:
            return
        play_alert_sound()
        self._timer = threading.Timer(self.interval, self._beep_loop)
        self._timer.daemon = True
        self._timer.start()

    def update(self, is_alert):
        if is_alert and not self._was_alert:
            self._was_alert = True
            self._stop = False
            play_alert_sound()
            self._beep_loop()
        elif not is_alert and self._was_alert:
            self._was_alert = False
            self._stop = True
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def stop(self):
        self._stop = True
        if self._timer:
            self._timer.cancel()
            self._timer = None


# ==========================================
# УТИЛИТЫ И СТРУКТУРЫ
# ==========================================

def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


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


class SystemState(Enum):
    CALIBRATING = "КАЛИБРОВКА"
    ACTIVE = "АКТИВЕН"
    MILD_FATIGUE = "УСТАЛ"
    DROWSY = "СОНЛИВОСТЬ"
    ALERT = "ВНИМАНИЕ!"
    DISTRACTED = "НЕТ ЛИЦА"


@dataclass
class CalibrationData:
    ear_samples: List[float] = field(default_factory=list)
    mar_samples: List[float] = field(default_factory=list)
    yaw_samples: List[float] = field(default_factory=list)
    pitch_samples: List[float] = field(default_factory=list)
    face_w_samples: List[float] = field(default_factory=list)
    nn_samples: List[float] = field(default_factory=list)

    ear_thr: float = 0.21
    mar_thr: float = 0.45
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

        print(f"   EAR: median={ear_med:.3f}, std={ear_std:.3f}")
        print(f"        candidate1 (65%)={candidate1:.3f}, candidate2 (med-2.5std)={candidate2:.3f}")
        print(f"        -> threshold={self.ear_thr:.3f}")

        if self.mar_samples:
            mar_arr = np.array(self.mar_samples)
            mar_med = np.median(mar_arr)
            mar_std = 1.4826 * np.median(np.abs(mar_arr - mar_med))
            self.mar_thr = max(0.25, mar_med + 1.8 * mar_std)
            print(f"   MAR: median={mar_med:.3f}, std={mar_std:.3f}")
            print(f"        -> threshold={self.mar_thr:.3f}")
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


# ==========================================
# BIO EXTRACTOR
# ==========================================

class RobustBiometricExtractor:
    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.R_EYE_V_PAIRS = [
            (159, 145),
            (160, 144),
            (158, 153),
        ]
        self.R_EYE_H = (33, 133)

        self.L_EYE_V_PAIRS = [
            (386, 374),
            (385, 380),
            (387, 373),
        ]
        self.L_EYE_H = (263, 362)

        self.MOUTH_V_PAIRS = [
            (13, 14),
            (81, 178),
            (82, 87),
            (311, 402),
            (312, 317),
        ]

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
        bbox = (int(max(0, xmin - pad_x)), int(max(0, ymin - pad_y)),
                int(min(w, xmax + pad_x)), int(min(h, ymax + pad_y)))

        return feats, True, bbox


# ==========================================
# CORE LOGIC
# ==========================================

class FatigueSystem:
    def __init__(self):
        self.calib = CalibrationData()
        self.state = SystemState.CALIBRATING
        self.temporal = TemporalState()
        self.extractor = RobustBiometricExtractor()

        self.use_nn = False
        try:
            self.sess = ort.InferenceSession(
                MODEL_PATH, providers=['CPUExecutionProvider'])
            self.use_nn = True
            print("NN загружена")
        except Exception:
            print("NN не найдена, био-режим")

        self.smoother = AsymmetricSmoother(SMOOTH_RISE, SMOOTH_FALL)
        self.score_hist = deque(maxlen=200)

    def start_calibration(self):
        self.calib = CalibrationData()
        self.calib.start_time = time.time()
        self.state = SystemState.CALIBRATING
        self.temporal = TemporalState()
        self.smoother = AsymmetricSmoother(SMOOTH_RISE, SMOOTH_FALL)
        print("\nНачало калибровки...")

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
            'has_face': has_face, 'bbox': bbox, 'state': self.state,
            'score': 0.0, 'metrics': {}, 'debug': {}
        }

        no_face_dur = self.temporal.update_face_status(has_face, ts)
        if no_face_dur > NO_FACE_TIME_THR and self.state != SystemState.CALIBRATING:
            self.state = SystemState.DISTRACTED
            return res

        if not has_face:
            return res

        ear, mar = float(feats[2]), float(feats[3])
        yaw, pitch = float(feats[4]), float(feats[5])
        face_w = bbox[2] - bbox[0]

        if self.state == SystemState.CALIBRATING:
            nn_p = 0.0
            if self.use_nn:
                try:
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if crop.size > 0:
                        inp = self.preprocess_nn(crop)
                        bio = np.expand_dims(feats, axis=0).astype(np.float32)
                        out = self.sess.run(None, {self.sess.get_inputs()[0].name: inp,
                                                   self.sess.get_inputs()[1].name: bio})
                        nn_p = float(np.exp(
                            out[0][0][1] - np.max(out[0][0])) / np.sum(np.exp(out[0][0] - np.max(out[0][0]))))
                except Exception:
                    pass

            self.calib.add(ear, mar, yaw, pitch, face_w, nn_p)
            prog = self.calib.get_progress()
            res['calib_prog'] = prog

            if prog >= 1.0 and self.calib.compute():
                self.state = SystemState.ACTIVE
                print(f"\nКалибровка завершена!")
                print(f"   EAR порог: {self.calib.ear_thr:.3f}")
                print(f"   MAR порог: {self.calib.mar_thr:.3f}")
                print(f"   Pitch база: {self.calib.pitch_base:.1f}")

            res['metrics'] = {'ear': ear, 'mar': mar, 'pitch': pitch}
            return res

        curr_ear_thr = self.calib.get_adapted_ear_thr(face_w)

        is_closed = ear < curr_ear_thr
        is_yawning = mar > self.calib.mar_thr

        pitch_delta = pitch - self.calib.pitch_base
        is_nodding = (pitch_delta > HEAD_PITCH_THRESH_REL)

        c_dur, y_dur, n_dur = self.temporal.update_events(
            is_closed, is_yawning, is_nodding, ts)

        nn_score = 0.0
        if self.use_nn:
            try:
                crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if crop.size > 0:
                    inp = self.preprocess_nn(crop)
                    bio = np.expand_dims(feats, axis=0).astype(np.float32)
                    out = self.sess.run(None, {self.sess.get_inputs()[0].name: inp,
                                               self.sess.get_inputs()[1].name: bio})
                    probs = np.exp(out[0][0] - np.max(out[0][0])) / \
                        np.sum(np.exp(out[0][0] - np.max(out[0][0])))
                    raw = float(probs[1])
                    nn_score = max(0.0, (raw - self.calib.nn_offset) /
                                   (1.0 - self.calib.nn_offset + 1e-6))
            except Exception:
                pass

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
        res['score'] = smooth_score

        if smooth_score > 0.85:
            self.state = SystemState.ALERT
        elif smooth_score > 0.50:
            self.state = SystemState.DROWSY
        elif smooth_score > 0.30:
            self.state = SystemState.MILD_FATIGUE
        else:
            self.state = SystemState.ACTIVE

        res['metrics'] = {
            'ear': ear, 'ear_thr': curr_ear_thr,
            'mar': mar, 'mar_thr': self.calib.mar_thr,
            'pitch': pitch, 'pitch_base': self.calib.pitch_base,
            'perclos': perclos,
            'c_dur': c_dur, 'y_dur': y_dur, 'n_dur': n_dur,
            'is_closed': is_closed, 'is_yawning': is_yawning,
        }
        res['history'] = list(self.score_hist)

        return res


# ==========================================
# UI / VISUALIZER
# ==========================================

class ModernUI:
    def __init__(self):
        self.alert_alpha = 0.0
        self.alert_dir = 1
        self.v_ear = 0.3
        self.v_mar = 0.0
        self.v_pitch = 0.0

        self.font_16 = load_font(16)
        self.font_18 = load_font(18)
        self.font_20 = load_font(20)
        self.font_22 = load_font(22)
        self.font_28 = load_font(28)
        self.font_48 = load_font(48)

    @staticmethod
    def _render_texts(img, texts):
        if not texts:
            return
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for text, org, font, color in texts:
            draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_bar(self, img, x, y, w, h, val, max_val, color, title, font, thresh=None):
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        ratio = min(1.0, max(0.0, val / max_val))
        cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), color, -1)
        if thresh is not None:
            tx = int((thresh / max_val) * w)
            if 0 < tx < w:
                cv2.line(img, (x + tx, y - 2), (x + tx, y + h + 2), (255, 255, 255), 2)
        return [(title, (x, y - 22), font, (200, 200, 200))]

    def draw(self, frame, res):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        state = res['state']
        texts = []

        if state in [SystemState.ALERT, SystemState.DISTRACTED]:
            self.alert_alpha += 0.1 * self.alert_dir
            if self.alert_alpha > 0.6:
                self.alert_dir = -1
            if self.alert_alpha < 0.2:
                self.alert_dir = 1

            col = (0, 0, 255)
            txt = "ВНИМАНИЕ! УСТАЛОСТЬ!"
            if state == SystemState.DISTRACTED:
                col = (0, 165, 255)
                txt = "ОТВЛЕЧЕНИЕ / НЕТ ЛИЦА"

            cv2.rectangle(overlay, (0, 0), (w, h), col, -1)
            frame = cv2.addWeighted(overlay, self.alert_alpha, frame, 1.0 - self.alert_alpha, 0)
            bbox = self.font_48.getbbox(txt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            texts.append((txt, (w // 2 - tw // 2, h // 2 - th // 2), self.font_48, (255, 255, 255)))
            self._render_texts(frame, texts)
            return frame

        if state == SystemState.CALIBRATING:
            prog = res.get('calib_prog', 0)
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cx, cy = w // 2, h // 2
            bw = 300
            cv2.rectangle(frame, (cx - bw // 2, cy),
                          (int(cx - bw // 2 + bw * prog), cy + 20), (0, 255, 0), -1)
            cv2.rectangle(frame, (cx - bw // 2, cy),
                          (cx + bw // 2, cy + 20), (100, 100, 100), 2)
            texts.append(("КАЛИБРОВКА... СМОТРИТЕ ПРЯМО", (cx - 180, cy - 40),
                          self.font_28, (255, 255, 255)))
            met = res.get('metrics', {})
            texts.append((f"EAR: {met.get('ear', 0):.3f}  MAR: {met.get('mar', 0):.3f}",
                          (cx - 100, cy + 60), self.font_18, (150, 150, 150)))
            self._render_texts(frame, texts)
            return frame

        score = res.get('score', 0.0)
        met = res.get('metrics', {})
        bbox = res.get('bbox')

        if bbox:
            col = (0, 255, 0)
            if score > 0.3:
                col = (0, 255, 255)
            if score > 0.5:
                col = (0, 165, 255)
            if score > 0.8:
                col = (0, 0, 255)
            cv2.rectangle(overlay, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), col, 2)
            texts.append((f"{state.value} {int(score * 100)}%",
                          (bbox[0], bbox[1] - 25), self.font_22, col))

        ph, pw = 230, 260
        px, py = 20, h - ph - 20
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (80, 80, 80), 1)

        self.v_ear = 0.7 * self.v_ear + 0.3 * met.get('ear', 0)
        self.v_mar = 0.7 * self.v_mar + 0.3 * met.get('mar', 0)
        p_raw = met.get('pitch', 0) - met.get('pitch_base', 0)
        self.v_pitch = 0.8 * self.v_pitch + 0.2 * abs(p_raw)

        ear_thr = met.get('ear_thr', 0.21)
        ear_c = (0, 255, 0) if self.v_ear > ear_thr else (0, 0, 255)
        texts.extend(self.draw_bar(overlay, px + 10, py + 25, 180, 8, self.v_ear,
                                   0.45, ear_c, f"Глаза (пор:{ear_thr:.2f})", self.font_16, ear_thr))

        mar_thr = met.get('mar_thr', 0.45)
        mar_c = (0, 255, 0) if self.v_mar < mar_thr else (0, 0, 255)
        texts.extend(self.draw_bar(overlay, px + 10, py + 60, 180, 8, self.v_mar,
                                   0.80, mar_c, f"Рот (пор:{mar_thr:.2f})", self.font_16, mar_thr))

        pit_c = (0, 255, 0) if self.v_pitch < HEAD_PITCH_THRESH_REL else (0, 0, 255)
        texts.extend(self.draw_bar(overlay, px + 10, py + 95, 180, 8, self.v_pitch,
                                   25.0, pit_c, "Наклон головы", self.font_16, HEAD_PITCH_THRESH_REL))

        lines = [
            f"Глаза закрыты: {met.get('c_dur', 0):.1f}с",
            f"Кивок:         {met.get('n_dur', 0):.1f}с",
            f"Зевок:         {met.get('y_dur', 0):.1f}с",
            f"PERCLOS:       {met.get('perclos', 0) * 100:.0f}%"
        ]
        for i, l in enumerate(lines):
            texts.append((l, (px + 10, py + 120 + i * 22), self.font_16, (180, 180, 180)))

        hist = res.get('history', [])
        if len(hist) > 1:
            gx, gy = w - 220, h - 120
            gw, gh = 200, 100
            cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (10, 10, 10), -1)
            pts = []
            for i, v in enumerate(hist):
                x = int(gx + (i / 200) * gw)
                y = int(gy + gh - (v * gh))
                pts.append([x, y])
            cv2.polylines(
                overlay, [np.array(pts, np.int32)], False, (0, 140, 255), 2)
            texts.append(("УРОВЕНЬ УСТАЛОСТИ", (gx, gy - 25), self.font_18, (200, 200, 200)))

        self._render_texts(overlay, texts)
        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


# ==========================================
# MAIN LOOP
# ==========================================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    system = FatigueSystem()
    ui = ModernUI()
    sound_mgr = AlertSoundManager(interval=2.0)

    print("=" * 40)
    print("  СИСТЕМА ДЕТЕКТИРОВАНИЯ УСТАЛОСТИ v2.1")
    print("=" * 40)
    print("[q] Выход  [r] Перекалибровка")
    print()

    system.start_calibration()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            res = system.process(frame)
            view = ui.draw(frame, res)

            sound_mgr.update(res['state'] == SystemState.ALERT)

            cv2.imshow("Детектор усталости", view)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('r'):
                system.start_calibration()
    finally:
        sound_mgr.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
