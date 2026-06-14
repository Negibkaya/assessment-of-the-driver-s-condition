import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

# ==========================================
# ⚙️ НАСТРОЙКИ (CONFIG)
# ==========================================
MODEL_PATH = "mobilenet_bio_fatigue.onnx"

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


# ==========================================
# 🛠 УТИЛИТЫ И СТРУКТУРЫ
# ==========================================

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
    CALIBRATING = "CALIBRATING"
    ACTIVE = "ACTIVE"
    MILD_FATIGUE = "TIRED"
    DROWSY = "DROWSY"
    ALERT = "!!! WAKE UP !!!"
    DISTRACTED = "NO FACE / DISTRACTED"


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
        if ear > 0.08:  # фильтр мусора
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

        # =====================
        # 1. EAR THRESHOLD (ИСПРАВЛЕНО)
        # =====================
        ear_arr = np.array(self.ear_samples)
        ear_med = np.median(ear_arr)
        ear_std = 1.4826 * np.median(np.abs(ear_arr - ear_med))

        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
        # - Используем 0.65 вместо 0.85 (больше запас до прищура)
        # - max() вместо min() для выбора более НИЗКОГО порога
        candidate1 = ear_med * 0.65  # 65% от нормы
        candidate2 = ear_med - 2.5 * ear_std

        # Берём МЕНЬШИЙ кандидат (более низкий порог = меньше ложных срабатываний)
        self.ear_thr = max(0.12, min(candidate1, candidate2))

        print(f"   EAR: median={ear_med:.3f}, std={ear_std:.3f}")
        print(
            f"        candidate1 (65%)={candidate1:.3f}, candidate2 (med-2.5std)={candidate2:.3f}")
        print(f"        → threshold={self.ear_thr:.3f}")

        # =====================
        # 2. MAR THRESHOLD (ИСПРАВЛЕНО)
        # =====================
        if self.mar_samples:
            mar_arr = np.array(self.mar_samples)
            mar_med = np.median(mar_arr)
            mar_std = 1.4826 * np.median(np.abs(mar_arr - mar_med))

            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
            # - Используем 1.8 вместо 3.0 (легче детектировать зевок)
            # - Понижен минимум с 0.35 до 0.25
            self.mar_thr = max(0.25, mar_med + 1.8 * mar_std)

            print(f"   MAR: median={mar_med:.3f}, std={mar_std:.3f}")
            print(f"        → threshold={self.mar_thr:.3f}")
        else:
            self.mar_thr = MAR_THRESH_DEFAULT

        # 3. Head Pose Base
        self.yaw_base = float(np.mean(self.yaw_samples))
        self.pitch_base = float(np.mean(self.pitch_samples))

        # 4. NN Offset
        if self.nn_samples:
            self.nn_offset = float(np.percentile(self.nn_samples, 90))

        # 5. Face Width
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
# 📐 УЛУЧШЕННЫЙ BIO EXTRACTOR
# ==========================================

class RobustBiometricExtractor:
    """
    ИСПРАВЛЕНИЯ:
    1. EAR: использует 6 пар точек (3 на глаз) вместо 2
    2. MAR: нормализует на ШИРИНУ ЛИЦА, а не ширину рта
    3. Добавлены правильные индексы MediaPipe
    """

    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # ========== EAR LANDMARKS ==========
        # Пары (верхнее веко, нижнее веко) для вертикальных расстояний

        # Right eye (левый на изображении) - глаз ЧЕЛОВЕКА справа
        self.R_EYE_V_PAIRS = [
            (159, 145),  # центральная пара
            (160, 144),  # правее
            (158, 153),  # левее
        ]
        self.R_EYE_H = (33, 133)  # внешний угол, внутренний угол

        # Left eye (правый на изображении) - глаз ЧЕЛОВЕКА слева
        self.L_EYE_V_PAIRS = [
            (386, 374),  # центральная пара
            (385, 380),  # левее
            (387, 373),  # правее
        ]
        self.L_EYE_H = (263, 362)  # внешний угол, внутренний угол

        # ========== MAR LANDMARKS ==========
        # Вертикальные пары (верхняя губа внутри, нижняя губа внутри)
        self.MOUTH_V_PAIRS = [
            (13, 14),    # центр
            (81, 178),   # левее от центра
            (82, 87),    # ещё левее
            (311, 402),  # правее от центра
            (312, 317),  # ещё правее
        ]

        # Ширина лица (для нормализации MAR)
        self.FACE_WIDTH_PTS = (234, 454)  # левый край лица, правый край

    def _calc_ear(self, pts, v_pairs, h_pair):
        """
        EAR = среднее(вертикальные расстояния) / горизонтальное расстояние
        Использует несколько пар точек для стабильности
        """
        vert_distances = []
        for upper_idx, lower_idx in v_pairs:
            dist = np.linalg.norm(pts[upper_idx] - pts[lower_idx])
            vert_distances.append(dist)

        avg_vert = np.mean(vert_distances)
        horiz = np.linalg.norm(pts[h_pair[0]] - pts[h_pair[1]])

        # Классическая формула EAR
        return avg_vert / (horiz + 1e-6)

    def _calc_mar(self, pts, face_width):
        """
        MAR = среднее(вертикальные расстояния рта) / (ширина_лица * коэфф)

        КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: нормализуем на ширину ЛИЦА, а не рта!
        При зевке рот растягивается вширь, что занижало MAR.
        """
        vert_distances = []
        for upper_idx, lower_idx in self.MOUTH_V_PAIRS:
            dist = np.linalg.norm(pts[upper_idx] - pts[lower_idx])
            vert_distances.append(dist)

        avg_vert = np.mean(vert_distances)

        # Нормализация на ширину лица (коэфф 0.22 подобран эмпирически)
        # При закрытом рте: MAR ≈ 0.05-0.15
        # При зевке: MAR ≈ 0.4-0.8
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

        # Ширина лица для нормализаций
        face_width = np.linalg.norm(
            pts[self.FACE_WIDTH_PTS[0]] - pts[self.FACE_WIDTH_PTS[1]]
        ) + 1e-6

        # ========== 1. EAR (улучшенный) ==========
        l_ear = self._calc_ear(pts, self.L_EYE_V_PAIRS, self.L_EYE_H)
        r_ear = self._calc_ear(pts, self.R_EYE_V_PAIRS, self.R_EYE_H)
        ear = (l_ear + r_ear) / 2.0

        # ========== 2. MAR (улучшенный) ==========
        mar = self._calc_mar(pts, face_width)

        # ========== 3. Head Pose ==========
        nose = pts[1]
        eye_mid = (pts[33] + pts[263]) / 2.0

        dx = nose[0] - eye_mid[0]
        dy = nose[1] - eye_mid[1]

        yaw = (dx / face_width) * 100
        pitch = (dy / face_width) * 100

        feats[:] = [l_ear, r_ear, ear, mar, yaw, pitch, 0, 1]

        # Bbox
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        pad_x, pad_y = (xmax-xmin)*0.2, (ymax-ymin)*0.2
        bbox = (int(max(0, xmin-pad_x)), int(max(0, ymin-pad_y)),
                int(min(w, xmax+pad_x)), int(min(h, ymax+pad_y)))

        return feats, True, bbox


# ==========================================
# 🧠 CORE LOGIC
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
            print("✅ NN Loaded")
        except:
            print("⚠️ NN Not found, running bio-only")

        self.smoother = AsymmetricSmoother(SMOOTH_RISE, SMOOTH_FALL)
        self.score_hist = deque(maxlen=200)

    def start_calibration(self):
        self.calib = CalibrationData()
        self.calib.start_time = time.time()
        self.state = SystemState.CALIBRATING
        self.temporal = TemporalState()
        self.smoother = AsymmetricSmoother(SMOOTH_RISE, SMOOTH_FALL)
        print("\n🔄 Starting calibration...")

    def preprocess_nn(self, crop):
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32)/255.0 -
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

        # ========== CALIBRATION ==========
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
                except:
                    pass

            self.calib.add(ear, mar, yaw, pitch, face_w, nn_p)
            prog = self.calib.get_progress()
            res['calib_prog'] = prog

            if prog >= 1.0 and self.calib.compute():
                self.state = SystemState.ACTIVE
                print(f"\n✅ Calibration complete!")
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
        is_nodding = (pitch_delta > HEAD_PITCH_THRESH_REL)

        c_dur, y_dur, n_dur = self.temporal.update_events(
            is_closed, is_yawning, is_nodding, ts)

        # ========== SCORING ==========
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
            except:
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
            'is_closed': is_closed, 'is_yawning': is_yawning,  # для отладки
        }
        res['history'] = list(self.score_hist)

        return res


# ==========================================
# 🎨 UI / VISUALIZER
# ==========================================

class ModernUI:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.alert_alpha = 0.0
        self.alert_dir = 1
        self.v_ear = 0.3
        self.v_mar = 0.0
        self.v_pitch = 0.0

    def draw_bar(self, img, x, y, w, h, val, max_val, color, title, thresh=None):
        cv2.rectangle(img, (x, y), (x+w, y+h), (40, 40, 40), -1)
        ratio = min(1.0, max(0.0, val / max_val))
        cv2.rectangle(img, (x, y), (x+int(w*ratio), y+h), color, -1)
        if thresh is not None:
            tx = int((thresh/max_val)*w)
            if 0 < tx < w:
                cv2.line(img, (x+tx, y-2), (x+tx, y+h+2), (255, 255, 255), 2)
        cv2.putText(img, title, (x, y-5), self.font, 0.4, (200, 200, 200), 1)

    def draw(self, frame, res):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        state = res['state']

        if state in [SystemState.ALERT, SystemState.DISTRACTED]:
            self.alert_alpha += 0.1 * self.alert_dir
            if self.alert_alpha > 0.6:
                self.alert_dir = -1
            if self.alert_alpha < 0.2:
                self.alert_dir = 1

            col = (0, 0, 255)
            txt = "FATIGUE ALERT!"
            if state == SystemState.DISTRACTED:
                col = (0, 165, 255)
                txt = "DISTRACTED / NO FACE"

            cv2.rectangle(overlay, (0, 0), (w, h), col, -1)
            cv2.putText(overlay, txt, (w//2-200, h//2),
                        self.font, 1.5, (255, 255, 255), 3)
            return cv2.addWeighted(overlay, self.alert_alpha, frame, 1.0-self.alert_alpha, 0)

        if state == SystemState.CALIBRATING:
            prog = res.get('calib_prog', 0)
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cx, cy = w//2, h//2
            bw = 300
            cv2.rectangle(frame, (cx-bw//2, cy),
                          (int(cx-bw//2 + bw*prog), cy+20), (0, 255, 0), -1)
            cv2.rectangle(frame, (cx-bw//2, cy),
                          (cx+bw//2, cy+20), (100, 100, 100), 2)
            cv2.putText(frame, "CALIBRATING... LOOK STRAIGHT",
                        (cx-180, cy-30), self.font, 0.8, (255, 255, 255), 1)

            # Показываем текущие значения при калибровке
            met = res.get('metrics', {})
            cv2.putText(frame, f"EAR: {met.get('ear', 0):.3f}  MAR: {met.get('mar', 0):.3f}",
                        (cx-100, cy+60), self.font, 0.5, (150, 150, 150), 1)
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
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), col, 2)
            cv2.putText(frame, f"{state.value} {int(score*100)}%",
                        (bbox[0], bbox[1]-10), self.font, 0.6, col, 2)

        # Dashboard
        ph, pw = 200, 260
        px, py = 20, h - ph - 20
        cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (10, 10, 10), -1)
        cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (80, 80, 80), 1)

        self.v_ear = 0.7*self.v_ear + 0.3*met.get('ear', 0)
        self.v_mar = 0.7*self.v_mar + 0.3*met.get('mar', 0)
        p_raw = met.get('pitch', 0) - met.get('pitch_base', 0)
        self.v_pitch = 0.8*self.v_pitch + 0.2*abs(p_raw)

        # EAR bar (инвертированная логика цвета - красный когда ниже порога)
        ear_thr = met.get('ear_thr', 0.21)
        ear_c = (0, 255, 0) if self.v_ear > ear_thr else (0, 0, 255)
        self.draw_bar(overlay, px+10, py+30, 180, 10, self.v_ear,
                      0.45, ear_c, f"Eyes (thr:{ear_thr:.2f})", ear_thr)

        # MAR bar (красный когда выше порога)
        mar_thr = met.get('mar_thr', 0.45)
        mar_c = (0, 255, 0) if self.v_mar < mar_thr else (0, 0, 255)
        self.draw_bar(overlay, px+10, py+70, 180, 10, self.v_mar,
                      0.80, mar_c, f"Mouth (thr:{mar_thr:.2f})", mar_thr)

        # Pitch bar
        pit_c = (0, 255, 0) if self.v_pitch < HEAD_PITCH_THRESH_REL else (
            0, 0, 255)
        self.draw_bar(overlay, px+10, py+110, 180, 10, self.v_pitch,
                      25.0, pit_c, "Head Nod", HEAD_PITCH_THRESH_REL)

        # Debug info
        lines = [
            f"Eyes closed: {met.get('c_dur', 0):.1f}s",
            f"Nodding:     {met.get('n_dur', 0):.1f}s",
            f"Yawning:     {met.get('y_dur', 0):.1f}s",
            f"PERCLOS:     {met.get('perclos', 0)*100:.0f}%"
        ]
        for i, l in enumerate(lines):
            cv2.putText(overlay, l, (px+10, py+145+i*18),
                        self.font, 0.45, (180, 180, 180), 1)

        # Graph
        hist = res.get('history', [])
        if len(hist) > 1:
            gx, gy = w - 220, h - 120
            gw, gh = 200, 100
            cv2.rectangle(overlay, (gx, gy), (gx+gw, gy+gh), (10, 10, 10), -1)
            pts = []
            for i, v in enumerate(hist):
                x = int(gx + (i / 200) * gw)
                y = int(gy + gh - (v * gh))
                pts.append([x, y])
            cv2.polylines(
                overlay, [np.array(pts, np.int32)], False, (0, 140, 255), 2)
            cv2.putText(overlay, "FATIGUE LEVEL", (gx, gy-10),
                        self.font, 0.5, (200, 200, 200), 1)

        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


# ==========================================
# 🚀 MAIN LOOP
# ==========================================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    system = FatigueSystem()
    ui = ModernUI()

    print("=" * 40)
    print("    DRIVER FATIGUE SYSTEM v2.0")
    print("=" * 40)
    print("[q] Quit  [r] Recalibrate")
    print()

    system.start_calibration()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        res = system.process(frame)
        view = ui.draw(frame, res)

        cv2.imshow("Fatigue Detection", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            system.start_calibration()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
