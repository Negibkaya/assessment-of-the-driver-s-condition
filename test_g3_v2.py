"""
BASELINE APPLICATION: MobileNetV3 без биометрии
Для сравнения с методом адаптивного слияния
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# ==========================================
# ⚙️ НАСТРОЙКИ
# ==========================================
MODEL_PATH = "baseline_cnn.onnx"  # ← Baseline модель (только 1 вход - image)

# Параметры сглаживания
SMOOTH_RISE = 0.35
SMOOTH_FALL = 0.08

# Пороги состояний
THRESHOLD_MILD = 0.30
THRESHOLD_DROWSY = 0.50
THRESHOLD_ALERT = 0.85


# ==========================================
# 🛠 УТИЛИТЫ
# ==========================================

class AsymmetricSmoother:
    """Асимметричное сглаживание: быстрый рост, медленное падение"""

    def __init__(self, rise=0.35, fall=0.08, start=0.0):
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
    ACTIVE = "ACTIVE"
    MILD_FATIGUE = "TIRED"
    DROWSY = "DROWSY"
    ALERT = "!!! WAKE UP !!!"
    NO_MODEL = "MODEL ERROR"


@dataclass
class DetectionResult:
    state: SystemState = SystemState.ACTIVE
    score: float = 0.0
    raw_score: float = 0.0
    confidence: float = 0.0
    fps: float = 0.0
    history: list = field(default_factory=list)


# ==========================================
# 🧠 BASELINE СИСТЕМА (только CNN)
# ==========================================

class BaselineFatigueSystem:
    """
    Простая система на основе только CNN.
    Без MediaPipe, без биометрии, без калибровки.
    """

    def __init__(self, model_path: str):
        self.model_loaded = False
        self.session = None
        self.input_name = None

        # Загрузка модели
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.model_loaded = True
            print(f"✅ Model loaded: {model_path}")
            print(f"   Input: {self.input_name}")
            print(f"   Provider: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

        # Сглаживание
        self.smoother = AsymmetricSmoother(SMOOTH_RISE, SMOOTH_FALL)

        # История для графика
        self.score_history = deque(maxlen=200)

        # FPS
        self.prev_time = time.time()
        self.fps = 0.0

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Предобработка кадра для модели.
        ImageNet нормализация.
        """
        # Resize до 224x224
        img = cv2.resize(frame, (224, 224))

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Нормализация ImageNet
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # HWC → CHW, добавляем batch dimension
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)

        return img

    def process(self, frame: np.ndarray) -> DetectionResult:
        """Обработка одного кадра"""
        result = DetectionResult()

        # FPS
        curr_time = time.time()
        self.fps = 0.9 * self.fps + 0.1 * \
            (1.0 / (curr_time - self.prev_time + 1e-6))
        self.prev_time = curr_time
        result.fps = self.fps

        if not self.model_loaded:
            result.state = SystemState.NO_MODEL
            return result

        try:
            # Предобработка
            input_tensor = self.preprocess(frame)

            # Инференс
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0][0]  # [active_logit, fatigue_logit]

            # Softmax для получения вероятностей
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # Вероятность усталости (класс 1)
            raw_fatigue_prob = float(probs[1])
            result.raw_score = raw_fatigue_prob
            result.confidence = float(np.max(probs))

            # Сглаживание
            smooth_score = self.smoother.update(raw_fatigue_prob)
            result.score = smooth_score

            # История для графика
            self.score_history.append(smooth_score)
            result.history = list(self.score_history)

            # Определение состояния
            if smooth_score > THRESHOLD_ALERT:
                result.state = SystemState.ALERT
            elif smooth_score > THRESHOLD_DROWSY:
                result.state = SystemState.DROWSY
            elif smooth_score > THRESHOLD_MILD:
                result.state = SystemState.MILD_FATIGUE
            else:
                result.state = SystemState.ACTIVE

        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            result.state = SystemState.NO_MODEL

        return result


# ==========================================
# 🎨 UI
# ==========================================

class SimpleUI:
    """Минималистичный UI для baseline"""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.alert_phase = 0.0

    def get_state_color(self, state: SystemState) -> tuple:
        """Цвет в зависимости от состояния"""
        colors = {
            SystemState.ACTIVE: (0, 255, 0),       # Зелёный
            SystemState.MILD_FATIGUE: (0, 255, 255),  # Жёлтый
            SystemState.DROWSY: (0, 165, 255),     # Оранжевый
            SystemState.ALERT: (0, 0, 255),        # Красный
            SystemState.NO_MODEL: (128, 128, 128)  # Серый
        }
        return colors.get(state, (255, 255, 255))

    def draw_score_bar(self, frame, x, y, w, h, score):
        """Прогресс-бар усталости"""
        # Фон
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

        # Заполнение с градиентом цвета
        fill_w = int(w * score)
        if fill_w > 0:
            # Цвет от зелёного к красному
            if score < 0.3:
                color = (0, 255, 0)
            elif score < 0.5:
                color = (0, 255, 255)
            elif score < 0.85:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)

        # Пороговые линии
        for thresh, label in [(THRESHOLD_MILD, ""), (THRESHOLD_DROWSY, ""), (THRESHOLD_ALERT, "")]:
            tx = x + int(w * thresh)
            cv2.line(frame, (tx, y), (tx, y + h), (255, 255, 255), 1)

        # Процент
        cv2.putText(frame, f"{int(score * 100)}%",
                    (x + w + 10, y + h - 2), self.font, 0.5, (255, 255, 255), 1)

    def draw_history_graph(self, frame, x, y, w, h, history):
        """График истории"""
        if len(history) < 2:
            return

        # Фон
        cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)

        # Пороговые линии
        for thresh in [THRESHOLD_MILD, THRESHOLD_DROWSY, THRESHOLD_ALERT]:
            ty = int(y + h * (1 - thresh))
            cv2.line(frame, (x, ty), (x + w, ty), (60, 60, 60), 1)

        # График
        points = []
        for i, val in enumerate(history):
            px = int(x + (i / len(history)) * w)
            py = int(y + h * (1 - val))
            points.append([px, py])

        if len(points) > 1:
            cv2.polylines(
                frame, [np.array(points, np.int32)], False, (0, 140, 255), 2)

        cv2.putText(frame, "FATIGUE HISTORY", (x, y - 5),
                    self.font, 0.4, (150, 150, 150), 1)

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Отрисовка UI"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        state = result.state
        score = result.score
        color = self.get_state_color(state)

        # ===== ALERT MODE =====
        if state == SystemState.ALERT:
            self.alert_phase += 0.15
            alpha = 0.3 + 0.2 * abs(np.sin(self.alert_phase))

            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Текст предупреждения
            text = "!!! FATIGUE ALERT !!!"
            text_size = cv2.getTextSize(text, self.font, 1.5, 3)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2
            cv2.putText(frame, text, (tx, ty), self.font,
                        1.5, (255, 255, 255), 3)

        # ===== СТАТУС ПАНЕЛЬ (верх) =====
        panel_h = 60
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # Название модели
        cv2.putText(frame, "BASELINE (CNN ONLY)", (10, 25),
                    self.font, 0.6, (100, 100, 100), 1)

        # Состояние
        cv2.putText(frame, state.value, (10, 50), self.font, 0.8, color, 2)

        # FPS
        cv2.putText(frame, f"FPS: {result.fps:.0f}", (w - 100, 25),
                    self.font, 0.5, (150, 150, 150), 1)

        # Raw vs Smooth
        cv2.putText(frame, f"Raw: {result.raw_score:.2f}  Smooth: {score:.2f}",
                    (w - 200, 50), self.font, 0.45, (150, 150, 150), 1)

        # ===== ПРОГРЕСС-БАР (низ) =====
        bar_y = h - 50
        bar_h = 25
        self.draw_score_bar(frame, 20, bar_y, w - 100, bar_h, score)

        # ===== ГРАФИК ИСТОРИИ =====
        graph_w, graph_h = 250, 100
        graph_x, graph_y = w - graph_w - 20, h - graph_h - 70
        self.draw_history_graph(frame, graph_x, graph_y,
                                graph_w, graph_h, result.history)

        # ===== РАМКА ВОКРУГ ЭКРАНА =====
        thickness = 4
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

        return frame


# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    print("=" * 50)
    print("  BASELINE FATIGUE DETECTION (CNN ONLY)")
    print("  No MediaPipe, No Biometrics")
    print("=" * 50)
    print("[q] Quit")
    print()

    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    # Настройка камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Инициализация системы
    system = BaselineFatigueSystem(MODEL_PATH)
    ui = SimpleUI()

    print("\n🚀 Starting detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            break

        # Зеркалирование
        frame = cv2.flip(frame, 1)

        # Обработка
        result = system.process(frame)

        # Отрисовка UI
        view = ui.draw(frame, result)

        # Показ
        cv2.imshow("Baseline Fatigue Detection", view)

        # Управление
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
