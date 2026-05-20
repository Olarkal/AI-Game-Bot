import ctypes
from ctypes import wintypes
import time
import threading
import queue
import os
import json
import random
import math
import sys
import traceback
import re
from collections import Counter
from itertools import groupby
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import torch
import torch.nn as nn
import torchvision.transforms as T 
try:
    import cupy as cp 
except ImportError:
    cp = None
from ultralytics import YOLO
import pytesseract
import requests
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from screeninfo import get_monitors
import win32gui
import win32con
import win32api
import keyboard
import pickle
from pathlib import Path

SendInput = ctypes.windll.user32.SendInput
global_log_queue = queue.Queue(maxsize=100)
global_stop_event = threading.Event()

# Конфигурация

OBS_CAMERA_INDEX = 2 #Номер виртуальной камеры OBS (0-3)
YOLO_MODEL_PATH = "yolov8m.pt" #Модель ИИ (8n, 8s, 8m, 8l, 8x)
ENABLE_GUI = True #Оверлей
DEFAULT_FRAME_RESIZE = (640, 480)#размер кадра виртуальной камеры
DEMO_RECORD_DURATION = 900  # Время записи демки в секундах
DEMO_VIDEO_FPS = 60  # FPS для записи демки
MAX_PROFILES = 20  # максимальное количество профилей
TIMESTEPS_PER_AGENT = 2048 #Кол-во шагов для одного агента
MAX_GENS = 10 #Кол-во поколений
BIND_PAUSE = "f1" #кнопка паузы
BIND_STOP = "f3" #кнопка завершения программы
BIND_RECORD = "f2" #кнопка записи демо
CLEAR_GPU_CACHE = True 
DEAD_ZONE_LEFT   = 0.15   # оверлей бота
DEAD_ZONE_TOP    = 0.082   # иконки игроков
DEAD_ZONE_BOTTOM = 0.82   # зона оружия
ACTION_DURATION_MOVE  = 0.12   # было 0.25
ACTION_DURATION_CLICK = 0.08
ACTION_DURATION_WAIT  = 0.03


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

UI_KEYWORDS = [
    "inventory", "инвентарь", "search", "поиск", "settings", "настройки", "pause", "пауза", "chat", "чат"
]

# WinAPI SendInput wrapper
user32 = ctypes.windll.user32
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
PUL = ctypes.POINTER(ctypes.c_ulong)

#Структуры WinAPI для управления HID
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", PUL)
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", PUL)
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD)
    ]

class _INPUTunion(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT)
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUTunion)
    ]

#Базовая функция отправки массива структур INPUT через SendInput
def send_input(inputs):
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    cbsize = ctypes.sizeof(INPUT)
    res = user32.SendInput(n, ctypes.byref(arr), cbsize)
    return res

#блок низкоуровневой генерации нажатий клавиш клавиатуры для Windows API
def make_key_input(vk, down=True):
    sc = user32.MapVirtualKeyW(vk, 0)
    flags = 0
    if not down:
        flags |= KEYEVENTF_KEYUP
    if sc != 0:
        flags |= KEYEVENTF_SCANCODE
    ki = KEYBDINPUT(vk, sc, flags, 0, ctypes.pointer(ctypes.c_ulong(0)))
    print(f"VK={vk}, Scan={sc}, Down={down}")
    return INPUT(INPUT_KEYBOARD, _INPUTunion(ki=ki))

#блок создает команду для относительного перемещения мыши на заданные координаты
def make_mouse_move(dx, dy):
    mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(ctypes.c_ulong(0)))
    return INPUT(INPUT_MOUSE, _INPUTunion(mi=mi))

#блок отвечает за генерацию кликов мыши
def make_mouse_click(button='left', down=True):
    if button.lower() in ('left', 'lmb'):
        flags = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
    elif button.lower() in ('right', 'rmb'):
        flags = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
    else:
        raise ValueError(f"Unknown mouse button: {button}")
    mi = MOUSEINPUT(0, 0, 0, flags, 0, ctypes.pointer(ctypes.c_ulong(0)))
    return INPUT(INPUT_MOUSE, _INPUTunion(mi=mi))

#блок предназначен для абсолютного позиционирования курсора мыши на экране в пикселях
def set_mouse_position(x, y):
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    w = (screen_w - 1) if screen_w > 1 else 1
    h = (screen_h - 1) if screen_h > 1 else 1
    abs_x = int(x * 65535 / (screen_w - 1))
    abs_y = int(y * 65535 / (screen_h - 1))
    mi = MOUSEINPUT(abs_x, abs_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, ctypes.pointer(ctypes.c_ulong(0)))
    return INPUT(INPUT_MOUSE, _INPUTunion(mi=mi))

#блок создает словарь VK, который связывает текстовые названия клавиш
VK = {
    'backspace': 0x08, 'tab':0x09, 'enter':0x0D, 'shift':0x10, 'ctrl':0x11, 'alt':0x12,
    'pause':0x13, 'capslock':0x14, 'esc':0x1B, 'space':0x20, 'page_up':0x21, 'page_down':0x22,
    'end':0x23, 'home':0x24, 'left':0x25, 'up':0x26, 'right':0x27, 'down':0x28,
    'print':0x2A, 'insert':0x2D, 'delete':0x2E,
    '0':0x30,'1':0x31,'2':0x32,'3':0x33,'4':0x34,'5':0x35,'6':0x36,'7':0x37,'8':0x38,'9':0x39,
    **{chr(ord('a')+i): 0x41 + i for i in range(26)},
    **{f'f{i}': 0x70 + i - 1 for i in range(1,13)},
    'comma':0xBC, 'minus':0xBD, 'period':0xBE, 'slash':0xBF, 'tilde':0xC0,
    'lbracket':0xDB, 'backslash':0xDC, 'rbracket':0xDD, 'apostrophe':0xDE,
}
VK_REVERSE = {v: k for k, v in VK.items()}

def vk_to_name(vk):
    return VK_REVERSE.get(vk, f"VK_{vk}")

#блок переводит текстовое название клавиши или одиночный символ в числовой код виртуальной клавиши
def key_to_vk(key):
    k = key.lower()
    if k in VK:
        return VK[k]
    if len(k)==1:
        c = k
        if 'a' <= c <= 'z':
            return ord(c.upper())
        if '0' <= c <= '9':
            return ord(c)
    return None

#В этом блоке описана сверточная нейросеть FeatureNet (архитектура Nature DQN), которая сжимает картинку игрового экрана в компактный вектор признаков (фичей) размером 512.
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        ).float()
    def forward(self, x):
        return self.cnn(x)

#В этом блоке реализован отдельный поток захвата кадров с виртуальной камеры OBS.
class FrameCaptureThread(threading.Thread):
    def __init__(self, env):
        super().__init__(daemon=True)
        self.env = env
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running and not self.env.stop_event.is_set():
            self.env.cap.grab()
            ret, frame = self.env.cap.retrieve()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.latest_frame = frame
                with self.env.lock:
                    self.env.current_frame = frame
            time.sleep(0.0)

#Объявление класса среды GameEnv на базе gymnasium
class GameEnv(gym.Env):
    active_gui = None
    current_env = None
    
    #здесь заложена вся основа среды! многопоточный захват, асинхронные воркеры на отправку и обработку действий, интеграция с YOLO, нейросеть предсказания следующего кадра, подгрузка паттернов из пользовательской демки.
    def __init__(self, game_name="", profile_id=0, is_final_profile=False, generation=0, saved_state=None):
        super(GameEnv, self).__init__()
        self.threads = []
        self.profile_id = profile_id
        self.generation = generation
        self.agent_id = profile_id
        self.is_final_profile = is_final_profile
        self.is_gui_destroyed = False
        # Накопительные метрики (сохраняем при перезапуске)
        self.total_reward_accumulated = 0.0
        self.total_reward = 0.0
        self.n_steps = 0
        self.total_steps_accumulated = 0
        self.profile_performance = {}
        self.is_shooter = False
        # Синхронизация и потокобезопасность
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.recording_active = False 
        self.frame_queue = queue.Queue(maxsize=30)
        self.action_queue = queue.Queue(maxsize=30)
        self.send_queue = queue.Queue(maxsize=30)
        self.lock = threading.Lock()
        self.held_keys = set()
        self.held_mouse = set()
        self.held_keys_lock = threading.Lock()
        # Состояние детекции окружения
        self.prev_frame = None
        self.prev_ocr_text = ""
        self.prev_objects = set()
        self._last_actions = []
        self._idle_action_count = 0
        self.known_objects = {}
        self.frame_count = 0
        self.frame_processed = 0
        self.last_demo_step = 0
        self.annotated_frame = None
        self.yolo_results = None
        self.last_api_call = 0
        self.api_call_interval = 0.1  # Уменьшено для FPS
        self.ui_mode = False
        self.current_frame = None
        self.current_reward = 0.0
        self.total_timesteps = TIMESTEPS_PER_AGENT 
        # Инициализация окна игры с защитой от спецсимволов в путях Windows
        raw_game_name = game_name or self.detect_game_window_name()
        self.game_name = raw_game_name
        self.safe_game_name = re.sub(r'[\\/*?:"<>|]', "_", raw_game_name) # Для путей файлов
        self.hwnd = win32gui.FindWindow(None, self.game_name)
        self._icm_queue = queue.Queue(maxsize=5)
        self._last_intrinsic = 0.0
        self._icm_thread = threading.Thread(target=self._icm_worker, daemon=True)
        self.threads.append(self._icm_thread)
        self._icm_thread.start()
        if not self.hwnd:
            self.debug_print("Окно игры не найдено, используется текущее активное окно")
            self.hwnd = win32gui.GetForegroundWindow()
        # Инициализация видеозахвата OBS
        self.cap = cv2.VideoCapture(OBS_CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Не удалось открыть виртуальную камеру OBS. Попробуй другой индекс.")
            self.check_available_cameras()
            raise Exception("Не удалось открыть камеру")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or DEFAULT_FRAME_RESIZE[0]
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or DEFAULT_FRAME_RESIZE[1]
        self.debug_print(f"Виртуальная камера: {self.width}x{self.height}")
        # Инициализация вычислительного девайса
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_device = torch.device("cpu")
        if self.device.type == "cuda":
            self.debug_print(f"PPO/ICM: CUDA ({torch.cuda.get_device_name(0)}) | YOLO: CPU")
        else:
            self.debug_print("Все компоненты на CPU")
        # YOLO всегда на CPU — не мешает игре
        self.model_yolo = YOLO(YOLO_MODEL_PATH).to(self.yolo_device)
        self.debug_print(f"YOLO device: cpu")
        self.debug_print(f"YOLO device: {self.model_yolo.device} | imgsz=320 | FP16={'yes' if self.device.type=='cuda' else 'no'}")
        # Пространство наблюдений строго под FeatureNet (C, H, W)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,84,84), dtype=np.uint8)
        # Описание действий бота
        self.actions = [
            ("wait", []), ("w", ["w"]), ("a", ["a"]), ("s", ["s"]), ("d", ["d"]),
            ("space", ["space"]), ("shift", ["shift"]), ("ctrl", ["ctrl"]), ("e", ["e"]),
            ("click_left", ["click_left"]), ("click_right", ["click_right"]),
            ("camera_left", ["camera_left"]), ("camera_right", ["camera_right"]),
            ("camera_up", ["camera_up"]), ("camera_down", ["camera_down"]),
            ("aim_at_person", ["aim_at_person"]), ("hold_left", ["hold_left"]),
            # Комбинации: движение + мышь
            ("w+camera_left",  ["w", "camera_left"]),
            ("w+camera_right", ["w", "camera_right"]),
            ("w+click_left",   ["w", "click_left"]),
            ("w+aim",          ["w", "aim_at_person"]),
            ("shift+w",        ["shift", "w"]),
            ("shift+w+camera_left",  ["shift", "w", "camera_left"]),
            ("shift+w+camera_right", ["shift", "w", "camera_right"]),
            ("ctrl+w",         ["ctrl", "w"]),
            ("aim+click_left", ["aim_at_person", "click_left"]),
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_map = [
            # 0  wait
            {'keys': [],          'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.05},
            # 1  w
            {'keys': ['w'],       'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25},
            # 2  a
            {'keys': ['a'],       'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25},
            # 3  s
            {'keys': ['s'],       'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25},
            # 4  d
            {'keys': ['d'],       'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25},
            # 5  space
            {'keys': ['space'],   'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.15},
            # 6  shift
            {'keys': ['shift'],   'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25},
            # 7  ctrl
            {'keys': ['ctrl'],    'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25},
            # 8  e
            {'keys': ['e'],       'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.15},
            # 9  click_left
            {'keys': [],          'mouse_move': (0, 0),     'mouse_click': 'left',     'duration': 0.25},
            # 10 click_right
            {'keys': [],          'mouse_move': (0, 0),     'mouse_click': 'right',    'duration': 0.25},
            # 11 camera_left  — увеличен сдвиг для реальной смены угла обзора
            {'keys': [],          'mouse_move': (-600, 0),  'mouse_click': None,       'duration': 0.25},
            # 12 camera_right
            {'keys': [],          'mouse_move': (600, 0), 'mouse_click': None,       'duration': 0.25},
            # 13 camera_up
            {'keys': [],          'mouse_move': (0, -200),   'mouse_click': None,       'duration': 0.25},
            # 14 camera_down
            {'keys': [],          'mouse_move': (0, 200),  'mouse_click': None,       'duration': 0.25},
            # 15 aim_at_person
            {'keys': [],          'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25, 'aim_at_person': True},
            # 16 hold_left
            {'keys': [],          'mouse_move': (0, 0),     'mouse_click': 'hold_left','duration': 0.25,  'aim_at_person': True},
            # 17 w + camera_left
            {'keys': ['w'],       'mouse_move': (-600, 0),  'mouse_click': None,       'duration': 0.25},
            # 18 w + camera_right
            {'keys': ['w'],       'mouse_move': (600, 0), 'mouse_click': None,       'duration': 0.25},
            # 19 w + click_left
            {'keys': ['w'],       'mouse_move': (0, 0),     'mouse_click': 'left',     'duration': 0.25},
            # 20 w + aim
            {'keys': ['w'],       'mouse_move': (0, 0),     'mouse_click': None,       'duration': 0.25,  'aim_at_person': True},
            # 21 shift + w
            {'keys': ['shift','w'],'mouse_move': (0, 0),    'mouse_click': None,       'duration': 0.25},
            # 22 shift + w + camera_left
            {'keys': ['shift','w'],'mouse_move': (-600, 0), 'mouse_click': None,       'duration': 0.25},
            # 23 shift + w + camera_right
            {'keys': ['shift','w'],'mouse_move': (600, 0),'mouse_click': None,       'duration': 0.25},
            # 24 ctrl + w
            {'keys': ['ctrl','w'], 'mouse_move': (0, 0),    'mouse_click': None,       'duration': 0.25},
            # 25 aim + click_left
            {'keys': [],          'mouse_move': (0, 0),     'mouse_click': 'left',     'duration': 0.25,  'aim_at_person': True},
        ]
        # Конфиг игры и парсинг жанра
        config = self.load_config()
        self.game_description = config.get("game_description", "")
        if not self.game_description:
            self.debug_print("Получаю описание игры в интернете...")
            try:
                self.fetch_game_description()
                self.debug_print(f"Описание игры: {self.game_description}")
            except Exception as e:
                self.debug_print(f"Ошибка при получении описания: {e}")
        self.description_keywords, self.is_shooter = self.analyze_game_description()
        # Инициализация архитектурных сетей ИИ
        self.feature_net = FeatureNet().to(self.device).float()
        self.forward_model = nn.Sequential(
            nn.Linear(512 + len(self.actions), 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        ).to(self.device).float()
        self.optimizer = torch.optim.Adam(
            list(self.feature_net.parameters()) + list(self.forward_model.parameters()),
            lr=1e-3
        )
        self.model = None
        # ЕСЛИ ЕСТЬ СОХРАНЕННОЕ СОСТОЯНИЕ (ПРИ ПЕРЕЗАПУСКЕ ИЛИ МУТАЦИИ) — ВОССТАНАВЛИВАЕМ
        if saved_state:
            self.load_state_dict_custom(saved_state)
        # Создаем директорию для демок заранее, чтобы избежать краша путей
        os.makedirs(f"profiles/{self.safe_game_name}/shared_demo", exist_ok=True)
        self.demo_cache = None
        self.demo_access_count = 0
        self.demo_video_path = f"profiles/{self.safe_game_name}/shared_demo/demo_video.avi"
        self.demo_actions_path = f"profiles/{self.safe_game_name}/shared_demo/demo_actions.json"
        self.has_demo = os.path.exists(self.demo_video_path) and os.path.exists(self.demo_actions_path)
        self.demo_actions = []
        self.behavior_archetype = None
        if self.has_demo:
            self.load_demo_actions()
            self.demo_access_count = 0
            self.analyze_demo_behavior()
        # Запуск многопоточных воркеров
        self.capture_thread = FrameCaptureThread(self)
        self.capture_thread.start()
        self.threads.append(self.capture_thread)
        self.frame_worker_thread = threading.Thread(target=self.frame_capture_worker, daemon=True)
        self.threads.append(self.frame_worker_thread)
        self.frame_worker_thread.start()
        self.send_worker_thread = threading.Thread(target=self._send_worker, daemon=True)
        self.threads.append(self.send_worker_thread)
        self.send_worker_thread.start()
        self.action_worker_thread = threading.Thread(target=self._action_worker, daemon=True)
        self.threads.append(self.action_worker_thread)
        self.action_worker_thread.start()
        self.yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self.threads.append(self.yolo_thread)
        self.yolo_thread.start()
        self.aim_thread = threading.Thread(target=self._aim_worker, daemon=True)
        self.threads.append(self.aim_thread)
        self.aim_thread.start()
        GameEnv.current_env = self
        console_thread = threading.Thread(target=self.console_listener, args=(self.stop_event, self.pause_event), daemon=True)
        self.threads.append(console_thread)
        console_thread.start()
        self.restrict_cursor_to_window()
        self.pause_event.set()
    
    #Этот блок отвечает за вывод отладочных сообщений в консоль.
    def debug_print(self, message, force=False):
        # Печатаем, если бот активен, или если это важное системное сообщение (force)
        # Также полезно видеть логи инициализации, когда pause_event еще не сброшен
        if force or "Ошибка" in message or "критический" in message.lower() or not self.pause_event.is_set():
            print(f"[{self.game_name}][Gen {self.generation}][Agent {self.agent_id}] {message}")

    #блок кода отвечает за разбор записанной человеком демонстрации и формирование архетипа поведения.
    def analyze_demo_behavior(self):
        if not self.demo_actions:
            self.debug_print("Нет действий в демо для анализа")
            return
        # Анализ частых действий
        all_actions = [action['keys'] + action['mouse_buttons'] for action in self.demo_actions]
        flat_actions = [item for sublist in all_actions for item in sublist]
        action_counts = Counter(flat_actions)
        most_common_actions = action_counts.most_common(5)
        # Анализ последовательностей (например, цепочки действий)
        sequences = []
        for action in self.demo_actions:
            seq = tuple(sorted(set(action['keys'] + action['mouse_buttons'])))
            if seq:
                sequences.append(seq)
        # Считаем реальную частоту всех уникальных комбинаций клавиш
        sequence_counts = Counter(sequences)
        common_sequences = sequence_counts.most_common(5)
        # Анализ движения мыши (средние dx, dy)
        mouse_moves = [action['mouse_pos'] for action in self.demo_actions if action['mouse_pos'] != [0, 0]]
        if mouse_moves:
            avg_dx = sum(abs(pos[0]) for pos in mouse_moves) / len(mouse_moves)
            avg_dy = sum(abs(pos[1]) for pos in mouse_moves) / len(mouse_moves)
        else:
            avg_dx, avg_dy = 0, 0
        # Формируем архетип, конвертируя tuple в list для безопасного json.dumps
        self.behavior_archetype = {
            "most_common_actions": [[act, count] for act, count in most_common_actions],
            "common_sequences": [[list(seq), count] for seq, count in common_sequences],
            "average_mouse_movement": (avg_dx, avg_dy)
        }
        self.debug_print(f"Архетип поведения из демо: {json.dumps(self.behavior_archetype, ensure_ascii=False, indent=2)}")

    #метод отвечает за сохранение весов нейросетей, дампов модели PPO и обновление конфигурационного файла со статистикой обучения.
    def save_training_state(self, save_path=None, is_cleanup=False):
        if is_cleanup and self.n_steps == 0 and self.total_reward == 0:
            print(f"[DEBUG] Пропуск дублирующего сохранения при очистке для Agent {self.agent_id}")
            return
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Используем безопасное имя или дефолтный путь профиля
        save_path = save_path or f"profiles/{self.safe_game_name}"
        os.makedirs(save_path, exist_ok=True)
        # 1. Сохранение PPO модели (Stable-Baselines3)
        try:
            if hasattr(self, 'model') and self.model is not None:
                ppo_save_path = os.path.join(save_path, f"model.zip")
                self.model.save(ppo_save_path)
                self.debug_print(f"PPO модель сохранена в {ppo_save_path}, Шагов: {self.frame_processed}")
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении PPO модели: {e}")
        # 2. Сохранение Feature Net (Фиксированное имя для стабильной загрузки + бэкап)
        try:
            feature_net_save_path = os.path.join(save_path, "feature_net.pth")
            torch.save(self.feature_net.state_dict(), feature_net_save_path)
            # Резервная копия (опционально, для истории поколений)
            backup_dir = os.path.join(save_path, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            torch.save(self.feature_net.state_dict(), os.path.join(backup_dir, f"feature_net_{timestamp}.pth"))
            self.debug_print(f"Feature net сохранена в {feature_net_save_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении feature_net: {e}")
        # 3. Сохранение Forward Model (Curiosity)
        try:
            forward_model_save_path = os.path.join(save_path, "forward_model.pth")
            torch.save(self.forward_model.state_dict(), forward_model_save_path)
            # Резервная копия
            torch.save(self.forward_model.state_dict(), os.path.join(backup_dir, f"forward_model_{timestamp}.pth"))
            self.debug_print(f"Forward model сохранена в {forward_model_save_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении forward_model: {e}")
        # 4. Потокобезопасное сохранение статистики в config.json
        try:
            with self.lock: # Защищаем операцию от вмешательства GUI-потока
                config = self.load_config()
                if config is None:
                    config = {}
                incoming_reward = self.total_reward
                global_reward = config.get("total_reward", 0.0)
                new_total_reward = global_reward + incoming_reward
                self.total_reward = 0.0
                self.total_reward_accumulated = new_total_reward
                config["total_reward"] = new_total_reward 
                config["frame_processed"] = int(self.frame_processed)
                config["last_saved_at"] = timestamp
                if not config.get("rewards"):
                    config["rewards"] = {}
                config["rewards"][timestamp] = round(incoming_reward, 4)
                config["known_objects"] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in self.known_objects.items()
                } if self.known_objects else {}
                self.save_config(config)
            self.debug_print(f"Config обновлён: total_reward={new_total_reward}, frame_processed={self.frame_processed}")
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении config.json: {e}", force=True)
    
    #Логика восстановления весов сетей
    def load_training_state(self, ppo_path=None, feature_net_path=None, forward_model_path=None):
        # 1. Загрузка PPO модели
        try:
            if ppo_path and os.path.exists(ppo_path):
                # Из импортов stable-baselines3, загружаем напрямую в переменную среды
                from stable_baselines3 import PPO
                # Загружаем модель, принудительно привязывая её к текущему девайсу среды
                self.model = PPO.load(ppo_path, env=self, device=self.device)
                self.debug_print(f"PPO модель успешно загружена из {ppo_path}", force=True)
            elif ppo_path:
                self.debug_print(f"Файл PPO модели не найден по пути: {ppo_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при загрузке PPO модели: {e}", force=True)
        # 2. Загрузка Feature Net с защитой map_location
        try:
            if feature_net_path and os.path.exists(feature_net_path):
                # map_location гарантирует, что веса загрузятся на правильный девайс (CPU/CUDA)
                state_dict = torch.load(feature_net_path, map_location=self.device)
                self.feature_net.load_state_dict(state_dict)
                self.feature_net.to(self.device).float()
                self.debug_print(f"Feature net загружена из {feature_net_path}")
            elif feature_net_path:
                self.debug_print(f"Файл feature_net не найден: {feature_net_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при загрузке feature_net: {e}", force=True)
        # 3. Загрузка Forward Model с защитой map_location
        try:
            if forward_model_path and os.path.exists(forward_model_path):
                state_dict = torch.load(forward_model_path, map_location=self.device)
                self.forward_model.load_state_dict(state_dict)
                self.forward_model.to(self.device).float()
                self.debug_print(f"Forward model загружена из {forward_model_path}")
            elif forward_model_path:
                self.debug_print(f"Файл forward_model не найден: {forward_model_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при загрузке forward_model: {e}", force=True)
        # 4. Пересборка общего оптимизатора для Curiosity на основе новых загруженных параметров
        self.optimizer = torch.optim.Adam(
            list(self.feature_net.parameters()) + list(self.forward_model.parameters()), 
            lr=1e-3
        )
        # Синхронизируем статистику из конфига, чтобы шаги и награды не начинались с нуля
        try:
            config = self.load_config()
            self.total_reward_accumulated = config.get("total_reward", 0.0)
            self.frame_processed = config.get("frame_processed", 0)
            self.debug_print(f"Локальная статистика синхронизирована: Отработано фреймов={self.frame_processed}")
        except Exception as e:
            self.debug_print(f"Не удалось синхронизировать статистику конфига при загрузке: {e}")

    #Это ядро твоего генетического алгоритма.
    def evolve_model(self, profile_paths, elite_fraction=0.2, mutate=True):
        """
        Эволюция модели: выбирает профиль на основе производительности и применяет мутации к элите.
        """
        if not profile_paths:
            self.debug_print("Нет профилей для эволюции", force=True)
            return None
        # 1. Собираем производительность всех профилей
        performances = []
        for profile_path in profile_paths:
            config_path = os.path.join(profile_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        total_reward = config.get("total_reward", 0.0)
                        performances.append((profile_path, total_reward))
                except Exception as e:
                    self.debug_print(f"Ошибка чтения конфига {config_path}: {e}")
        if not performances:
            self.debug_print("Нет доступных профилей с валидными конфигами для эволюции", force=True)
            return None
        # Сортируем профили: лучшие (с максимальной наградой) в самом начале списка
        performances.sort(key=lambda x: x[1], reverse=True)
        sorted_paths = [p[0] for p in performances]
        # 2. Вместо слепого рандома делаем Турнирный отбор (выбираем 3 случайных и берем лучшего из них)
        # Это гарантирует, что у сильных профилей шанс размножиться выше, но сохраняется разнообразие
        tournament_size = min(3, len(sorted_paths))
        tournament_candidates = random.sample(sorted_paths, tournament_size)
        # Так как sorted_paths упорядочен, берем кандидата с минимальным индексом в нем (он самый сильный)
        selected_path = min(tournament_candidates, key=lambda p: sorted_paths.index(p))
        # 3. Теперь индекс в упорядоченном списке ТОЧНО определяет элитарность
        rank = sorted_paths.index(selected_path)
        is_elite = rank < max(1, int(len(sorted_paths) * elite_fraction))
        self.debug_print(f"Эволюция: выбран профиль {selected_path} (Ранг в поколении: {rank + 1}/{len(sorted_paths)}), is_elite={is_elite}", force=True)
        # 4. Умный поиск файлов моделей (поддерживает и фиксированные имена, и таймстемпы)
        all_files = os.listdir(selected_path)
        ppo_files = [f for f in all_files if f.endswith(".zip")]
        # Ищем feature_net.pth ИЛИ старые файлы feature_net_*.pth
        fn_files = [f for f in all_files if f == "feature_net.pth" or (f.startswith("feature_net_") and f.endswith(".pth"))]
        # Ищем forward_model.pth ИЛИ старые файлы forward_model_*.pth
        fw_files = [f for f in all_files if f == "forward_model.pth" or (f.startswith("forward_model_") and f.endswith(".pth"))]
        if not (ppo_files and fn_files and fw_files):
            self.debug_print(f"Критическая ошибка: в профиле {selected_path} отсутствует один из компонентов весов!", force=True)
            return None
        # Берем самые свежие файлы по дате изменения
        latest_ppo = max(ppo_files, key=lambda f: os.path.getmtime(os.path.join(selected_path, f)))
        latest_feature_net = max(fn_files, key=lambda f: os.path.getmtime(os.path.join(selected_path, f)))
        latest_forward_model = max(fw_files, key=lambda f: os.path.getmtime(os.path.join(selected_path, f)))
        # Загружаем веса в нейросети текущей среды
        self.load_training_state(
            ppo_path=os.path.join(selected_path, latest_ppo),
            feature_net_path=os.path.join(selected_path, latest_feature_net),
            forward_model_path=os.path.join(selected_path, latest_forward_model)
        )
        # 5. Применяем мутации, только если mutate=True и профиль РЕАЛЬНО элитный
        if mutate and is_elite:
            self.debug_print(f"Применяем направленную мутацию (+5% шума) к параметрам Curiosity элиты", force=True)
            with torch.no_grad():
                for param in self.feature_net.parameters():
                    noise = torch.randn_like(param) * 0.05
                    param.add_(noise)
                for param in self.forward_model.parameters():
                    noise = torch.randn_like(param) * 0.05
                    param.add_(noise)
            # Пересоздаем оптимизатор, так как тензоры мутировали
            self.optimizer = torch.optim.Adam(
                list(self.feature_net.parameters()) + list(self.forward_model.parameters()), 
                lr=1e-3
            )
        self.debug_print(f"Эволюция завершена штатно. Выбранный родительский профиль: {selected_path}")
        return selected_path

    #Логика очистки очередей, освобождения камеры, очистки кэша CUDA и мягкого глушения потоков
    def cleanup(self, is_absolute_end=False):
        self.debug_print(f"Очистка GameEnv для profile_id={self.profile_id}", force=True)
        # 1. Безопасное сохранение состояния без удваивания метрик
        self.save_training_state(is_cleanup=True)
        # 2. Логика инкремента агентов для генетического алгоритма
        next_agent = int(self.profile_id)
        # Переходим к следующему агенту, если этот полностью отработал шаги, 
        # ИЛИ если это штатный переход, переданный из управляющего цикла main
        if is_absolute_end or (self.total_steps_accumulated + self.frame_processed >= self.total_timesteps):
            next_agent += 1
            self.debug_print(f"Агент {self.profile_id} завершил лимит шагов. Переключаемся на Агента {next_agent}")
        else:
            self.debug_print(f"Экстренная остановка Агента {self.profile_id}. ID для сохранения: {next_agent}")
            
        save_global_state(self.game_name, self.generation, next_agent,
                          [self.get_profile_dir()], MAX_GENS, MAX_PROFILES)
        # 3. Сигнализируем всем потокам о немедленной остановке
        self.stop_event.set()
        self.pause_event.set()
        self.is_gui_destroyed = True
        # Отпускаем все залипшие виртуальные клавиши в системе
        self.release_all_keys()
        # 4. Сначала останавливаем циклы потоков (выставляем running = False)
        self.threads = [t for t in self.threads if t.is_alive()]
        for thread in self.threads:
            if thread.is_alive():
                if hasattr(thread, 'running'):
                    thread.running = False
        # 5. Освобождаем железные и программные ресурсы видеозахвата OBS
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.debug_print("Освобождение видеозахвата камеры OBS")
        # 6. Теперь, когда потоки больше ничего не пишут в очереди, спокойно вытряхиваем их
        for q in [self.frame_queue, self.action_queue, self.send_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        # 7. Джойним потоки, давая им до 3 секунд на закрытие
        for thread in self.threads:
            if thread.is_alive():
                self.debug_print(f"Ожидание завершения потока воркера: {thread.name}")
                thread.join(timeout=3.0)
                if thread.is_alive():
                    self.debug_print(f"Поток {thread.name} проигнорировал join, оставляем как демон")
        # 8. Финальная зачистка видеопамяти видеокарты
        if CLEAR_GPU_CACHE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.debug_print("Очищена память GPU (CLEAR_GPU_CACHE=True)")
        self.debug_print("Очистка среды полностью завершена", force=True)

    #В этом методе сосредоточена вся логика управления горячими клавишами.
    def console_listener(self, stop_event: threading.Event, pause_event: threading.Event):
        self.debug_print(f"Консольный слушатель запущен. Нажмите {BIND_PAUSE} для паузы/возобновления, {BIND_STOP} для остановки и сохранения, {BIND_RECORD} для записи/остановки демонстрации")
        while not stop_event.is_set() and not global_stop_event.is_set():
            try:
                # === 1. ОБРАБОТКА ПАУЗЫ / СНЯТИЯ С ПАУЗЫ ===
                if keyboard.is_pressed(BIND_PAUSE):
                    if self.recording_event.is_set():
                        self.debug_print(f"Нельзя использовать {BIND_PAUSE} во время записи демонстрации")
                        global_log_queue.put("Нельзя ставить на паузу/возобновлять во время записи")
                    elif not pause_event.is_set():
                        pause_event.set()
                        self.debug_print(f"Бот остановлен на {BIND_PAUSE}", force=True)
                        global_log_queue.put(f"Бот приостановлен\nНажмите {BIND_RECORD} для записи демо")
                    else:
                        if self.has_demo:
                            pause_event.clear()
                            self.debug_print(f"Бот возобновлен на {BIND_PAUSE}", force=True)
                            global_log_queue.put("Бот возобновлен")
                        else:
                            self.debug_print(f"Запуск невозможен: отсутствует демо. Запишите демонстрацию {BIND_RECORD}")
                            global_log_queue.put(f"Нельзя запустить: запишите демо на {BIND_RECORD}")
                    # Ждём отпускания клавиши PAUSE
                    while keyboard.is_pressed(BIND_PAUSE):
                        time.sleep(0.01)
                # === 2. ОБРАБОТКА ЗАПИСИ ДЕМОНСТРАЦИИ ===
                if keyboard.is_pressed(BIND_RECORD):
                    if not self.recording_active:
                        # --- старт записи ---
                        self.recording_event.set()
                        self.stop_recording_event.clear()
                        self.recording_active = True
                        threading.Thread(target=self.record_demo, daemon=True, name="DemoRecorder").start()
                        self.debug_print("Начата запись демонстрации")
                        global_log_queue.put("Начата запись демонстрации")
                    else:
                        # --- остановка записи ---
                        self.stop_recording_event.set()
                        self.recording_event.clear()
                        self.recording_active = False
                        self.debug_print("Остановлена запись демонстрации")
                        global_log_queue.put("Остановлена запись демонстрации")
                        # Потокобезопасно отправляем сигнал в GUI об изменении статуса
                        global_log_queue.put("UPDATE_STATUS_LABEL:Bot paused")
                    # Ждём отпускания клавиши RECORD
                    while keyboard.is_pressed(BIND_RECORD):
                        time.sleep(0.01)
                # === 3. ОБРАБОТКА СТОП-КЛАВИШИ (ВЫХОД) ===
                if keyboard.is_pressed(BIND_STOP):
                    self.debug_print(f"Клавиша {BIND_STOP} нажата. Инициирован экстренный выход...", force=True)
                    global_log_queue.put("Экстренная остановка... Сохранение состояния.")
                    # Записываем глобальное состояние, фиксируя текущего агента
                    profile_paths = [self.get_profile_dir()]
                    next_agent = int(self.profile_id)  
                    save_global_state(self.game_name, self.generation, next_agent, profile_paths, MAX_GENS, MAX_PROFILES)
                    # Включаем паузу (set) и стоп, чтобы мгновенно заморозить все параллельные воркеры
                    pause_event.set()
                    stop_event.set()
                    global_stop_event.set()  # Сигнализируем главному циклу main о выходе
                    # Ждём отпускания клавиши STOP, чтобы не спамить в консоль
                    while keyboard.is_pressed(BIND_STOP):
                        time.sleep(0.01)
                    return
                time.sleep(0.01)  # Разгружаем CPU
            except Exception as e:
                self.debug_print(f"Ошибка в цикле console_listener: {e}", force=True)
                time.sleep(0.5)
        sys.exit(0)
    
    #Логика записи демо
    def record_demo(self):
        """Запись видео и действий игрока в течение 30 мин или до остановки"""
        video_writer = None
        try:
            os.makedirs(os.path.dirname(self.demo_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(self.demo_video_path, fourcc, DEMO_VIDEO_FPS, (self.width, self.height))
            self.demo_actions = []
            start_time = time.time()
            last_action_time = start_time
            frame_duration = 1.0 / DEMO_VIDEO_FPS
            key_map = {
            'minus': '-', 'comma': ',', 'period': '.', 'slash': '/', 
            'tilde': '`', 'lbracket': '[', 'rbracket': ']', 'backslash': '\\',
            'apostrophe': "'", 'semicolon': ';'
            }
            # 1. Формируем список поддерживаемых клавиш для отслеживания
            tracked_keys = []
            seen_keys = set()
            for key in VK.keys():
                mapped_key = key_map.get(key, key)
                try:
                    keyboard.is_pressed(mapped_key)
                    if mapped_key not in seen_keys:
                        tracked_keys.append(mapped_key)
                        seen_keys.add(mapped_key)
                except ValueError:
                    continue
            # Отключаем ограничение курсора на время демонстрации, чтобы игрок мог управлять свободно        
            ctypes.windll.user32.ClipCursor(None)
            self.debug_print("Ограничение курсора временно снято для записи демонстрации", force=True)
            # 2. Основной цикл записи
            while (time.time() - start_time < DEMO_RECORD_DURATION and 
                   not self.stop_event.is_set() and 
                   not self.stop_recording_event.is_set()):
                frame_start = time.time()
                # Захват кадра
                ret, frame = self.cap.read()
                if ret:
                    video_writer.write(frame)
                    # Запись действий (клавиш и мыши) каждые 0.1 сек
                    current_time = time.time()
                    if current_time - last_action_time >= 0.1:
                        pressed_keys = set()
                        for key in tracked_keys:
                            try:
                                if keyboard.is_pressed(key):
                                    pressed_keys.add(key)
                            except ValueError as e:
                                continue
                        # Состояние кнопок мыши через Win32 API
                        mouse_buttons = []
                        if win32api.GetKeyState(0x01) < -1:  # ЛКМ
                            mouse_buttons.append("left")
                        if win32api.GetKeyState(0x02) < -1:  # ПКМ
                            mouse_buttons.append("right")
                        # позиция мыши
                        try:
                            cur_pos = win32api.GetCursorPos()
                            center = (self.width // 2, self.height // 2)
                            mouse_delta = [
                                cur_pos[0] - center[0],
                                cur_pos[1] - center[1]
                            ]
                            # Возвращаем курсор в центр чтобы считать следующую дельту
                            if mouse_delta[0] != 0 or mouse_delta[1] != 0:
                                win32api.SetCursorPos(center)
                        except Exception:
                            mouse_delta = [0, 0]

                        # Сохраняем снимок действия (УБРАН СПАМ-ПРИНТ КООРДИНАТ)
                        self.demo_actions.append({
                            'timestamp': current_time - start_time,
                            'keys': list(pressed_keys),
                            'mouse_pos': mouse_delta,
                            'mouse_buttons': mouse_buttons
                        })
                        last_action_time = current_time
                else:
                    # Мягкий лог без засорения консоли при единичных пропусках
                    time.sleep(0.005)
                    continue
                # Динамический расчет задержки для удержания стабильного FPS видео
                elapsed = time.time() - frame_start
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
            # 3. Сохраняем собранные действия в файл JSON
            if self.demo_actions:
                try:
                    with open(self.demo_actions_path, 'w') as f:
                        json.dump(self.demo_actions, f, indent=4)
                    self.debug_print(f"Действия игрока ({len(self.demo_actions)} шагов) сохранены в json", force=True)
                    self.has_demo = True  # Обновляем флаг, так как демонстрация записана
                    # Запускаем анализ записанного поведения игрока
                    if hasattr(self, 'analyze_demo_behavior'):
                        self.analyze_demo_behavior()
                except Exception as e:
                    self.debug_print(f"Ошибка при сохранении файла действий: {e}", force=True)
            else:
                self.debug_print("Предупреждение: Запись завершена, но не было перехвачено ни одного действия!", force=True)
        except Exception as e:
            self.debug_print(f"Критическая ошибка в record_demo: {e}", force=True)
        finally:
            # Гарантированное освобождение ресурсов
            if video_writer is not None:
                video_writer.release()
            self.recording_event.clear()
            self.recording_active = False
            self.debug_print("Поток записи демо завершил работу, файлы закрыты.", force=True)
            # Восстанавливаем фиксацию мыши в окне игры
            if hasattr(self, 'restrict_cursor_to_window'):
                self.restrict_cursor_to_window()

    #Загрузка сохраненных действий из демонстрации игрока для IRL обучения
    def load_demo_actions(self):
        try:
            if not os.path.exists(self.demo_actions_path):
                self.debug_print(f"Файл демонстрации не найден по пути: {self.demo_actions_path}")
                self.has_demo = False
                self.demo_actions = []
                return False
            with open(self.demo_actions_path, 'r') as f:
                loaded_actions = json.load(f)
            # Валидация: проверяем, что это список и он не пустой
            if not isinstance(loaded_actions, list) or len(loaded_actions) == 0:
                self.debug_print(f"Файл демки {self.demo_actions_path} поврежден или пуст!", force=True)
                self.has_demo = False
                self.demo_actions = []
                return False
            self.demo_actions = loaded_actions
            self.has_demo = True  
            self.debug_print(f"Успешно загружено {len(self.demo_actions)} шагов демонстрации для IRL", force=True)
            # Запуск анализа распределения действий игрока
            if hasattr(self, 'analyze_demo_behavior'):
                self.analyze_demo_behavior()
            return True
        except json.JSONDecodeError as jde:
            self.debug_print(f"Ошибка парсинга JSON в файле демонстрации: {jde}", force=True)
            self.has_demo = False
            self.demo_actions = []
            return False
        except Exception as e:
            self.debug_print(f"Непредвиденная ошибка при load_demo_actions: {e}", force=True)
            self.has_demo = False
            self.demo_actions = []
            return False

    # Загрузка сохраненных действий из демонстрации игрока для IRL обучения
    def pretrain_with_demo(self, model):
        if not self.has_demo or not self.demo_actions:
            self.debug_print("Нет демонстраций для предобучения")
            return

        if not hasattr(model.policy, 'optimizer') or model.policy.optimizer is None:
            self.debug_print("Оптимизатор SB3 не инициализирован, пропуск pretrain")
            return

        def map_demo_action_to_index(demo_action):
            keys = demo_action.get('keys', [])
            mouse_buttons = demo_action.get('mouse_buttons', [])
            mouse_pos = demo_action.get('mouse_pos', [0, 0])
            dx = mouse_pos[0] if len(mouse_pos) > 0 else 0
            dy = mouse_pos[1] if len(mouse_pos) > 1 else 0
            keys_set = set(keys)
            if 'shift' in keys_set and 'w' in keys_set:
                if dx < -20: return 22
                if dx > 20:  return 23
                return 21
            if 'ctrl' in keys_set and 'w' in keys_set:
                return 24
            if 'w' in keys_set:
                if 'left' in mouse_buttons: return 19
                if dx < -20: return 17
                if dx > 20:  return 18
                return 1
            if 'a' in keys_set: return 2
            if 's' in keys_set: return 3
            if 'd' in keys_set: return 4
            if 'space' in keys_set: return 5
            if 'shift' in keys_set: return 6
            if 'ctrl' in keys_set: return 7
            if 'e' in keys_set: return 8
            if 'left' in mouse_buttons: return 9
            if 'right' in mouse_buttons: return 10
            if dx < -30: return 11
            if dx > 30:  return 12
            if dy < -30: return 13
            if dy > 30:  return 14
            return 0

        fps = DEMO_VIDEO_FPS
        segment_duration = 15
        NUM_SEGMENTS = 3

        cap = cv2.VideoCapture(self.demo_video_path)
        if not cap.isOpened():
            self.debug_print("Не удалось открыть видео демонстрации")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            self.debug_print("Видео демки пустое")
            cap.release()
            return

        total_duration = total_frames / fps
        segment_frames = int(segment_duration * fps)
        max_start_frame = max(0, total_frames - segment_frames)

        self.debug_print(
            f"Pretrain: {NUM_SEGMENTS} сегмента по {segment_duration}s "
            f"из {total_duration:.1f}s total"
        )

        model.policy.train()
        total_trained = 0

        for seg in range(NUM_SEGMENTS):
            start_frame = random.randint(0, max_start_frame)
            start_time_sec = start_frame / fps
            end_time_sec = start_time_sec + segment_duration

            segment_actions = [
                a for a in self.demo_actions
                if start_time_sec <= a['timestamp'] <= end_time_sec
            ]
            if not segment_actions:
                self.debug_print(f"Сег {seg+1}: нет действий, пропуск")
                continue

            self.debug_print(
                f"Сег {seg+1}/{NUM_SEGMENTS}: {start_time_sec:.1f}s–{end_time_sec:.1f}s | "
                f"кадр {start_frame}/{total_frames} | действий: {len(segment_actions)}"
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(segment_frames):
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    frames.append(np.zeros((84, 84, 3), dtype=np.uint8))
                    continue
                try:
                    frames.append(cv2.resize(frame, (84, 84)))
                except Exception:
                    frames.append(np.zeros((84, 84, 3), dtype=np.uint8))

            if not frames:
                self.debug_print(f"Сег {seg+1}: пустой, пропуск")
                continue

            trained = 0
            prev_action_idx = -1

            for i, frame in enumerate(frames):
                current_time = start_time_sec + i / fps
                closest_action = min(
                    segment_actions,
                    key=lambda x: abs(x['timestamp'] - current_time),
                    default=None
                )
                if not closest_action:
                    continue

                action_idx = map_demo_action_to_index(closest_action)

                if action_idx == prev_action_idx and i % 10 != 0:
                    continue
                prev_action_idx = action_idx

                try:
                    obs = self.get_state(frame)
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    if obs_tensor.ndim == 3 and obs_tensor.shape[-1] == 3:
                        obs_tensor = obs_tensor.permute(2, 0, 1)
                    obs_input = obs_tensor.unsqueeze(0)

                    dist = model.policy.get_distribution(obs_input)
                    logits = dist.distribution.logits
                    target = torch.tensor([action_idx], dtype=torch.long, device=self.device)
                    loss = nn.CrossEntropyLoss()(logits, target)

                    model.policy.optimizer.zero_grad()
                    loss.backward()
                    model.policy.optimizer.step()

                    trained += 1

                    if trained % 30 == 0:
                        self.debug_print(
                            f"Сег {seg+1} | кадр {i}/{len(frames)} | "
                            f"t={current_time:.1f}s | action={action_idx} | "
                            f"loss={loss.item():.4f}"
                        )
                except Exception as e:
                    self.debug_print(f"Ошибка pretrain сег {seg+1} кадр {i}: {e}")

            total_trained += trained
            self.debug_print(f"Сег {seg+1} завершён: {trained} кадров обучено")
            time.sleep(0.3)

        cap.release()
        self.demo_access_count += 1
        self.debug_print(f"Pretrain завершён: {total_trained} кадров по {NUM_SEGMENTS} сегментам")
        
    # Это функция возвращает дескриптор окна, которое прямо сейчас находится в фокусе пользователя
    def detect_game_window_name(self):
        if not hasattr(self, 'game_name') or not self.game_name:
            # Если имя игры вообще не задано, откатываемся на активное окно
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            return title if title else "UnknownGame"
        target_keyword = self.game_name.lower()
        found_titles = []
        # Функция-колбэк для перебора всех окон в Windows
        def enum_windows_callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                # Проверяем, содержит ли заголовок окна имя нашей игры
                if target_keyword in title.lower():
                    found_titles.append(title)
            return True
        try:
            win32gui.EnumWindows(enum_windows_callback, None)
        except Exception:
            pass
        # Если нашли совпадение — берем его, если нет — бот не падает, а ищет активное
        if found_titles:
            # Возвращаем первое найденное совпадение
            return found_titles[0]
        # Запасной вариант: если игра не запущена, смотрим что сейчас на экране
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        return title if title else "UnknownGame"

    #Устанавливает описание игры для ИИ-агента и сохраняет его в конфиг профиля.
    def set_game_description(self, desc):
        self.game_description = desc
        try:
            # Загружаем текущий конфиг
            config = self.load_config()
            if config is None:
                config = {}
            config["game_description"] = desc
            # Сохраняем обновленный конфиг
            self.save_config(config)
            self.debug_print(f"Описание игры успешно обновлено и сохранено в конфиг.")
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении описания игры в конфиг: {e}")

    #Автоматический сбор контекста из Википедии через REST API
    def fetch_game_description(self, lang="en"):
        try:
            base_name = self.game_name.strip()
            # Шаг 1: Пробуем найти по чистому имени
            game = self.game_name.replace(" ", "_")
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{game}"
            response = requests.get(url, timeout=5, headers={"User-Agent": "AI-GameBot"})
            if response.status_code == 200:
                data = response.json()
                desc = data.get("extract", "")
                if desc:
                    self.set_game_description(desc)
                    return desc
            # Шаг 2: Fallback (если 404 или нет описания), пробуем с припиской "_(video_game)"
            self.debug_print(f"Прямая ссылка не найдена, пробуем поиск с тегом (video game)...")
            game_fallback = f"{base_name}_(video_game)".replace(" ", "_")
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{game_fallback}"
            response = requests.get(url, timeout=5, headers={"User-Agent": "GameBot/1.0"})
            if response.status_code == 200:
                data = response.json()
                desc = data.get("extract", "")
                if desc:
                    self.set_game_description(desc)
                    return desc
            # Если оба варианта не принесли результата
            msg_not_found = f"Описание игры не найдено в Википедии ({lang})."
            self.set_game_description(msg_not_found)
            return msg_not_found
        except Exception as e:
            msg_error = f"Описание игры недоступно (ошибка API: {e})"
            self.set_game_description(msg_error)
            return msg_error

    #анализ текста на основе правил, чтобы агент сам понимал, какие высокоуровневые задачи перед ним стоят, опираясь на выкачанное из Википедии описание.
    def analyze_game_description(self):
        if not self.game_description:
            return [], False
        description = self.game_description.lower()
        game_name_lower = self.game_name.lower()
        shooter_keywords = ["shoot", "shooter", "first-person", "fps", "third-person", "tps", "kill", "eliminate", "aim", "weapon", "gun", "battle", "combat", "fire", "reload", "tactical"]
        # Проверяем, относится ли игра к шутерам (по тексту или известным тегам)
        known_shooters = ["counter-strike", "call of duty", "battlefield", "valorant", "overwatch", "cs2", "quake", "doom"]
        is_shooter = any(kw in description for kw in shooter_keywords) or any(shooter in game_name_lower for shooter in known_shooters)
        # Базовые универсальные действия, которые есть вообще в любой игре
        keywords = ["move", "jump", "interact", "survive", "attack", "collect"]
        if is_shooter:
            # Для шутеров добавляем специфичные боевые триггеры, сохраняя базу
            keywords.extend(["shoot", "kill", "eliminate", "plant", "defuse", "objective", "aim", "reload"])
        else:
            # Для песочниц, RPG и выживалок добавляем крафт и исследование
            keywords.extend(["build", "craft", "collect", "deliver"])
        # Специфичные триггеры для Minecraft (оставляем как пасхалку/исключение для тестов)
        if "minecraft" in game_name_lower:
            keywords.extend(["ender dragon", "end portal", "ender pearl", "blaze rod"])
        # Очищаем список от дубликатов (если они совпали при extend)
        keywords = list(set(keywords))
        # Поиск совпадений в тексте Википедии
        found_keywords = [kw for kw in keywords if kw in description]
        self.debug_print(f"Жанр: {'Шутер' if is_shooter else 'Песочница/Другое'} | Найденные маркеры целей: {found_keywords}")
        return found_keywords, is_shooter

    #Универсально и точно находит координаты (rect) окна игры.
    def get_game_window_rect(self):
        target_hwnd = None
        if hasattr(self, 'game_name') and self.game_name:
            target_keyword = self.game_name.lower()
            
            def enum_windows_callback(hwnd, extra):
                nonlocal target_hwnd
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if target_keyword in title.lower():
                        target_hwnd = hwnd
                        return False  # Нашли окно, прерываем поиск
                return True
            try:
                win32gui.EnumWindows(enum_windows_callback, None)
            except Exception:
                pass
        # Шаг 2: Если по имени ничего не нашли, берем текущее активное окно (fallback)
        if not target_hwnd:
            target_hwnd = win32gui.GetForegroundWindow()
        if not target_hwnd:
            self.debug_print("Целевое окно не найдено в системе.")
            return None
        # Проверяем видимость окна
        style = win32gui.GetWindowLong(target_hwnd, win32con.GWL_STYLE)
        if not (style & win32con.WS_VISIBLE):
            self.debug_print("Целевое окно свернуто или невидимо.")
            return None
        # Шаг 3: Точное получение координат через DWM (без учета системных теней)
        try:
            rect = wintypes.RECT()
            DWMWA_EXTENDED_FRAME_BOUNDS = 9
            result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(target_hwnd),
                wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
                ctypes.byref(rect),
                ctypes.sizeof(rect)
            )
            if result == 0:  # 0 означает S_OK (успех)
                # Возвращаем кортеж в классическом формате (left, top, right, bottom)
                return (rect.left, rect.top, rect.right, rect.bottom)
        except Exception as e:
            self.debug_print(f"DwmGetWindowAttribute не поддерживается или вызвал ошибку: {e}. Откат на GetWindowRect.")
        # Старый запасной вариант, если DWM почему-то дал сбой
        try:
            return win32gui.GetWindowRect(target_hwnd)
        except Exception as e:
            self.debug_print(f"Критическая ошибка получения координат окна: {e}")
            return None

    #удержание (локирование) курсора мыши внутри игрового окна.
    def restrict_cursor_to_window(self):
        if self.ui_mode:
            rect_tuple = self.get_game_window_rect()
            if rect_tuple:
                try:
                    # Распаковываем координаты из нашего кортежа
                    left, top, right, bottom = rect_tuple
                    # Создаем структуру RECT, которую ожидает Windows API
                    rect = wintypes.RECT(left, top, right, bottom)
                    # Блокируем курсор внутри созданного прямоугольника
                    ctypes.windll.user32.ClipCursor(ctypes.byref(rect))
                except Exception as e:
                    self.debug_print(f"Не удалось заблокировать курсор: {e}")
            else:
                # Если окно игры не найдено, принудительно освобождаем курсор
                ctypes.windll.user32.ClipCursor(None)
        else:
            # Если ui_mode отключен — возвращаем мыши полную свободу
            ctypes.windll.user32.ClipCursor(None)

    #Оптимизированный фоновый поток для работы YOLO детекции.
    def _yolo_worker(self):
        results = None
        last_inference_time = 0
        INFERENCE_INTERVAL = 0.05  # 20 инференсов/сек — баланс скорости и нагрузки

        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                now = time.time()
                if now - last_inference_time < INFERENCE_INTERVAL:
                    time.sleep(0.005)
                    continue

                # Берём САМЫЙ СВЕЖИЙ кадр — без накопления очереди
                with self.lock:
                    frame = self.current_frame
                    if frame is not None:
                        frame = frame.copy()

                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue

                try:
                    with torch.inference_mode():
                        results = self.model_yolo(
                            frame, 
                            classes=[0],
                            conf=0.25,
                            imgsz=320,
                            verbose=False,
                            device=self.yolo_device,
                        )
                    last_inference_time = time.time()
                except Exception as e:
                    self.debug_print(f"YOLO inference error: {e}")
                    time.sleep(0.05)
                    continue

                # Аннотация для оверлея
                annotated = frame
                if ENABLE_GUI and results and len(results) > 0:
                    try:
                        annotated = results[0].plot()
                    except Exception:
                        pass

                try:
                    resized = cv2.resize(annotated, (self.width, self.height))
                except Exception:
                    resized = frame

                with self.lock:
                    if results is not None:
                        self.yolo_results = results
                        self.yolo_last_update = time.time()
                    self.annotated_frame = resized

            except Exception as e:
                self.debug_print(f"YOLO worker critical error: {e}")
                time.sleep(0.1)

    #Фоновый поток отправки команд ввода (клавиатура/мышь) через SendInput.
    def _send_worker(self):
        """
        Фоновый поток отправки команд ввода (клавиатура/мышь) через SendInput.
        Ориентируется на универсальный поиск окна.
        """
        # Локальный кэш для дескриптора окна, чтобы не искать его миллисекунду
        last_hwnd = None
        last_focus_check = 0
        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
                    self.release_all_keys()
                    time.sleep(0.1)
                    continue
                # Забираем элемент из очереди (без лишних таймаутов, если пусто — уйдет в исключение)
                item = self.send_queue.get(timeout=0.01)
                typ = item.get('type')
                # Шаг 1: Проверка и удержание фокуса окна игры
                current_time = time.time()
                # Проверяем фокус не чаще, чем раз в 0.5 секунд, чтобы не спамить ОС
                if current_time - last_focus_check > 0.5:
                    rect = self.get_game_window_rect()
                    
                    if rect:
                        # Используем наш универсальный метод поиска окна, который мы писали ранее.
                        # Предполагается, что get_game_window_rect или аналогичный метод сохраняет/знает целевой HWND.
                        # Если в твоем классе HWND не сохранялся, найдем его по rect через окно в фокусе,
                        # либо переиспользуем логикуEnumWindows.
                        
                        # Для простоты: если игра запущена, SetForegroundWindow нужна только если мы улетели из неё
                        current_foreground = win32gui.GetForegroundWindow()
                        
                        # Переключаем фокус, только если активное окно сменилось и игра потеряла приоритет
                        # (Проверку по имени делаем через твой базовый game_name)
                        if self.game_name.lower() not in win32gui.GetWindowText(current_foreground).lower():
                            # Находим HWND окна, заголовок которого содержит имя нашей игры
                            def enum_cb(hwnd, extra):
                                nonlocal current_foreground
                                if win32gui.IsWindowVisible(hwnd) and self.game_name.lower() in win32gui.GetWindowText(hwnd).lower():
                                    extra.append(hwnd)
                                return True
                            
                            hwnds = []
                            win32gui.EnumWindows(enum_cb, hwnds)
                            
                            if hwnds:
                                hwnd = hwnds[0]
                                try:
                                    win32gui.SetForegroundWindow(hwnd)
                                    time.sleep(0.02)
                                except Exception:
                                    pass # Windows может заблокировать фокус, если мы пишем код параллельно
                    
                    last_focus_check = current_time

                # Шаг 2: Получаем rect для валидации абсолютных координат мыши
                rect = self.get_game_window_rect()

                # Шаг 3: Обработка команд из очереди
                if typ == 'press_keys':
                    vks = item.get('vks', [])
                    inputs = [make_key_input(vk, down=True) for vk in vks]
                    if inputs:
                        send_input(inputs)
                        for vk in vks:
                            global_log_queue.put(f"Нажата клавиша: {vk_to_name(vk)}")

                elif typ == 'release_keys':
                    vks = item.get('vks', [])
                    inputs = [make_key_input(vk, down=False) for vk in vks]
                    if inputs:
                        send_input(inputs)
                        for vk in vks:
                            global_log_queue.put(f"Отпущена клавиша: {vk_to_name(vk)}")

                elif typ == 'mouse_move':
                    dx = item.get('dx', 0)
                    dy = item.get('dy', 0)
                    if dx != 0 or dy != 0:
                        send_input([make_mouse_move(dx, dy)])
                        global_log_queue.put(f"Движение мыши: dx={dx}, dy={dy}")

                elif typ == 'mouse_move_absolute':
                    x = item.get('x')
                    y = item.get('y')
                    if rect and x is not None and y is not None:
                        new_x = max(rect[0], min(rect[2] - 1, x))
                        new_y = max(rect[1], min(rect[3] - 1, y))
                        send_input([set_mouse_position(new_x, new_y)])
                        global_log_queue.put(f"Абсолютное движение мыши: x={new_x}, y={new_y}")
                    else:
                        if x is not None and y is not None:
                            send_input([set_mouse_position(x, y)])
                            global_log_queue.put(f"Абсолютное движение мыши: x={x}, y={y}")

                elif typ == 'mouse_click':
                    btn = item.get('button', 'left')
                    down = item.get('down', True)
                    
                    send_input([make_mouse_click(btn, down)])
                    
                    # Если это был клик (down=True), даем микро-паузу прямо внутри потока отправки,
                    # чтобы игра успела его считать, прежде чем прилетит команда на отпускание
                    if down:
                        time.sleep(0.03) 
                        
                    click_type = "нажатие" if down else "отпускание"
                    global_log_queue.put(f"{click_type} кнопки мыши ({btn})")

                # Микро-сон для разгрузки ядра процессора
                time.sleep(0.001)

            except queue.Empty:
                continue
            except Exception as e:
                self.debug_print(f"Send worker error: {e}")

    #мозг высокоуровневой логики твоих действий. Он принимает абстрактное задание на действие, разбивает сложные движения мыши на плавные микро-шаги
    def _action_worker(self):
        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    action_item = self.action_queue.get(timeout=0.02)
                except queue.Empty:
                    continue

                dx, dy   = action_item.get('mouse_move', (0, 0))
                click    = action_item.get('mouse_click', None)
                duration = action_item.get('duration', 0.15)

                # --- плавное движение мыши ---
                if dx != 0 or dy != 0:
                    steps = max(1, int(duration / 0.016))  # шаги по ~16мс
                    remainder_x = 0.0
                    remainder_y = 0.0
                    step_sleep = duration / steps
                    for _ in range(steps):
                        exact_dx = (dx / steps) + remainder_x
                        exact_dy = (dy / steps) + remainder_y
                        sub_dx = int(exact_dx)
                        sub_dy = int(exact_dy)
                        remainder_x = exact_dx - sub_dx
                        remainder_y = exact_dy - sub_dy
                        if sub_dx != 0 or sub_dy != 0:
                            send_input([make_mouse_move(sub_dx, sub_dy)])
                        time.sleep(step_sleep)

                # --- клик (только если нет движения, иначе aim_at_person сам кликает) ---
                if click and not click.startswith('hold_'):
                    btn = 'left' if 'left' in click else 'right'
                    send_input([make_mouse_click(btn, True)])
                    time.sleep(0.1)   # держим 100мс — игра точно зарегистрирует
                    send_input([make_mouse_click(btn, False)])
                    time.sleep(0.05)  # пауза после отпускания

                elif click and click.startswith('hold_'):
                    btn = 'left' if 'left' in click else 'right'
                    if btn.upper() not in self.held_mouse:
                        self.held_mouse.add(btn.upper())
                        send_input([make_mouse_click(btn, True)])

            except Exception as e:
                self.debug_print(f"Action worker error: {e}")
                
    def _aim_worker(self):
        AIM_HZ = 30
        AIM_INTERVAL = 1.0 / AIM_HZ

        Kp = 0.25
        Ki = 0.01
        Kd = 0.08

        integral_x = 0.0
        integral_y = 0.0
        prev_dx = 0.0
        prev_dy = 0.0
        integral_limit = 50.0
        shoot_threshold = None

        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
                    integral_x = integral_y = 0.0
                    prev_dx = prev_dy = 0.0
                    time.sleep(0.1)
                    continue

                with self.lock:
                    yolo_results = getattr(self, 'yolo_results', None)
                    yolo_time = getattr(self, 'yolo_last_update', 0)

                if time.time() - yolo_time > 0.3:
                    integral_x = integral_y = 0.0
                    time.sleep(AIM_INTERVAL)
                    continue

                if not self.is_shooter or yolo_results is None:
                    time.sleep(AIM_INTERVAL)
                    continue

                if len(yolo_results[0].boxes.data) == 0:
                    integral_x = integral_y = 0.0
                    prev_dx = prev_dy = 0.0
                    time.sleep(AIM_INTERVAL)
                    continue

                detections = yolo_results[0].boxes.data.cpu().numpy()
                center_x_frame = self.width / 2
                center_y_frame = self.height / 2

                # Ищем ближайшего к центру person вне мёртвых зон
                closest = None
                closest_det = None
                min_dist = float('inf')
                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    if not self._is_valid_detection(x1, y1, x2, y2):
                        continue
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    d = ((cx - center_x_frame) ** 2 + (cy - center_y_frame) ** 2) ** 0.5
                    if d < min_dist:
                        min_dist = d
                        closest = (cx, cy)
                        closest_det = det

                if closest is None or closest_det is None:
                    integral_x = integral_y = 0.0
                    time.sleep(AIM_INTERVAL)
                    continue

                dx = closest[0] - center_x_frame
                dy = closest[1] - center_y_frame

                if shoot_threshold is None:
                    shoot_threshold = max(15, int(self.width * 0.025))

                # Выстрел только если центр кадра физически внутри bbox персоны
                x1, y1, x2, y2 = closest_det[:4]
                crosshair_on_target = (x1 < center_x_frame < x2 and y1 < center_y_frame < y2)

                if crosshair_on_target:
                    global_log_queue.put(f"🎯 Прицел внутри bbox — выстрел! dist={min_dist:.0f}")
                    send_input([make_mouse_click('left', True)])
                    time.sleep(0.08)
                    send_input([make_mouse_click('left', False)])
                    integral_x = integral_y = 0.0
                    prev_dx = prev_dy = 0.0
                    time.sleep(0.25)
                    continue

                # PID вычисление
                p_x = Kp * dx
                p_y = Kp * dy

                integral_x = max(-integral_limit, min(integral_limit, integral_x + dx))
                integral_y = max(-integral_limit, min(integral_limit, integral_y + dy))
                i_x = Ki * integral_x
                i_y = Ki * integral_y

                d_x = Kd * (dx - prev_dx)
                d_y = Kd * (dy - prev_dy)
                prev_dx = dx
                prev_dy = dy

                out_x = p_x + i_x + d_x
                out_y = p_y + i_y + d_y

                max_step = 100
                out_x = max(-max_step, min(max_step, out_x))
                out_y = max(-max_step, min(max_step, out_y))

                mouse_dx = int(out_x)
                mouse_dy = int(out_y)

                if mouse_dx != 0 or mouse_dy != 0:
                    send_input([make_mouse_move(mouse_dx, mouse_dy)])
                    global_log_queue.put(
                        f"🎯 PID аим: dx={mouse_dx} dy={mouse_dy} "
                        f"dist={min_dist:.0f} P=({p_x:.1f},{p_y:.1f})"
                    )

                time.sleep(AIM_INTERVAL)

            except Exception as e:
                self.debug_print(f"_aim_worker error: {e}")
                time.sleep(0.1)
            
    #мост между низкоуровневым потоком захвата экрана и всей остальной логикой бота
    def frame_capture_worker(self):
        last_processed_frame = None
        while not self.stop_event.is_set():
            try:
                new_frame = None
                # Шаг 1: Максимально быстро забираем кадр и СРАЗУ отпускаем первый лок
                with self.capture_thread.lock:
                    if self.capture_thread.latest_frame is not None:
                        # Сравниваем ссылки, чтобы не копировать один и тот же статичный кадр
                        if self.capture_thread.latest_frame is not last_processed_frame:
                            new_frame = self.capture_thread.latest_frame.copy()
                            last_processed_frame = self.capture_thread.latest_frame
                # Если нового кадра нет, просто спим и идем на следующий круг
                if new_frame is None:
                    time.sleep(0.002)
                    continue
                # Шаг 2: Под вторым локом сохраняем и распределяем кадр
                with self.lock:
                    self.current_frame = new_frame
                    # Если бот на паузе — в очередь обучения кадр не пихаем
                    if not self.pause_event.is_set():
                        # Быстро освобождаем место, если очередь забита
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self.frame_queue.put(self.current_frame, timeout=0.005)
                        except queue.Full:
                            pass
                # Небольшой отдых, согласованный с частотой обновления кадров
                time.sleep(0.002)
            except Exception as e:
                self.debug_print(f"Ошибка в frame_capture_worker: {e}")
                time.sleep(0.02)
    
    #метод проверяет, зажата ли клавиша, и если нет
    def hold_key(self, key_name):
        vk = key_to_vk(key_name)
        if not vk:
            return False
        with self.held_keys_lock:
            if key_name not in self.held_keys:
                self.held_keys.add(key_name)
                self.send_queue.put({'type':'press_keys','vks':[vk]})
        return True
    
    #Универсальное освобождение зажатой клавиши.
    def release_key(self, key_name):
        vk = key_to_vk(key_name)
        if not vk:
            return False
        with self.held_keys_lock:
            if key_name in self.held_keys:
                self.held_keys.remove(key_name)
                self.send_queue.put({'type':'release_keys','vks':[vk]})
        return True

    #Этот метод предназначен для прожатия комбинаций клавиш
    def press_combo(self, keys_list, hold_time=0.05):
        if not keys_list:
            return
        self.action_queue.put({
            'keys': keys_list,
            'mouse_move': (0, 0),
            'mouse_click': None,
            'duration': hold_time
        })

    #Безопасный клик мышью.
    def click(self, button='left'):
        btn = 'left' if button=='left' else 'right'
        self.action_queue.put({
            'keys': [],
            'mouse_move': (0, 0),
            'mouse_click': btn,  # Передаем просто строку 'left' или 'right'
            'duration': 0.02     # Время удержания клика внутри воркера
        })

    #Безопасное и плавное смещение мыши (камеры)
    def move_mouse(self, dx, dy):
        if dx == 0 and dy == 0:
            return
        self.action_queue.put({
            'keys': [],
            'mouse_move': (dx, dy),
            'mouse_click': None,
            'duration': duration,
            'human_mouse_smooth': False  # Можно включить True, если у тебя там прописана кривая Безье
        })

    #Экстренное освобождение всех зажатых клавиш клавиатуры и кнопок мыши.
    def release_all_keys(self):
        def enum_cb(hwnd_loc, extra):
            if win32gui.IsWindowVisible(hwnd_loc) and self.game_name.lower() in win32gui.GetWindowText(hwnd_loc).lower():
                extra.append(hwnd_loc)
            return True
        hwnds = []
        win32gui.EnumWindows(enum_cb, hwnds)
        if hwnds:
            game_hwnd = hwnds[0]
            current_foreground = win32gui.GetForegroundWindow()
            # Если игра сейчас не на переднем плане — возвращаем ей фокус
            if current_foreground != game_hwnd:
                try:
                    win32gui.SetForegroundWindow(game_hwnd)
                    self.debug_print("Фокус возвращён игровому окну перед освобождением клавиш")
                    time.sleep(0.05)  # Даем Windows немного времени на переключение фокуса
                except Exception:
                    self.debug_print("Не удалось вернуть фокус игровому окну")
        # Отпускаем клавиши клавиатуры
        with self.held_keys_lock:
            if self.held_keys:
                for key in list(self.held_keys):
                    vk = key_to_vk(key)
                    if vk is not None:
                        self.send_queue.put({'type': 'release_keys', 'vks': [vk]})
                        self.debug_print(f"Экстренно отпущена клавиша: {key}")
                self.held_keys.clear()
        # Отпускаем кнопки мыши
        if self.held_mouse:
            for button in list(self.held_mouse):
                self.send_queue.put({'type': 'mouse_click', 'button': button, 'down': False})
                self.debug_print(f"Экстренно отпущена кнопка мыши: {button}")
            self.held_mouse.clear()
        # Даём время воркеру отправки успеть переварить эти команды
        time.sleep(0.05)

    #Передает задачу плавного смещения прицела (относительно текущей позиции)
    def smooth_mouse_move(self, target_dx, target_dy, total_time=0.15):
        if target_dx == 0 and target_dy == 0:
            return
        # Вместо ручного цикла со sleep, мы просто отдаем задачу в action_worker.
        self.action_queue.put({
            'keys': [],
            'mouse_move': (int(target_dx), int(target_dy)),
            'mouse_click': None,
            'duration': total_time,
            'human_mouse_smooth': True  # Включаем сглаживание (человеческий профиль)
        })
        global_log_queue.put(f"Задано плавное смещение прицела: dx={target_dx}, dy={target_dy}")

    #Плавное перемещение прицела к цели с симуляцией человеческой микро-моторики.
    def human_mouse_move_smooth(self, target_dx, target_dy, total_time=0.2):
        if target_dx == 0 and target_dy == 0:
            return
        # Настоящий человек не двигает мышь линейно. Он сначала ускоряется, а ближе к цели замедляется (профиль скорости Изинга / колоколообразная кривая).
        steps = max(5, int(total_time / 0.015))  # Шаги по ~15 мс
        # Генерируем коэффициенты распределения скорости (синусоидальный профиль)
        factors = []
        for i in range(steps):
            # Аргумент от 0 до Pi дает красивую плавную дугу скорости
            factors.append(math.sin((i / steps) * math.pi))
        total_factor = sum(factors)
        # Разбиваем общий путь на шаги согласно профилю скорости + добавляем микро-шум руки
        remainder_x = 0.0
        remainder_y = 0.0
        for i in range(steps):
            # Базовая доля движения на этом шаге
            step_share = factors[i] / total_factor
            # Добавляем крошечный случайный шум руки (микро-дрожание в пределах 1-2 пикселей) Шум уменьшается по мере приближения к цели
            noise_reduction = 1.0 - (i / steps)
            hand_noise_x = random.uniform(-1.5, 1.5) * noise_reduction
            hand_noise_y = random.uniform(-1.0, 1.0) * noise_reduction
            exact_dx = (target_dx * step_share) + hand_noise_x + remainder_x
            exact_dy = (target_dy * step_share) + hand_noise_y + remainder_y
            sub_dx = int(exact_dx)
            sub_dy = int(exact_dy)
            remainder_x = exact_dx - sub_dx
            remainder_y = exact_dy - sub_dy
            if sub_dx != 0 or sub_dy != 0:
                self.send_queue.put({'type': 'mouse_move', 'dx': sub_dx, 'dy': sub_dy})
            # Спим внутри воркера, если метод вызван асинхронно, либо если мы вызываем это прямо из action_worker.
            time.sleep(total_time / steps)
    
    #Этот метод — ядро твоего аимбота. Логика поиска ближайшей к прицелу цели через дельту евклидова расстояния
    def aim_at_person(self, yolo_results):
        if not self.hwnd:
            return
        if not yolo_results or len(yolo_results[0].boxes.data) == 0:
            return

        # Координаты bbox уже в пространстве оригинального кадра (self.width x self.height)
        center_x_frame = self.width / 2
        center_y_frame = self.height / 2

        detections = yolo_results[0].boxes.data.cpu().numpy()

        closest_person = None
        min_distance = float('inf')

        for det in detections:
            x1, y1, x2, y2 = det[:4]
            # Пропускаем детекции в мёртвых зонах
            if not self._is_valid_detection(x1, y1, x2, y2):
                continue
            center_x = (x1 + x2) / 2
            target_y = y1 + (y2 - y1) * 0.20
            distance = ((center_x - center_x_frame) ** 2 + (target_y  - center_y_frame) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_person = (center_x, target_y )

        if not closest_person:
            return

        dx = closest_person[0] - center_x_frame
        dy = closest_person[1] - center_y_frame

        self.debug_print(f"aim_at_person: raw dx={dx:.1f} dy={dy:.1f} dist={min_distance:.1f}")

        # Чувствительность — подбирай: если мышь перелетает цель уменьшай
        sens = 0.8
        mouse_dx = int(dx * sens)
        mouse_dy = int(dy * sens)

        max_snap = 300
        mouse_dx = max(-max_snap, min(max_snap, mouse_dx))
        mouse_dy = max(-max_snap, min(max_snap, mouse_dy))

        # Порог выстрела — 3% от ширины кадра
        shoot_threshold = max(20, int(self.width * 0.03))

        if abs(mouse_dx) < shoot_threshold and abs(mouse_dy) < shoot_threshold:
            global_log_queue.put(f"🎯 На цели! dist={min_distance:.1f} px — выстрел!")
            send_input([make_mouse_click('left', True)])
            time.sleep(0.1)
            send_input([make_mouse_click('left', False)])
            return

        global_log_queue.put(f"🎯 Аим: dx={mouse_dx} dy={mouse_dy} dist={min_distance:.1f} threshold={shoot_threshold}")

        steps = 4
        rx, ry = 0.0, 0.0
        for _ in range(steps):
            ex = (mouse_dx / steps) + rx
            ey = (mouse_dy / steps) + ry
            sx = int(ex)
            sy = int(ey)
            rx = ex - sx
            ry = ey - sy
            if sx != 0 or sy != 0:
                send_input([make_mouse_move(sx, sy)])
            time.sleep(0.012)
    
    def _icm_worker(self):
        prev_feature = None
        while not self.stop_event.is_set():
            try:
                try:
                    item = self._icm_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                state = item['state']
                action = item['action']

                # БЕЗ no_grad — нужны градиенты для backward
                img_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                current_feature = self.feature_net(img_tensor)
                action_onehot = torch.nn.functional.one_hot(
                    torch.tensor(action, device=self.device),
                    num_classes=self.action_space.n
                ).float()
                input_tensor = torch.cat((current_feature.squeeze(0), action_onehot))
                pred_feature = self.forward_model(input_tensor)

                if prev_feature is not None:
                    # Intrinsic reward — без градиентов
                    with torch.no_grad():
                        intrinsic = ((pred_feature - prev_feature.squeeze(0)) ** 2).mean().item()
                        self._last_intrinsic = min(intrinsic, 1000.0)

                    # Backward pass — с градиентами
                    loss = ((pred_feature - prev_feature.squeeze(0).detach()) ** 2).mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Сохраняем без градиентов чтобы не держать граф в памяти
                prev_feature = current_feature.detach()

            except Exception as e:
                self.debug_print(f"ICM worker error: {e}")
            
    def _is_valid_detection(self, x1, y1, x2, y2):
        """Фильтрует детекции в мёртвых зонах кадра"""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Мёртвые зоны (в долях от размера кадра)
        left_dead   = self.width  * DEAD_ZONE_LEFT
        top_dead    = self.height * DEAD_ZONE_TOP
        bottom_dead = self.height * DEAD_ZONE_BOTTOM  # оружие снизу

        if cx < left_dead:
            return False  # в зоне оверлея
        if cy < top_dead:
            return False  # в зоне иконок
        if cy > bottom_dead:
            return False  # в зоне оружия

        return True
    
    #метод проверки виртуальных камер
    def check_available_cameras(self):
        self.debug_print("Запуск сканирования видеоустройств...")
        found_any = False
        max_to_check = 8  # Проверяем индексы от 0 до 7
        # Вместо бесконечного цикла проверяем фиксированный диапазон
        for index in range(max_to_check):
            try:
                # Инициализируем захват с DirectShow
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                # Задаем жесткий таймаут на открытие на случай фриза драйвера
                cap.set(cv2.CAP_PROP_TIMEOUT, 1000) 
                if cap.isOpened():
                    # Получаем разрешение для информативности логов
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.debug_print(f"[Успех] Найдена камера под индексом {index} ({w}x{h})")
                    cap.release()
                    found_any = True
                else:
                    # Не делаем break, просто пишем в дебаг и идем дальше
                    self.debug_print(f"Индекс {index}: устройств не обнаружено или занято")
            except Exception as e:
                self.debug_print(f"Ошибка при опросе индекса {index}: {e}")
        if not found_any:
            self.debug_print("[Внимание] Не найдено ни одного активного видеоустройства.")
        else:
            self.debug_print("Сканирование завершено. Если нужная камера (например, OBS Virtual) не выбрана, укажите её индекс вручную.")

    #Этот метод отвечает за распознавание текста на экране
    def get_ocr_text(self, frame):
        # Если кадр не пришел, сразу выходим, чтобы не тратить время
        if frame is None or frame.size == 0:
            return ""
        try:
            # Шаг 1: Безопасное кадрирование (ROI) с защитой от вылета за границы экрана
            if hasattr(self, 'roi_box') and self.roi_box is not None:
                x, y, w, h = self.roi_box
                h_max, w_max = frame.shape[:2]
                # Срезаем область по координатам x, y, ширина, высота
                img_cropped = frame[max(0, y):min(h_max, y+h), max(0, x):min(w_max, x+w)]
            else:
                # Если roi_box нигде не объявлен, работаем со всем кадром
                img_cropped = frame
            # Шаг 2: Предобработка изображения для Tesseract (работаем СТРОГО с img_cropped)
            # Перевод в оттенки серого
            gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            # Жесткая бинаризация (белый текст на черном фоне) — убирает градиенты и шумы бэкграунда игры
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Шаг 3: Конфигурация и запуск Tesseract
            # --psm 7 говорит Тессеракту, что перед ним ровно одна строка текста (идеально для логов/чата)
            custom_config = r'--psm 7 --oem 3'
            text = pytesseract.image_to_string(thresh, lang='eng+rus', config=custom_config)
            return text.strip()
        except Exception as e:
            # Если упадет cv2 (например, формат кадра битый) или Tesseract не установлен в системе
            self.debug_print(f"OCR error: {e}")
            return ""

    #Этот метод защищает бота от «безумного» поведения. Если игрок открыл меню закупки, нажал ESC (паузу) или зашел в настройки, нейросеть не должна пытаться фликать по иконкам интерфейса.
    def detect_ui(self, frame, ocr_text, yolo_results):
        # Шаг 1: Быстрая проверка через ключевые слова OCR
        text_low = (ocr_text or "").lower()
        for w in UI_KEYWORDS + ["пауза", "меню игры"]:
            if w in text_low:
                self.debug_print(f"UI обнаружен через OCR: {w}")
                return True
        # Шаг 2: Проверка через детекции YOLO
        if yolo_results and len(yolo_results[0].boxes.data) > 0:
            names = set()
            try:
                has_names_dict = hasattr(yolo_results[0], 'names')
                for det in yolo_results[0].boxes.data:
                    cls = int(det[5])
                    if has_names_dict:
                        cls_name = yolo_results[0].names.get(cls, "").strip()
                    else:
                        cls_name = ""
                    # Добавляем в сет только валидные имена классов
                    if cls_name:
                        names.add(cls_name.lower())
                ui_elements = {'inventory', 'menu', 'ui', 'search', 'pause'}
                # Ищем пересечение множеств
                detected_ui = ui_elements.intersection(names)
                if detected_ui:
                    self.debug_print(f"UI обнаружен через YOLO: {detected_ui}")
                    return True
            except Exception as e:
                self.debug_print(f"Ошибка при анализе YOLO в detect_ui: {e}")
        return False
    
    #Сбрасывает среду Gymnasium перед началом нового игрового эпизода.
    def reset(self, **kwargs):
        # Шаг 1: Безопасно берем стартовый кадр из нашей асинхронной системы
        frame = None
        with self.lock:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                frame = self.current_frame.copy()
        # Если фоновый воркер еще не успел поймать кадр, делаем одну попытку прочитать напрямую
        if frame is None and hasattr(self, 'cap') and self.cap.isOpened():
            ret, cap_frame = self.cap.read()
            if ret:
                frame = cap_frame
        if frame is not None:
            with self.lock:
                self.current_frame = frame.copy()
                self.annotated_frame = frame.copy()
                self.prev_frame = frame.copy()
                # Полностью сбрасываем контекст прошлых детекций под локом
                self.prev_ocr_text = ""
                self.prev_objects = set()
            # Генерация начального вектора состояния/наблюдения (Observation)
            self.current_state = self.get_state(frame)
        else:
            # Дефолтный пустой спейс (Box), если игра еще не запустилась или кадр пустой
            # Фиксируем размерность (3, 84, 84) в соответствии с твоей моделью (например, CNN)
            self.current_state = np.zeros((3, 84, 84), dtype=np.uint8)
            with self.lock:
                self.prev_frame = None
                self.annotated_frame = None
                self.prev_ocr_text = ""
                self.prev_objects = set()
        # Шаг 2: Сброс внутренних счетчиков шагов и таймингов
        self.frame_count = 0
        self.total_steps_accumulated += self.frame_processed
        self.frame_processed = 0
        # Шаг 3: Экстренно останавливаем персонажа, чтобы он не продолжал бежать из прошлого эпизода
        self.release_all_keys()
        # Шаг 4: Аккумулируем награды для графиков обучения (TensorBoard / W&B)
        self.total_reward_accumulated += self.total_reward
        self.total_reward = 0.0
        # Возвращаем по стандарту Gymnasium (obs, info)
        return self.current_state, {}

    #Этот метод get_state готовит наблюдение (Observation) для твоей RL-модели, сжимая игровой кадр до классического размера $84 \times 84$ пикселя
    def get_state(self, frame):
        try:
            # Шаг 1: Быстрый ресайз на CPU средствами OpenCV (интерполяция AREA идеальна для уменьшения)
            resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            # Шаг 2: Меняем оси с (H, W, C) на (C, H, W), как требуют CNN в PyTorch/Stable-Baselines
            state = np.transpose(resized, (2, 0, 1))
            # Возвращаем чистый uint8. Массив занимает всего ~21 КБ в памяти.
            return state.astype(np.uint8)
        except Exception as e:
            self.debug_print(f"Ошибка в get_state: {e}")
            # Возвращаем заполненный нулями массив в случае сбоя кадра
            return np.zeros((3, 84, 84), dtype=np.uint8)

    #Этот метод step получился по-настоящему монструозным. В нём переплелись воедино классическая среда RL (Gym), асинхронный ИИ-ассистент (YOLO-аимбот), модуль внутренней мотивации (ICM) и даже веб-парсер.
    def step(self, action):
        start_time = time.time()
        reward = 0.0
        description_reward = 0.0
        info = {'ui_mode': self.ui_mode}

        # --- ШАГ 1: ЗАХВАТ КАДРА ---
        frame = None
        if self.lock.acquire(blocking=True, timeout=0.1):
            try:
                if hasattr(self, 'current_frame') and self.current_frame is not None:
                    frame = self.current_frame.copy()
            finally:
                self.lock.release()
        if frame is None or frame.size == 0:
            frame = getattr(self, 'prev_frame', None)
            if frame is None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # --- ШАГ 2: ПАУЗА ---
        if self.pause_event.is_set():
            self.release_all_keys()
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)
            # После снятия паузы берём свежий кадр и продолжаем нормально
            if self.lock.acquire(blocking=True, timeout=0.1):
                try:
                    if hasattr(self, 'current_frame') and self.current_frame is not None:
                        frame = self.current_frame.copy()
                finally:
                    self.lock.release()
            new_state = self.get_state(frame)
            ocr_text = self.get_ocr_text(frame)
            self.ui_mode = self.detect_ui(frame, ocr_text, getattr(self, 'yolo_results', None))
            self.restrict_cursor_to_window()
            global_log_queue.put("Бот приостановлен")
            return new_state, 0.0, False, True, {'ui_mode': self.ui_mode}

        # --- ШАГ 3: ICM FORWARD PASS ---
        intrinsic = getattr(self, '_last_intrinsic', 0.0)
        try:
            if self.current_state.shape != (3, 84, 84):
                self.current_state = np.transpose(self.current_state, (2, 0, 1))
            self._icm_queue.put_nowait({'state': self.current_state.copy(), 'action': action})
        except Exception:
            pass

        # --- ШАГ 4: YOLO ДО ДЕЙСТВИЯ — чтобы аим работал по свежим данным ---
        with self.lock:
            yolo_results = self.yolo_results if hasattr(self, 'yolo_results') else None
        if yolo_results:
            global_log_queue.put(f"📷 YOLO кадр актуален")

        # --- ШАГ 5: АИМ ДО ДЕЙСТВИЯ если person в кадре ---
        action_name = self.actions[action][0]
        person_detected = False
        person_in_center = False
        dist_norm = 1.0

        if self.is_shooter and yolo_results and len(yolo_results[0].boxes.data) > 0:
            try:
                detections = yolo_results[0].boxes.data.cpu().numpy()
                frame_width = self.width
                frame_height = self.height
                center_x_frame = frame_width / 2
                center_y_frame = frame_height / 2
                best_dist = float('inf')
                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    # Фильтр мёртвых зон
                    if not self._is_valid_detection(x1, y1, x2, y2):
                        continue
                    person_detected = True
                    cx = (x1 + x2) / 2
                    ty = y1 + (y2 - y1) * 0.20
                    dist_x = abs(cx - center_x_frame) / frame_width
                    dist_y = abs(ty - center_y_frame) / frame_height
                    d = (dist_x ** 2 + dist_y ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        dist_norm = d
                if dist_norm < 0.03:
                    person_in_center = True
            except Exception as e:
                self.debug_print(f"Ошибка предварительного аима: {e}")

        # --- ШАГ 6: ВЫПОЛНЕНИЕ ДЕЙСТВИЯ ---
        self.perform_action(action)

        # --- ШАГ 7: OCR ---
        ocr_text = self.prev_ocr_text if (self.frame_processed % 30 != 0) else self.get_ocr_text(frame)
        change_reward = self.calculate_change_reward(self.prev_frame, frame) if (self.prev_frame is not None) else 0.0
        ocr_reward = self.calculate_ocr_reward(self.prev_ocr_text, ocr_text)

        # --- ШАГ 8: НАГРАДЫ ---
        current_objects = set()
        exploration_reward = 0.0
        web_action_reward = 0.0
        enemy_reward = 0.0

        relevant_classes = {'person', 'item', 'object'}
        if self.is_shooter:
            relevant_classes = {'person', 'weapon', 'bomb'}
        elif "minecraft" in self.game_name.lower():
            relevant_classes = {'person', 'animal', 'block', 'item'}

        if yolo_results and len(yolo_results[0].boxes.data) > 0:
            try:
                detections = yolo_results[0].boxes.data.cpu().numpy()
                names_dict = yolo_results[0].names
                detected_classes = []

                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    # Фильтр мёртвых зон — оверлей слева, иконки сверху, оружие снизу
                    if not self._is_valid_detection(x1, y1, x2, y2):
                        continue
                    cls = int(det[5]) if len(det) > 5 else 0
                    cls_name = names_dict.get(cls, "person").lower()
                    detected_classes.append(cls_name)
                    if cls_name in relevant_classes:
                        current_objects.add(cls_name)

                self.debug_print(f"Обнаруженные классы: {detected_classes}")
                
                if self.is_shooter:
                    if person_detected:
                        # Награда за аим — только если активно целимся или аим-тред довёл прицел
                        # dist_norm от 0 (в центре) до ~1.4 (угол кадра)
                        # Порог увеличен до 1.0 чтобы награда была даже при дальней цели
                        aim_reward = max(0.0, 2.0 * (1.0 - dist_norm / 1.0))
                        enemy_reward += aim_reward

                        # Бонус если PPO сам выбрал аим-действие
                        if action_name in ["aim_at_person", "w+aim", "aim+click_left"]:
                            enemy_reward += 0.5

                        # Постоянная награда за то что видим врага — мотивирует искать
                        enemy_reward += 0.2

                        global_log_queue.put(
                            f"🏆 aim_reward={aim_reward:.2f} dist={dist_norm:.2f} "
                            f"{'🎯CENTER' if person_in_center else ''}"
                        )
                    else:
                        # Врага нет — мотивируем двигаться и исследовать
                        move_actions = ["w", "a", "s", "d", "shift+w", "ctrl+w",
                                        "w+camera_left", "w+camera_right"]
                        if action_name in move_actions:
                            enemy_reward += 0.15  # небольшая награда за движение
                        # Штраф за аим в пустоту
                        if action_name in ["aim_at_person", "w+aim", "aim+click_left"]:
                            enemy_reward -= 0.5

                    # Штраф за стену — двигался но картинка не изменилась
                    if action_name in ["w", "a", "s", "d", "shift+w", "ctrl+w"] and change_reward < 0.01:
                        enemy_reward -= 0.4
                        global_log_queue.put("🧱 Стена -0.4")

                    # Штраф за экстремальный задир камеры — только вертикаль без аима
                    if action_name in ["camera_up", "camera_down"] and not person_detected:
                        enemy_reward -= 0.2

                    # Штраф за спам одним действием
                    self._last_actions.append(action_name)
                    if len(self._last_actions) > 6:
                        self._last_actions.pop(0)
                    if len(self._last_actions) == 6 and len(set(self._last_actions)) == 1:
                        enemy_reward -= 0.5
                        global_log_queue.put(f"🔁 Спам {action_name} -0.5")
                    idle_actions = {"a", "s", "d", "ctrl", "shift", "space", "wait",
                                "camera_left", "camera_right", "camera_up", "camera_down"}
                    if not hasattr(self, '_idle_action_count'):
                        self._idle_action_count = 0
                    if action_name in idle_actions:
                        self._idle_action_count += 1
                    else:
                        self._idle_action_count = 0  # сброс если сделал что-то полезное
                    if self._idle_action_count >= 10:
                        enemy_reward -= 10.0
                        self._idle_action_count = 0  # сброс после штрафа
                        global_log_queue.put(f"⛔ Штраф: 10 шагов без движения вперёд! -10")

                    # Стрельба
                    shoot_actions = ["click_left", "hold_left", "w+click_left", "aim+click_left"]
                    if action_name in shoot_actions:
                        if person_in_center:
                            enemy_reward += 15.0
                            global_log_queue.put("💥 Точный выстрел! +15")
                        elif person_detected:
                            # Частичная награда если близко к цели
                            partial = max(0.0, 3.0 * (1.0 - dist_norm / 0.3))
                            if partial > 0:
                                enemy_reward += partial
                                global_log_queue.put(f"💫 Близкий выстрел +{partial:.1f}")
                            else:
                                enemy_reward -= 1.0
                                global_log_queue.put("💨 Выстрел мимо -1.0")
                        else:
                            enemy_reward -= 2.0
                            global_log_queue.put("❌ Выстрел в пустоту -2.0")
                # Эксплорация
                new_objects = current_objects - self.prev_objects
                exploration_reward = min(len(new_objects) * 0.5, 2.0)
                if new_objects:
                    web_action_reward += 0.2 * len(new_objects)
                self.prev_objects = current_objects

            except Exception as e:
                self.debug_print(f"Ошибка логики наград YOLO: {e}")

        # --- ШАГ 9: ICM BACKWARD PASS ---
        new_state = self.get_state(frame)

        # --- ШАГ 10: UI + ФИНАЛИЗАЦИЯ ---
        self.ui_mode = self.detect_ui(frame, ocr_text, yolo_results)
        self.restrict_cursor_to_window()
        if self.ui_mode:
            self.release_all_keys()
        if self.description_keywords:
            if any(kw in self.description_keywords for kw in ["shoot", "kill", "eliminate"]) and action_name == "click_left":
                description_reward += 1.0

        reward = (intrinsic * 0.001) + exploration_reward + change_reward + ocr_reward + web_action_reward + enemy_reward + description_reward
        self.current_reward = reward
        self.total_reward += reward

        self.prev_frame = frame.copy()
        self.prev_ocr_text = ocr_text
        self.current_state = new_state
        self.frame_processed += 1

        self.debug_print(f"Step FPS: {1/(time.time()-start_time+1e-9):.1f}, Reward: {reward:.2f}")
        if self.frame_processed % 20 == 0:
            print(f"[DEBUG] step: total_reward={self.total_reward}, frame_processed={self.frame_processed}")

        return new_state, reward, False, False, {}

    #мотивировать агента не стоять на месте и наказывать за статичную картинку
    def calculate_change_reward(self, prev_frame, new_frame):
        if prev_frame is None or new_frame is None:
            return 0.0
        try:
            # ОПТИМИЗАЦИЯ: Вместо тяжелых оригинальных кадров 
            # сжимаем их до микро-размера (например, 100x100)
            small_prev = cv2.resize(prev_frame, (100, 100), interpolation=cv2.INTER_NEAREST)
            small_new = cv2.resize(new_frame, (100, 100), interpolation=cv2.INTER_NEAREST)
            # Считаем разницу на маленькой матрице (мгновенно для CPU)
            diff = cv2.absdiff(small_prev, small_new)
            m = np.mean(diff)
            # Изменение порога: для мелкого кадра порог 10-15 обычно эквивалентен 20 на большом
            return 0.05 if m > 12.0 else -0.005
        except Exception as e:
            self.debug_print(f"Ошибка в calculate_change_reward: {e}")
            return 0.0

    #Метод calculate_ocr_reward — отличный способ привязать награду к смысловым текстовым триггерам
    def calculate_ocr_reward(self, prev_text, new_text):
        if not new_text or prev_text == new_text:
            return 0.0
        new_text_lower = new_text.lower()
        prev_text_lower = prev_text.lower() if prev_text else ""
        # Список наших триггеров
        target_keywords = ["log", "stone", "inventory", "поиск"]
        # Ищем ключевые слова, которые ПОЯВИЛИСЬ только сейчас (их не было в prev_text)
        for kw in target_keywords:
            if kw in new_text_lower and kw not in prev_text_lower:
                self.debug_print(f"[OCR Reward] Обнаружено новое целевое слово: '{kw}'! Награда: 0.6")
                return 0.6
        return 0.05

    #проверка флага ui_mode защищает игру от случайных ложных нажатий, когда агент находится в меню или инвентаре
    def perform_action(self, action):
        try:
            action_data = None
            if isinstance(self.action_map, (list, tuple)):
                if 0 <= action < len(self.action_map):
                    action_data = self.action_map[action]
                else:
                    self.debug_print(f"[ERROR] action index {action} out of range")
                    return
            elif isinstance(self.action_map, dict):
                action_data = self.action_map.get(action)

            if action_data is None or not isinstance(action_data, dict):
                return

            keys     = action_data.get('keys', [])
            dx, dy   = action_data.get('mouse_move', (0, 0))
            click    = action_data.get('mouse_click', None)
            duration = action_data.get('duration', 0.12)

            if self.ui_mode:
                if 'e' not in keys and 'escape' not in keys:
                    return

            action_name = self.actions[action][0]

            # --- Клавиши: нажать, подождать duration, отпустить ---
            vk_list = [vk for k in keys for vk in [key_to_vk(k)] if vk]

            if vk_list:
                send_input([make_key_input(vk, down=True) for vk in vk_list])
                global_log_queue.put(f"▶ {action_name} | {'+'.join(keys)}")
                time.sleep(duration)
                send_input([make_key_input(vk, down=False) for vk in vk_list])
                global_log_queue.put(f"■ {action_name} | released")
                time.sleep(0.03)

            # --- Мышь (только если action явно двигает камеру, не аим) ---
            elif dx != 0 or dy != 0:
                self.action_queue.put({
                    'mouse_move': (dx, dy),
                    'mouse_click': click,
                    'duration': duration,
                })
                global_log_queue.put(f"🖱 {action_name} | dx={dx} dy={dy}")
                time.sleep(duration)

            # --- Клик без движения ---
            elif click and not click.startswith('hold_'):
                btn = 'left' if 'left' in click else 'right'
                send_input([make_mouse_click(btn, True)])
                time.sleep(0.1)
                send_input([make_mouse_click(btn, False)])
                global_log_queue.put(f"🖱 {action_name} | click {btn}")

            else:
                # wait action
                time.sleep(duration)

        except Exception as e:
            self.debug_print(f"Критическая ошибка в perform_action: {e}")

    #парсить Wikipedia API на лету, чтобы бот сам понимал свойства предметов
    def search_object_info(self, obj_name, lang="en"):
        obj_name = obj_name.lower().strip()
        # --- ШАГ 1: ЛОКАЛЬНАЯ БАЗА ЗНАНИЙ (Экономит 100% сетевого времени) ---
        local_knowledge = {
            "person": ["click_left", "aim_at_person"] if self.is_shooter else ["e"],
            "weapon": ["e", "click_right"],
            "bomb": ["e"],
            "door": ["e"],
            "chest": ["e", "click_right"],
            "zombie": ["click_left"],
            "skeleton": ["click_left"],
            "creeper": ["click_left"],
            "item": ["e"]
        }
        if obj_name in local_knowledge:
            # Если объект специфичен для режима (например, person в шутере), отдаем приоритет
            return local_knowledge[obj_name]
        # --- ШАГ 2: ПРОВЕРКА ЛОКАЛЬНОГО КЭША ---
        if obj_name in self.known_objects:
            return self.known_objects[obj_name]
        # --- ШАГ 3: ЗАЩИТА ОТ СПАМА ЗАПРОСОВ (RATE LIMIT) ---
        current_time = time.time()
        if current_time - self.last_api_call < self.api_call_interval:
            # Чтобы агент не остался без действий, пока ждет кулдаун API, возвращаем базовый безопасный набор действий
            return ["e"] if "minecraft" in self.game_name.lower() else ["click_left"]
        # --- ШАГ 4: БЕЗОПАСНЫЙ СИНХРОННЫЙ ЗАПРОС (С МИНИМАЛЬНЫМ ТАЙМАУТОМ) ---
        self.debug_print(f"Отправка Wikipedia REST запроса для объекта: {obj_name}")
        self.last_api_call = current_time  # Сразу обновляем таймер, чтобы защитить параллельные шаги
        try:
            page = obj_name.replace(" ", "_")
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{page}"
            # Уменьшаем таймаут до 1.5 секунд. Если Википедия не ответила мгновенно — бот идет дальше
            response = requests.get(url, timeout=1.5)
            if response.status_code != 200:
                self.debug_print(f"Wikipedia вернула статус {response.status_code} для {obj_name}")
                self.known_objects[obj_name] = ["unknown"]
                return ["unknown"]
            data = response.json()
            content = data.get("extract", "").lower()
            actions = []
            # Парсинг контекста
            if any(w in content for w in ["press e", "interact", "use", "open", "door", "lever"]):
                actions.append("e")
            if any(w in content for w in ["click", "attack", "shoot", "kill", "weapon", "enemy", "monster"]):
                actions.append("click_left")
            if any(w in content for w in ["right click", "place", "build", "block"]):
                actions.append("click_right")
            # Дополнительная валидация на основе флагов среды
            if self.is_shooter and obj_name == "person":
                actions.extend(["click_left", "aim_at_person"])
            # Фильтруем дубликаты
            actions = list(set(actions))
            if not actions:
                actions = ["unknown"]
            # Сохраняем в кэш в оперативной памяти
            self.known_objects[obj_name] = actions
            # ВАЖНО: Убираем self.save_config() отсюда! Будем сохранять кэш один раз в конце эпизода (в методе reset или close)
            self.debug_print(f"Успешно сохранено в RAM кэш для '{obj_name}': {actions}")
            return actions
        except requests.exceptions.Timeout:
            self.debug_print(f"[TIMEOUT] Wikipedia API не ответила вовремя для {obj_name}")
            return ["unknown"]
        except Exception as e:
            self.debug_print(f"Ошибка выполнения search_object_info для {obj_name}: {e}")
            return ["unknown"]

    # Возвращает путь к директории профиля агента
    def get_profile_dir(self):
        # Убираем только строго запрещенные в путях Windows символы, сохраняя регистр и пробелы
        forbidden_chars = '<>:"/\\|?*'
        clean_game_name = "".join(c for c in str(self.game_name) if c not in forbidden_chars).strip()
        # Формируем строгий кроссплатформенный путь
        profile_path = Path("profiles") / clean_game_name / f"gen_{self.generation}_agent_{self.profile_id}"
        try:
            # parents=True — создает всю цепочку папок (profiles, затем игру, затем агента)
            # exist_ok=True — не выдает ошибку, если папка уже существует
            profile_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.debug_print(f"[ERROR] Не удалось создать директорию профиля {profile_path}: {e}")
        # Возвращаем стандартную строку для полной совместимости со всем твоим проектом
        return str(profile_path)

    # Загружает конфигурацию профиля агента
    def load_config(self):
        profile_dir = self.get_profile_dir()
        config_path = Path(profile_dir) / "config.json"
        # Структура дефолтного конфига
        default_config = {
            "actions": {str(i): act[0] for i, act in enumerate(self.actions)},
            "rewards": {},
            "known_objects": {},
            "total_reward": 0.0,
        }
        if not config_path.exists():
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
                return default_config
            except Exception as e:
                self.debug_print(f"[ERROR] Не удалось записать дефолтный конфиг: {e}")
                return default_config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config = {**default_config, **config}
            # Возвращаем старую добрую логику: загружаем как есть (из JSON приходят списки)
            self.known_objects = config.get("known_objects", {})
            self.total_reward = float(config.get("total_reward", 0.0))
            
            self.debug_print(f"Конфиг успешно загружен. Найдено объектов в базе данных: {len(self.known_objects)}")
            return config
        except (json.JSONDecodeError, ValueError) as e:
            self.debug_print(f"[WARNING] Конфиг-файл поврежден или пуст ({e}). Откатываемся на default_config.")
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
            self.known_objects = default_config["known_objects"]
            self.total_reward = default_config["total_reward"]
            return default_config
        except Exception as e:
            self.debug_print(f"Критическая ошибка при загрузке конфига: {e}")
            return default_config
        
    # Сохраняет текущую конфигурацию, прогресс и базу знаний агента на диск.
    def save_config(self, config=None):
        profile_dir = self.get_profile_dir()
        config_path = Path(profile_dir) / "config.json"
        # Если конфиг не передан извне, собираем актуальное состояние из рантайма
        if config is None:
            # Расчет честной награды: все прошлые эпизоды + то, что набито в текущем прямо сейчас
            actual_total_reward = self.total_reward_accumulated + getattr(self, 'total_reward', 0.0)
            # БЕЗОПАСНОСТЬ: Сохраняем действия ИЛИ как полную структуру (если они нужны для load_config),
            # ИЛИ страхуем, чтобы act[0] не ломал логику. Давай сохраним полную структуру self.actions,
            # чтобы ничего не потерять, но если там список/кортеж — пишем его целиком.
            serialized_actions = {}
            for i, act in enumerate(self.actions):
                # Если act — это уже список/кортеж (например, ['click_left', ...]) или словарь
                if isinstance(act, (list, tuple, dict)):
                    serialized_actions[str(i)] = act
                else:
                    # Если это просто одиночная строка
                    serialized_actions[str(i)] = [act]
            conf = {
                "actions": serialized_actions,
                "rewards": {},
                # Защищаем чтение базы знаний (только списки, никаких сетов, как мы и договорились!)
                "known_objects": getattr(self, 'known_objects', {}),
                "total_reward": float(actual_total_reward),
            }
        else:
            conf = config
        try:
            # Шаг 1: Запись во временный файл-слепок (атомарное сохранение)
            temp_config_path = config_path.with_suffix('.tmp')
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                # default=list страхует от случайных set-объектов, если они проскочат в known_objects
                json.dump(conf, f, indent=4, ensure_ascii=False, default=list)
            # Шаг 2: Мгновенная замена старого файла новым на уровне ОС
            temp_config_path.replace(config_path)
        except Exception as e:
            if hasattr(self, 'debug_print'):
                self.debug_print(f"[ERROR] Не удалось сохранить конфигурацию в {config_path}: {e}")
            else:
                print(f"[ERROR] Не удалось сохранить конфигурацию в {config_path}: {e}")

#красиво и безопасно останавливать процесс обучения нейросети
class StopCallback(BaseCallback):
    
    #Колбэк для безопасного прерывания обучения Stable-Baselines3 из параллельных потоков.
    def __init__(self, stop_event, debug_print=None):
        super(StopCallback, self).__init__()
        self.stop_event = stop_event
        self.debug_print = debug_print or print
        self.stop_triggered = False
        
    def _on_step(self) -> bool:
        if self.stop_event.is_set():
            if not self.stop_triggered:
                # Этот блок выполнится ОДИН раз в момент нажатия кнопки "Стоп"
                self.stop_triggered = True
                self.debug_print("[STOP] Получен сигнал экстренной остановки обучения.")
                # Попытка отправить сообщение в глобальный лог-сервис, если он доступен
                try:
                    global_log_queue.put("Обучение принудительно остановлено пользователем")
                except Exception:
                    pass
                # АВТО-СОХРАНЕНИЕ: Спасаем веса модели перед выходом, если у среды есть папка профиля
                try:
                    if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                        # Пытаемся выудить get_profile_dir из нашей кастомной Gym-среды
                        env = self.training_env.envs[0]
                        if hasattr(env, 'get_profile_dir'):
                            save_dir = env.get_profile_dir()
                            save_path = os.path.join(save_dir, "emergency_model_checkpoint.zip")
                            self.model.save(save_path)
                            self.debug_print(f"[SUCCESS] Веса модели успешно спасены в: {save_path}")
                except Exception as e:
                    self.debug_print(f"[WARNING] Не удалось автоматически сохранить модель при выходе: {e}")
            # Возвращаем False — команда для Stable-Baselines3 немедленно остановить .learn()
            return False
            
        return True

#периодический чекпоинтер для Stable-Baselines3, который будет сохранять веса нейросети во время долгого обучения.
class SaveCallback(BaseCallback):
    
    #Колбэк для периодического сохранения чекпоинтов модели и ведения истории обучения.
    def __init__(self, save_path: str, save_freq: int, env):
        super(SaveCallback, self).__init__()
        # save_path теперь рассматриваем как базовую директорию или шаблон
        self.base_save_path = save_path
        self.save_freq = save_freq
        self.env = env
        self.best_reward = -float('inf')
    
    #здесь совмещены и периодическое сохранение весов нейросети, и экстренный сброс модели при остановке, и логика прерывания обучения
    def _on_step(self) -> bool:
        # 1. Если среда на паузе, мы просто пропускаем логику сохранения,
        if self.env.pause_event.is_set():
            return True
        # Прерываем обучение, если получен сигнал остановки
        # (Оставляем как резервный предохранитель, если StopCallback не используется)
        if self.env.stop_event.is_set() or global_stop_event.is_set():
            self.env.debug_print("[SAVE] Обучение прервано. Выполняю экстренное сохранение...")
            self._save_model(suffix="emergency")
            return False
        try:
            # 2. Периодическое сохранение по частоте шагов
            if self.n_calls % self.save_freq == 0:
                # Сохраняем регулярный чекпоинт с номером шага, чтобы не затирать историю
                self._save_model(suffix=f"step_{self.n_calls}")
                self.env.save_training_state(save_path=os.path.dirname(self.base_save_path))
            # 3. Дополнительная умная логика: Сохранение ЛУЧШЕЙ модели (Best Model)
            current_reward = getattr(self.env, 'total_reward', 0.0)
            if current_reward > self.best_reward and self.n_calls > 100:
                self.best_reward = current_reward
                self._save_model(suffix="best")
                self.env.debug_print(f"[NEW BEST] Сохранена лучшая модель с наградой: {self.best_reward:.2f}")
        except Exception as e:
            self.env.debug_print(f"Ошибка в блоке логики SaveCallback на шаге {self.n_calls}: {e}")
        return True
    
    #Вспомогательный внутренний метод для безопасной генерации путей и записи модели.
    def _save_model(self, suffix: str):
        try:
            # Отрезаем расширение .zip, если оно было передано в путях, и собираем красивое имя
            base_dir, file_name = os.path.split(self.base_save_path)
            name_without_ext = os.path.splitext(file_name)[0]
            # Итоговое имя файла: например, "ppo_agent_step_5000.zip" или "ppo_agent_best.zip"
            actual_save_path = os.path.join(base_dir, f"{name_without_ext}_{suffix}.zip")
            # Создаем директорию, если её стерли
            os.makedirs(base_dir, exist_ok=True)
            # Запись весов алгоритма Stable-Baselines3
            self.model.save(actual_save_path)
            self.env.debug_print(
                f"[SAVE] Модель успешно сохранена: {actual_save_path} | "
                f"Reward: {self.env.total_reward:.2f} | Frames: {self.env.frame_processed}"
            )
        except Exception as e:
            self.env.debug_print(f"Критическая ошибка при физической записи модели ({suffix}): {e}")

def gui_thread(stop_event: threading.Event):
    print("gui_thread: Запуск потока GUI...")
    root = None
    try:
        root = tk.Tk()
        GameEnv.active_gui = root
        root.title("Bot Overlay")
        root.overrideredirect(True)
        root.attributes('-topmost', True)
        root.attributes('-alpha', 0.8)
        root.attributes('-disabled', True)
        print("gui_thread: Tkinter окно создано")

        # Начальные размеры (будут обновляться динамически)
        env_width, env_height = 640, 480
        window_width = min(480, env_width // 2)
        window_height = int(window_width * (env_height / env_width)) + 120
        print(f"gui_thread: Начальные размеры окна: {window_width}x{window_height}")

        def update_position():
            if stop_event.is_set() or global_stop_event.is_set():
                try:
                    root.quit()
                    root.destroy()
                    print("gui_thread: GUI закрыт в update_position")
                except tk.TclError:
                    pass
                finally:
                    GameEnv.active_gui = None
                return
            current_env = GameEnv.current_env
            rect = current_env.get_game_window_rect() if current_env else None
            if rect:
                game_left, game_top, game_right, game_bottom = rect
                game_h = game_bottom - game_top
                overlay_x = game_left + 10
                overlay_y = game_top + (game_h - window_height) // 2
            else:
                monitor = get_monitors()[0]
                overlay_x, overlay_y = 10, (monitor.height - window_height) // 2
            try:
                root.geometry(f"{window_width}x{window_height}+{overlay_x}+{overlay_y}")
            except tk.TclError:
                pass
            root.after(100, update_position)

        video_label = tk.Label(root, bg='black')
        video_label.pack(fill="both", expand=True)
        status_label = tk.Label(
            root,
            text="Ожидание окружения...",
            bg='black', fg='white', font=("Arial", 8)
        )
        status_label.pack(fill="x")
        demo_status_label = tk.Label(root, text="", bg='black', fg='red', font=("Arial", 8))
        demo_status_label.pack(fill="x")
        log_text = tk.Text(root, height=6, bg='black', fg='white', font=("Arial", 8))
        log_text.pack(fill="x")
        print("gui_thread: Виджеты GUI созданы")

        update_position()
        print("gui_thread: update_position запущен")

        fps_times = []
        last_update_time = 0
        update_interval = 1 / 60  # ← ФИКС: было 1/30, теперь 60 FPS

        def update():
            nonlocal last_update_time, env_width, env_height, window_width, window_height
            if stop_event.is_set() or global_stop_event.is_set():
                try:
                    root.quit()
                    root.destroy()
                    print("gui_thread: GUI закрыт в update")
                except tk.TclError:
                    pass
                finally:
                    GameEnv.active_gui = None
                return

            now = time.time()
            if now - last_update_time < update_interval:
                root.after(16, update)  # ← ФИКС: было 33мс (~30fps), теперь 16мс (~60fps)
                return
            last_update_time = now

            try:
                current_env = GameEnv.current_env
                if current_env is None:
                    status_label.config(text="Ожидание окружения...")
                    root.after(16, update)  # ← ФИКС
                    return

                # Динамически обновляем размеры окна, если изменились размеры env
                if current_env.width != env_width or current_env.height != env_height:
                    env_width, env_height = current_env.width, current_env.height
                    window_width = min(480, env_width // 2)
                    window_height = int(window_width * (env_height / env_width)) + 120
                    update_position()
                    print(f"gui_thread: Размеры обновлены: {window_width}x{window_height}")

                # ФИКС: берём annotated если есть, иначе сразу raw current_frame — без задержки YOLO
                with current_env.lock:
                    annotated = current_env.annotated_frame
                    raw = current_env.current_frame
                frame = annotated if annotated is not None else raw
                if frame is None:
                    frame = np.zeros((env_height, env_width, 3), dtype=np.uint8)

                display_w = window_width
                display_h = int(display_w * (env_height / env_width))
                frame_resized = cv2.resize(frame, (display_w, display_h))
                h, w = frame_resized.shape[:2]
                overlay = frame_resized.copy()
                left_px = int(w * DEAD_ZONE_LEFT)
                cv2.rectangle(overlay, (0, 0), (left_px, h), (0, 0, 0), -1)
                cv2.putText(overlay, "IGNORE", (4, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

                # Верхняя зона — иконки игроков
                top_px = int(h * DEAD_ZONE_TOP)
                cv2.rectangle(overlay, (left_px, 0), (w, top_px), (0, 0, 0), -1)
                cv2.putText(overlay, "IGNORE", (left_px + 4, top_px // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

                # Нижняя зона — оружие
                bottom_px = int(h * DEAD_ZONE_BOTTOM)
                cv2.rectangle(overlay, (left_px, bottom_px), (w, h), (0, 0, 0), -1)
                cv2.putText(overlay, "IGNORE", (left_px + 4, bottom_px + (h - bottom_px) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

                # Полупрозрачное наложение — оригинал виден под зонами
                frame_resized = cv2.addWeighted(overlay, 0.6, frame_resized, 0.4, 0)

                # Рамка активной зоны (что бот реально видит) — зелёная
                cv2.rectangle(frame_resized,
                              (left_px, top_px),
                              (w, bottom_px),
                              (0, 255, 0), 1)
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)

                if video_label.winfo_exists():
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)

                # FPS скользящее окно по последним 60 кадрам
                fps_times.append(now)
                if len(fps_times) > 60:
                    fps_times.pop(0)
                if len(fps_times) >= 2:
                    fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-9)
                else:
                    fps = 0.0

                # Обновление статуса
                steps = current_env.frame_processed
                reward = current_env.total_reward
                status_text = (
                    f"Gen: {current_env.generation}/{MAX_GENS} | Agent: {current_env.agent_id}/{MAX_PROFILES} | FPS: {fps:.1f}\n"
                    f"Steps: {steps}/{TIMESTEPS_PER_AGENT} | Total Reward: {reward:.2f}"
                )
                if current_env.pause_event.is_set():
                    status_text += "\nPaused"
                status_label.config(text=status_text)

                # Demo статус
                demo_status_label.config(
                    text="Ведётся видеофиксация" if current_env.recording_active else
                         ("Фрагмент игры: отсутствует" if not current_env.has_demo else f"Фрагмент игры найден, нажмите {BIND_PAUSE}"),
                    fg='yellow' if current_env.recording_active else ('red' if not current_env.has_demo else 'green')
                )

                # Логи
                max_logs = 5
                for _ in range(max_logs):
                    if global_log_queue.empty():
                        break
                    log_text.insert(tk.END, global_log_queue.get_nowait() + "\n")
                    log_text.see(tk.END)

            except Exception as e:
                print(f"gui_thread: Ошибка обновления GUI: {e}")

            root.after(16, update)  # ← ФИКС: было 33

        update()
        print("gui_thread: Цикл обновления запущен")
        root.mainloop()
        print("gui_thread: mainloop завершён")
    except Exception as e:
        print(f"gui_thread: Ошибка потока GUI: {e}")
    finally:
        if root is not None:
            try:
                root.quit()
                root.destroy()
                print("gui_thread: GUI закрыт в finally")
            except tk.TclError as e:
                print(f"gui_thread: Ошибка при закрытии GUI в finally: {e}")
        GameEnv.active_gui = None
    
def get_game_name():
    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd)
    print(f"Detected game window:", {title})
    return title if title else "UnknownGame"

def save_global_state(game_name, current_gen, current_agent, profile_paths, generations, num_agents):
    state_path = f"profiles/{game_name}/training_state.json"
    os.makedirs(f"profiles/{game_name}", exist_ok=True)
    state = {
        "current_gen": current_gen,
        "current_agent": current_agent,
        "profile_paths": profile_paths,
        "generations": generations,
        "num_agents": num_agents,
        
    }
    try:
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4)
            print(f"Состояние сохранено в {state_path}")
    except Exception as e:
        print(f"Глобальное состояние сохранено: gen={current_gen}, agent={current_agent}")

def load_global_state(game_name):
    state_path = f"profiles/{game_name}/training_state.json"
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                print(f"Состояние загружено из {state_path}")
                return state
        except Exception as e:
            print(f"Ошибка при загрузке состояния из {state_path}: {e}")
            return None
    else:
        print(f"Файл состояния {state_path} не найден, будет создан новый")
        return None

def main():
    global global_stop_event, global_log_queue
    global_stop_event = threading.Event()
    global_log_queue = queue.Queue(maxsize=50)

    game_name = get_game_name()
    # Фикс 2: чистим имя игры от запрещённых символов для безопасного использования в путях
    safe_game_name = re.sub(r'[\\/*?:"<>|]', "_", game_name)
    default_generations = MAX_GENS
    default_num_agents = MAX_PROFILES
    timesteps_per_agent = TIMESTEPS_PER_AGENT
    elite_fraction = 0.2
    random_fraction = 0.1
    profile_paths = []
    # Фикс 1: performances объявлен заранее, чтобы не было NameError при проверке чемпиона
    performances = []
    env = None
    final_env = None

    os.makedirs(f"profiles/{safe_game_name}", exist_ok=True)
    print(f"Создана директория profiles/{safe_game_name}")

    state = load_global_state(safe_game_name)
    if state:
        current_gen = state["current_gen"]
        current_agent = state["current_agent"]
        profile_paths = state["profile_paths"]
        generations = state.get("generations", default_generations)
        num_agents = state.get("num_agents", default_num_agents)
    else:
        current_gen = 0
        current_agent = 0
        profile_paths = []
        generations = default_generations
        num_agents = default_num_agents
        save_global_state(safe_game_name, current_gen, current_agent, profile_paths, generations, num_agents)

    os.makedirs(f"profiles/{safe_game_name}/shared_demo", exist_ok=True)

    if ENABLE_GUI:
        gui_stop_event = threading.Event()
        gui_thread_instance = threading.Thread(target=gui_thread, args=(gui_stop_event,), daemon=False)
        gui_thread_instance.start()
        print("GUI запущен один раз для всех агентов")

    try:
        while current_gen < generations:
            if global_stop_event.is_set():
                print("Глобальный стоп-событие установлено, выход из основного цикла")
                break
            print(f"Поколение {current_gen + 1}/{generations}")
            current_profile_paths = []

            for agent_id in range(current_agent, num_agents):
                # Фикс 2: используем safe_game_name во всех путях
                profile_dir = f"profiles/{safe_game_name}/gen_{current_gen}_agent_{agent_id}"
                os.makedirs(profile_dir, exist_ok=True)

                # Фикс 6: ищем реальный файл модели по паттерну, а не фиксированное "model.zip"
                existing_models = sorted(
                    [f for f in os.listdir(profile_dir) if f.endswith(".zip")],
                    key=lambda f: os.path.getmtime(os.path.join(profile_dir, f)),
                    reverse=True
                )
                model_path = os.path.join(profile_dir, existing_models[0]) if existing_models else os.path.join(profile_dir, "model.zip")

                env = GameEnv(game_name, profile_id=agent_id, is_final_profile=False, generation=current_gen)
                env.debug_print(f"Создано окружение для gen={current_gen}, agent_id={agent_id}")
                print(f"[DEBUG] New agent started: gen={current_gen}, agent_id={agent_id}, total_reward={env.total_reward}, total_reward_accumulated={env.total_reward_accumulated}")
                current_profile_paths.append(profile_dir)

                if existing_models:
                    try:
                        model = PPO.load(model_path, env=env)
                        config = env.load_config()
                        env.frame_processed = config.get("frame_processed", 0)
                        env.total_reward = config.get("total_reward", 0.0)
                    except Exception as e:
                        env.debug_print(f"Не удалось загрузить модель {model_path}: {e}")
                        model = PPO("CnnPolicy", env, n_steps=TIMESTEPS_PER_AGENT, batch_size=64, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')
                        env.frame_processed = 0
                        env.total_reward = 0.0
                else:
                    model = PPO("CnnPolicy", env, n_steps=TIMESTEPS_PER_AGENT, batch_size=64, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')
                    env.frame_processed = 0
                    env.total_reward = 0.0

                # Фикс 3: передаём все profile_paths в evolve_model — он сам делает турнирный отбор
                if profile_paths:
                    env.evolve_model(profile_paths)

                if env.has_demo and env.demo_access_count < 5:
                    try:
                        env.debug_print(">>> СТАРТ ПРЕТРЕЙНА")
                        env.pretrain_with_demo(model)
                        env.debug_print(">>> КОНЕЦ ПРЕТРЕЙНА")
                    except Exception as e:
                        env.debug_print(f"Ошибка в pretrain_with_demo для агента {agent_id}: {e}")
                        traceback.print_exc()

                env.model = model
                save_path = os.path.join(profile_dir, "model.zip")
                callbacks = [StopCallback(env.stop_event), SaveCallback(save_path, save_freq=TIMESTEPS_PER_AGENT, env=env)]
                remaining_timesteps = max(0, timesteps_per_agent - env.frame_processed)

                while remaining_timesteps > 0 and not env.stop_event.is_set() and not global_stop_event.is_set():
                    steps_to_run = min(TIMESTEPS_PER_AGENT, remaining_timesteps)
                    try:
                        model.learn(total_timesteps=steps_to_run, callback=callbacks)
                    except Exception as e:
                        env.debug_print(f"Ошибка обучения модели на шаге {env.frame_processed}: {e}")
                        traceback.print_exc()
                        break

                    remaining_timesteps -= steps_to_run

                    if env.has_demo and env.demo_access_count < 5 and env.frame_processed > 0:
                        try:
                            env.pretrain_with_demo(model)
                            env.last_demo_step = env.frame_processed
                        except Exception as e:
                            env.debug_print(f"Ошибка в pretrain_with_demo на шаге {env.frame_processed}: {e}")
                            traceback.print_exc()

                env.save_training_state()
                config = env.load_config()
                config["total_reward"] = env.total_reward
                env.save_config(config)

                current_agent = agent_id + 1
                save_global_state(safe_game_name, current_gen, current_agent, profile_paths, generations, num_agents)

                # Фикс 5: обнуляем env после cleanup, чтобы finally не вызвал его повторно
                env.cleanup()
                env = None
                time.sleep(1)

                if global_stop_event.is_set():
                    print("Глобальный стоп-событие установлено, пропуск дальнейших агентов")
                    break

            # Проверяем, завершили ли мы всех агентов
            if current_agent >= num_agents:
                print(f"Обработка профилей для следующего поколения: {current_profile_paths}")
                performances = []
                for profile_path in current_profile_paths:
                    config_path = os.path.join(profile_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            cfg = json.load(f)
                        performances.append((profile_path, cfg.get("total_reward", 0.0)))
                    else:
                        print(f"Предупреждение: файл config.json не найден в {profile_path}")
                print(f"Сформирован список performances: {performances}")
                if not performances:
                    # Фикс 4: была f-строка без f, {current_profile_paths} не подставлялся
                    print(f"Ошибка: список performances пуст, проверьте наличие config.json в {current_profile_paths}")

                if performances:
                    performances.sort(key=lambda x: x[1], reverse=True)
                    num_elite = max(1, int(num_agents * elite_fraction))
                    num_random = max(1, int(num_agents * random_fraction))
                    elite_paths = [p[0] for p in performances[:num_elite]]
                    remaining = performances[num_elite:]
                    random_paths = [p[0] for p in random.sample(remaining, min(num_random, len(remaining)))] if remaining else []
                    profile_paths = elite_paths + random_paths
                    print(f"Отобрано {len(profile_paths)} профилей для следующего поколения: {len(elite_paths)} элитных, {len(random_paths)} случайных")

                current_agent = 0
                current_gen += 1
                save_global_state(safe_game_name, current_gen, current_agent, profile_paths, generations, num_agents)
            else:
                save_global_state(safe_game_name, current_gen, current_agent, profile_paths, generations, num_agents)
                break

        # В конце выбираем чемпиона
        if current_gen == generations and performances:
            champion_path = performances[0][0]
            champion_reward = performances[0][1]
            print(f"Чемпион найден в поколении {current_gen-1}: {champion_path} с наградой {champion_reward}")
            final_profile_id = f"champion_gen_{current_gen-1}"
            final_profile_dir = f"profiles/{safe_game_name}/gen_{current_gen-1}_agent_{final_profile_id}"
            os.makedirs(final_profile_dir, exist_ok=True)

            final_env = GameEnv(game_name, profile_id=final_profile_id, is_final_profile=True, generation=current_gen-1)
            final_env.evolve_model([champion_path])
            final_env.save_training_state(save_path=final_profile_dir)
            save_global_state(safe_game_name, current_gen-1, final_profile_id, [final_profile_dir], generations, num_agents)
            final_env.cleanup()
            final_env = None
            print(f"Чемпион сохранён в {final_profile_dir}")

    except Exception as e:
        print(f"Ошибка в main: {e}")
        traceback.print_exc()
    finally:
        global_stop_event.set()
        # Фикс 5: env и final_env объявлены до try, проверяем просто на None
        if env is not None:
            env.cleanup()
        if final_env is not None:
            final_env.cleanup()
        if ENABLE_GUI:
            gui_stop_event.set()
            gui_thread_instance.join(timeout=5.0)
        state_path = f"profiles/{safe_game_name}/training_state.json"
        if current_gen >= generations:
            if os.path.exists(state_path):
                os.remove(state_path)
        else:
            save_global_state(safe_game_name, current_gen, current_agent, profile_paths, generations, num_agents)
        print("Программа полностью остановлена")

if __name__ == "__main__":
    main()
