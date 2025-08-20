import ctypes
from ctypes import wintypes
SendInput = ctypes.windll.user32.SendInput
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

global_log_queue = queue.Queue(maxsize=100)
global_stop_event = threading.Event()

# Конфигурация

OBS_CAMERA_INDEX = 2
YOLO_MODEL_PATH = "yolov8m.pt"
ENABLE_GUI = True
DEFAULT_FRAME_RESIZE = (640, 480)
DEMO_RECORD_DURATION = 900  # 15 минут в секундах
DEMO_VIDEO_FPS = 30  # FPS для записи видео

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

def send_input(inputs):
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    cbsize = ctypes.sizeof(INPUT)
    res = user32.SendInput(n, ctypes.byref(arr), cbsize)
    return res

def make_key_input(vk, down=True):
    sc = user32.MapVirtualKeyW(vk, 0)
    flags = KEYEVENTF_SCANCODE
    if not down:
        flags |= KEYEVENTF_KEYUP
    ki = KEYBDINPUT(0, sc, flags, 0, ctypes.pointer(ctypes.c_ulong(0)))
    return INPUT(INPUT_KEYBOARD, _INPUTunion(ki=ki))

def make_mouse_move(dx, dy):
    mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(ctypes.c_ulong(0)))
    return INPUT(INPUT_MOUSE, _INPUTunion(mi=mi))

def make_mouse_click(button='left', down=True):
    if button.lower() in ('left', 'lmb'):
        flags = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
    elif button.lower() in ('right', 'rmb'):
        flags = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
    else:
        raise ValueError(f"Unknown mouse button: {button}")
    mi = MOUSEINPUT(0, 0, 0, flags, 0, ctypes.pointer(ctypes.c_ulong(0)))
    return INPUT(INPUT_MOUSE, _INPUTunion(mi=mi))

def set_mouse_position(x, y):
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    abs_x = int(x * 65535 / (screen_w - 1))
    abs_y = int(y * 65535 / (screen_h - 1))
    mi = MOUSEINPUT(abs_x, abs_y, 0, MOUSEEVENTF_MOVE | 0x8000, 0, ctypes.pointer(ctypes.c_ulong(0)))
    return INPUT(INPUT_MOUSE, _INPUTunion(mi=mi))


# Полный мэппинг виртуальных ключей (без '=')

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

# FeatureNet

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
        
class SaveCallback:
    def __init__(self, save_path, save_freq, env):
        self.save_path = save_path
        self.save_freq = save_freq
        self.env = env
        self.step = 0

    def __call__(self, *args, **kwargs):
        self.step += 1
        if self.step % self.save_freq == 0:
            self.env.save_training_state(self.save_path)
            self.env.debug_print(f"Сохранение модели на шаге {self.step}")
        return False

class FrameCaptureThread(threading.Thread):
    def __init__(self, env):
        super().__init__(daemon=True)
        self.env = env
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running and not self.env.stop_event.is_set():
            # Захват всегда, даже на паузе
            for _ in range(5):
                self.env.cap.grab()
            ret, frame = self.env.cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
                # В очередь только если не пауза
                if not self.env.pause_event.is_set():
                    try:
                        if self.env.frame_queue.full():
                            self.env.frame_queue.get_nowait()
                        self.env.frame_queue.put(frame, timeout=0.01)
                    except queue.Full:
                        pass
            time.sleep(0.005)

# Основной класс среды

class GameEnv(gym.Env):
    active_gui = None

    def __init__(self, game_name="", profile_id=0, is_final_profile=False, generation=0):
        super(GameEnv, self).__init__()
        self.threads = []
        self.profile_id = profile_id
        self.generation = generation
        self.agent_id = profile_id
        self.is_final_profile = is_final_profile
        self.is_gui_destroyed = False
        self.total_reward = 0.0
        self.profile_performance = {}
        self.is_shooter = False
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
        self.prev_frame = None
        self.prev_ocr_text = ""
        self.prev_objects = set()
        self.known_objects = {}
        self.frame_count = 0
        self.frame_processed = 0
        self.annotated_frame = None
        self.yolo_results = None
        self.last_api_call = 0
        self.api_call_interval = 0.1  # Уменьшено для FPS
        self.ui_mode = False
        self.current_frame = None
        self.current_reward = 0.0
        self.total_timesteps = 20000
        self.game_name = game_name or self.detect_game_window_name()
        self.hwnd = win32gui.FindWindow(None, self.game_name)
        if not self.hwnd:
            self.debug_print("Окно игры не найдено, используется текущее активное окно")
            self.hwnd = win32gui.GetForegroundWindow()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.debug_print(f"CUDA доступен: {torch.cuda.get_device_name(0)}")
        else:
            self.debug_print("CUDA недоступен, используется CPU")
        self.model_yolo = YOLO(YOLO_MODEL_PATH).to(self.device)
        self.debug_print(f"YOLO device: {self.model_yolo.device}")
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,3), dtype=np.uint8)
        self.actions = [
            ("wait", []), ("w", ["w"]), ("a", ["a"]), ("s", ["s"]), ("d", ["d"]),
            ("space", ["space"]), ("shift", ["shift"]), ("e", ["e"]), ("ctrl", ["ctrl"]),
            ("click_left", ["click_left"]), ("click_right", ["click_right"]),
            ("camera_left", ["camera_left"]), ("camera_right", ["camera_right"]),
            ("camera_up", ["camera_up"]), ("camera_down", ["camera_down"]),
            ("aim_at_person", ["aim_at_person"]), ("hold_left", ["hold_left"])
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_map = [
            {'keys': [], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.15},
            {'keys': ['w'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.5},
            {'keys': ['a'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.5},
            {'keys': ['s'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.5},
            {'keys': ['d'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.5},
            {'keys': ['space'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.2},
            {'keys': ['shift'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.5},
            {'keys': ['ctrl'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.5},
            {'keys': ['e'], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.15},
            {'keys': [], 'mouse_move': (0,0), 'mouse_click': 'left', 'duration': 0.15},
            {'keys': [], 'mouse_move': (0,0), 'mouse_click': 'right', 'duration': 0.15},
            {'keys': [], 'mouse_move': (-700,0), 'mouse_click': None, 'duration': 0.12},
            {'keys': [], 'mouse_move': (700,0), 'mouse_click': None, 'duration': 0.12},
            {'keys': [], 'mouse_move': (0,-100), 'mouse_click': None, 'duration': 0.12},
            {'keys': [], 'mouse_move': (0,100), 'mouse_click': None, 'duration': 0.12},
            {'keys': [], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.2, 'aim_at_person': True},
            {'keys': [], 'mouse_move': (0,0), 'mouse_click': 'hold_left', 'duration': 0.2, 'aim_at_person': True}
        ]
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
        self.capture_thread = FrameCaptureThread(self)
        self.capture_thread.start()
        self.threads.append(self.capture_thread)
        self.send_worker_thread = threading.Thread(target=self._send_worker, daemon=True)
        self.threads.append(self.send_worker_thread)
        self.send_worker_thread.start()
        self.action_worker_thread = threading.Thread(target=self._action_worker, daemon=True)
        self.threads.append(self.action_worker_thread)
        self.action_worker_thread.start()
        self.frame_capture_thread = threading.Thread(target=self.frame_capture_worker, daemon=True)
        self.threads.append(self.frame_capture_thread)
        self.frame_capture_thread.start()
        self.yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self.threads.append(self.yolo_thread)
        self.yolo_thread.start()
        if ENABLE_GUI and GameEnv.active_gui is None:
            self.debug_print(f"Starting GUI for profile_id={self.profile_id}, is_final_profile={self.is_final_profile}")
            GameEnv.active_gui = self
            gui_thread_instance = threading.Thread(target=gui_thread, args=(self, self.stop_event), daemon=False)
            self.threads.append(gui_thread_instance)
            gui_thread_instance.start()
            self.debug_print("Запущен поток GUI")
        else:
            self.debug_print(f"Пропуск GUI для profile_id={self.profile_id}, is_final_profile={self.is_final_profile}, active_gui={GameEnv.active_gui}")
        console_thread = threading.Thread(target=self.console_listener, args=(self.stop_event, self.pause_event), daemon=True)
        self.threads.append(console_thread)
        console_thread.start()
        self.restrict_cursor_to_window()
        self.demo_cache = None
        self.demo_access_count = 0
        self.demo_video_path = f"profiles/{self.game_name}/shared_demo/demo_video.avi"
        self.demo_actions_path = f"profiles/{self.game_name}/shared_demo/demo_actions.json"
        self.has_demo = os.path.exists(self.demo_video_path) and os.path.exists(self.demo_actions_path)
        self.demo_actions = []
        self.behavior_archetype = None
        if self.has_demo:
            self.load_demo_actions()
            self.demo_access_count = 0
            self.analyze_demo_behavior()
        self.pause_event.set()

    def debug_print(self, message):
        if not self.pause_event.is_set():
            print(message)

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
        sequence_groups = [(key, sum(1 for _ in group)) for key, group in groupby(sequences)]
        common_sequences = sorted(sequence_groups, key=lambda x: x[1], reverse=True)[:5]

        # Анализ движения мыши (средние dx, dy)
        mouse_moves = [action['mouse_pos'] for action in self.demo_actions if action['mouse_pos'] != [0, 0]]
        if mouse_moves:
            avg_dx = sum(abs(pos[0]) for pos in mouse_moves) / len(mouse_moves)
            avg_dy = sum(abs(pos[1]) for pos in mouse_moves) / len(mouse_moves)
        else:
            avg_dx, avg_dy = 0, 0

        self.behavior_archetype = {
            "most_common_actions": most_common_actions,
            "common_sequences": common_sequences,
            "average_mouse_movement": (avg_dx, avg_dy)
        }
        self.debug_print(f"Архетип поведения из демо: {json.dumps(self.behavior_archetype, indent=2)}")

    def save_training_state(self, save_path=None):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path or self.get_profile_dir()
        os.makedirs(save_path, exist_ok=True)
        try:
            if hasattr(self, 'model') and self.model is not None:
                ppo_save_path = os.path.join(save_path, f"model.zip")
                self.model.save(ppo_save_path)
                self.debug_print(f"PPO модель сохранена в {ppo_save_path}, Frames Processed: {self.frame_processed}")
                config = self.load_config()
                config["total_reward"] = self.total_reward
                config["frame_processed"] = self.frame_processed
                self.save_config(config)
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении PPO модели: {e}")
        try:
            feature_net_save_path = os.path.join(save_path, f"feature_net_{timestamp}.pth")
            torch.save(self.feature_net.state_dict(), feature_net_save_path)
            self.debug_print(f"Feature net сохранена в {feature_net_save_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении feature_net: {e}")
        try:
            forward_model_save_path = os.path.join(save_path, f"forward_model_{timestamp}.pth")
            torch.save(self.forward_model.state_dict(), forward_model_save_path)
            self.debug_print(f"Forward model сохранена в {forward_model_save_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при сохранении forward_model: {e}")

    def load_training_state(self, ppo_path=None, feature_net_path=None, forward_model_path=None):
        try:
            if ppo_path and hasattr(self, 'model') and self.model is not None:
                self.model = PPO.load(ppo_path, env=self)
                self.debug_print(f"PPO модель загружена из {ppo_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при загрузке PPO модели: {e}")
        try:
            if feature_net_path:
                self.feature_net.load_state_dict(torch.load(feature_net_path))
                self.feature_net.to(self.device).float()
                self.debug_print(f"Feature net загружена из {feature_net_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при загрузке feature_net: {e}")
        try:
            if forward_model_path:
                self.forward_model.load_state_dict(torch.load(forward_model_path))
                self.forward_model.to(self.device).float()
                self.debug_print(f"Forward model загружена из {forward_model_path}")
        except Exception as e:
            self.debug_print(f"Ошибка при загрузке forward_model: {e}")
        self.optimizer = torch.optim.Adam(list(self.feature_net.parameters()) + list(self.forward_model.parameters()), lr=1e-3)

    def evolve_model(self, profile_paths):
        """
        Эволюция модели: выбирает лучшую модель по total_reward и применяет мутации или скрещивание.
        profile_paths: список путей к профилям с их конфигами и моделями.
        """
        # Собираем производительность всех профилей
        performances = []
        for profile_path in profile_paths:
            config_path = os.path.join(profile_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    total_reward = config.get("total_reward", 0.0)
                    performances.append((profile_path, total_reward))
        
        if not performances:
            self.debug_print("Нет доступных профилей для эволюции")
            return None
        
        # Сортируем по суммарной награде (убывание)
        performances.sort(key=lambda x: x[1], reverse=True)
        best_profile, best_reward = performances[0]
        self.debug_print(f"Лучший профиль: {best_profile} с наградой {best_reward}")

        # Загружаем лучшую модель
        ppo_files = [f for f in os.listdir(best_profile) if f.endswith(".zip")]
        feature_net_files = [f for f in os.listdir(best_profile) if f.startswith("feature_net_") and f.endswith(".pth")]
        forward_model_files = [f for f in os.listdir(best_profile) if f.startswith("forward_model_") and f.endswith(".pth")]
        
        if not (ppo_files and feature_net_files and forward_model_files):
            self.debug_print("Не найдены файлы модели в лучшем профиле")
            return None
        
        latest_ppo = max(ppo_files, key=lambda f: os.path.getmtime(os.path.join(best_profile, f)))
        latest_feature_net = max(feature_net_files, key=lambda f: os.path.getmtime(os.path.join(best_profile, f)))
        latest_forward_model = max(forward_model_files, key=lambda f: os.path.getmtime(os.path.join(best_profile, f)))
        
        self.load_training_state(
            ppo_path=os.path.join(best_profile, latest_ppo),
            feature_net_path=os.path.join(best_profile, latest_feature_net),
            forward_model_path=os.path.join(best_profile, latest_forward_model)
        )
        
        # Применяем мутации к feature_net и forward_model
        with torch.no_grad():
            for param in self.feature_net.parameters():
                noise = torch.randn_like(param) * 0.05  # Небольшая мутация (5% шума)
                param.add_(noise)
            for param in self.forward_model.parameters():
                noise = torch.randn_like(param) * 0.05
                param.add_(noise)
        
        self.debug_print("Эволюция модели завершена: загружен лучший профиль с мутацией")
        return best_profile

    def cleanup(self):
        self.debug_print(f"Очистка GameEnv для profile_id={self.profile_id}")
        self.save_training_state()
        save_global_state(self.game_name, self.generation, int(self.profile_id) + 1,
                  [self.get_profile_dir()], self.total_timesteps, 100)  # Используем значения из env, profile_paths с текущим
        self.stop_event.set()
        self.pause_event.set()
        self.is_gui_destroyed = True
        self.release_all_keys()
        for q in [self.frame_queue, self.action_queue, self.send_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.debug_print("Освобождение видеозахвата")
        if GameEnv.active_gui:
            try:
                GameEnv.active_gui.quit()  # Завершаем цикл mainloop
                GameEnv.active_gui.destroy()  # Закрываем окно
                GameEnv.active_gui = None
                self.debug_print("GUI закрыт в cleanup")
            except Exception as e:
                self.debug_print(f"Ошибка при закрытии GUI в cleanup: {e}")
        self.threads = [t for t in self.threads if t.is_alive()]
        for thread in self.threads:
            if thread.is_alive():
                self.debug_print(f"Ожидание потока {thread.name} для завершения")
                if hasattr(thread, 'running'):
                    thread.running = False
                thread.join(timeout=3.0)
                if thread.is_alive():
                    self.debug_print(f"Поток {thread.name} не завершился вовремя, игнорируем")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.debug_print("Очищена память GPU")
        self.debug_print("Очистка завершена")

    def console_listener(self, stop_event: threading.Event, pause_event: threading.Event):
        self.debug_print("Консольный слушатель запущен. Нажмите '=' для паузы/возобновления, 'Caps Lock' для остановки и сохранения, 'F12' для записи/остановки демонстрации")
        while not stop_event.is_set() and not global_stop_event.is_set():
            try:
                if keyboard.is_pressed("="):
                    if self.recording_event.is_set():
                        self.debug_print("Нельзя использовать '=' во время записи демонстрации")
                        global_log_queue.put("Нельзя ставить на паузу/возобновлять во время записи")
                    elif not pause_event.is_set():
                        pause_event.set()
                        self.debug_print("Бот остановлен на '='")
                        global_log_queue.put("Бот приостановлен\nНажмите F12 для записи демонстрации")
                    else:
                        if self.has_demo:
                            pause_event.clear()
                            self.debug_print("Бот возобновлен на '='")
                            global_log_queue.put("Бот возобновлен")
                        else:
                            self.debug_print("Нельзя возобновить: отсутствует фрагмент игры. Запишите демонстрацию (F12)")
                            global_log_queue.put("Нельзя запустить: отсутствует демонстрация\nНажмите F12 для записи")
                    while keyboard.is_pressed("="):
                        time.sleep(0.01)  # Ждём отпускания клавиши
                if keyboard.is_pressed("f12"):
                    if not self.recording_active:
                        # --- старт записи ---
                        self.recording_event.set()
                        self.stop_recording_event.clear()
                        self.recording_active = True
                        threading.Thread(target=self.record_demo, daemon=True).start()
                        self.debug_print("Начата запись демонстрации")
                        global_log_queue.put("Начата запись демонстрации")
                    else:
                        # --- остановка записи ---
                        self.stop_recording_event.set()
                        self.recording_event.clear()
                        self.recording_active = False
                        self.debug_print("Остановлена запись демонстрации")
                        global_log_queue.put("Остановлена запись демонстрации")
                        if hasattr(self, "active_gui") and self.active_gui:
                            for widget in self.active_gui.winfo_children():
                                if isinstance(widget, tk.Label):
                                    widget.config(text="Bot paused")

                    # ждём отпускания клавиши, чтобы не сработало несколько раз подряд
                    while keyboard.is_pressed("f12"):
                        time.sleep(0.05)
                if keyboard.is_pressed("capslock"):
                    self.debug_print("Caps Lock нажат, выключение бота")
                    global_log_queue.put("Caps Lock нажат, сохранение состояния и остановка")
                    self.save_training_state()
                    profile_paths = [self.get_profile_dir()]  # Текущий профиль агента
                    completed = self.frame_processed >= 20000  # Убедитесь, что это совпадает с timesteps_per_agent
                    next_agent = int(self.profile_id)  
                    save_global_state(self.game_name, self.generation, next_agent, profile_paths, 1000, 100)
                    stop_event.set()
                    global_stop_event.set()
                    self.release_all_keys()
                    try:
                        if self.cap and self.cap.isOpened():
                            self.cap.release()
                            self.debug_print("Видеозахват освобождён")
                    except:
                        self.debug_print("Ошибка при освобождении видеозахвата")
                    self.debug_print("Бот остановлен и состояние сохранено")
                    global_log_queue.put("Бот остановлен и сохранён")
                    # Принудительный выход даже в паузе
                    self.pause_event.clear()
                    while keyboard.is_pressed("capslock"):
                        time.sleep(0.01)
                    return
                time.sleep(0.01)
            except Exception as e:
                self.debug_print(f"Ошибка в console_listener: {e}")
                time.sleep(1)
        sys.exit(0)

    def record_demo(self):
        """Запись видео и действий игрока в течение 30 мин или до остановки"""
        try:
            os.makedirs(os.path.dirname(self.demo_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(self.demo_video_path, fourcc, DEMO_VIDEO_FPS, (self.width, self.height))
            self.demo_actions = []
            start_time = time.time()
            last_action_time = start_time
            
            key_map = {
            'minus': '-', 'comma': ',', 'period': '.', 'slash': '/', 
            'tilde': '`', 'lbracket': '[', 'rbracket': ']', 'backslash': '\\',
            'apostrophe': "'", 'semicolon': ';'
            }
            
            # Список клавиш для отслеживания (из VK)
            tracked_keys = []
            seen_keys = set()
            for key in VK.keys():
                mapped_key = key_map.get(key, key)
                try:
                    # Проверяем, поддерживается ли клавиша библиотекой keyboard
                    keyboard.is_pressed(mapped_key)
                    if mapped_key not in seen_keys:
                        tracked_keys.append(mapped_key)
                        seen_keys.add(mapped_key)
                except ValueError:
                    self.debug_print(f"Клавиша {mapped_key} не поддерживается библиотекой keyboard, пропускаем")
                    continue
                    
            user32 = ctypes.windll.user32
            user32.ClipCursor(None)
            self.debug_print("Попытка отключить ограничение курсора для записи")
        
            while (time.time() - start_time < DEMO_RECORD_DURATION and 
                   not self.stop_event.is_set() and 
                   not self.stop_recording_event.is_set()):
                ret, frame = self.cap.read()
                if ret:
                    video_writer.write(frame)
                    # Запись действий (клавиш и мыши)
                    current_time = time.time()
                    if current_time - last_action_time >= 0.1:  # Записываем каждые 0.1 сек
                        # Захват нажатых клавиш
                        pressed_keys = set()
                        for key in tracked_keys:
                            try:
                                if keyboard.is_pressed(key):
                                    pressed_keys.add(key)
                            except ValueError as e:
                                self.debug_print(f"Ошибка при проверке клавиши {key}: {e}")
                                continue
                    
                        # Захват состояния мыши
                        mouse_buttons = []
                        mouse_state = win32api.GetKeyState(0x01)  # ЛКМ
                        if mouse_state < -1:  # Нажата
                            mouse_buttons.append("left")
                        mouse_state = win32api.GetKeyState(0x02)  # ПКМ
                        if mouse_state < -1:
                            mouse_buttons.append("right")
                    
                        # Получаем позицию мыши
                        try:
                            mouse_pos = win32api.GetCursorPos()
                            self.debug_print(f"Позиция мыши: {mouse_pos}")
                        except Exception as e:
                            self.debug_print(f"Ошибка при получении позиции мыши: {e}")
                            mouse_pos = [960, 540]  #
                        
                        actions = {
                            'timestamp': current_time - start_time,
                            'keys': list(pressed_keys),
                            'mouse_pos': list(mouse_pos),
                            'mouse_buttons': mouse_buttons
                        }
                        self.demo_actions.append(actions)
                        last_action_time = current_time
                    time.sleep(1 / DEMO_VIDEO_FPS)
            
                else:
                    self.debug_print("Не удалось захватить кадр во время записи")
        
            # Сохраняем действия в файл
            try:
                with open(self.demo_actions_path, 'w') as f:
                    json.dump(self.demo_actions, f, indent=4)
                self.debug_print(f"Действия сохранены в {self.demo_actions_path}")
                self.has_demo = True  # Обновляем флаг, так как демонстрация записана
                self.analyze_demo_behavior()  # Анализ после записи
            except Exception as e:
                self.debug_print(f"Ошибка при сохранении demo_actions.json: {e}")
        
            # Освобождаем ресурсы
            video_writer.release()
            self.debug_print("Запись демонстрации завершена, видео сохранено")
            self.recording_event.clear()

            # Восстанавливаем ограничение курсора
            self.restrict_cursor_to_window()
        
        except Exception as e:
            self.debug_print(f"Ошибка в record_demo: {e}")
            try:
                video_writer.release()
            except:
                pass
            # Восстанавливаем ограничение курсора в случае ошибки
            self.recording_event.clear() 
            self.restrict_cursor_to_window()

    def load_demo_actions(self):
        """Загрузка действий из демонстрации для IRL"""
        try:
            with open(self.demo_actions_path, 'r') as f:
                self.demo_actions = json.load(f)
            self.debug_print(f"Загружено {len(self.demo_actions)} действий из демонстрации")
            self.analyze_demo_behavior()  # Анализ после загрузки
        except Exception as e:
            self.debug_print(f"Ошибка загрузки демонстрации: {e}")
            self.has_demo = False

    def pretrain_with_demo(self, model):
        if not self.has_demo or not self.demo_actions:
            self.debug_print("Нет демонстраций для предобучения")
            return
        # Путь для кэша
        cache_path = f"profiles/{self.game_name}/shared_demo/demo_segment_cache.pkl"
    
        # Проверяем наличие кэша
        start_frame = 0  # Значение по умолчанию для start_frame
        if hasattr(self, 'demo_cache') and self.demo_cache:
            frames, segment_actions = self.demo_cache
            self.debug_print("Используется кэшированный сегмент демки")
        else:
            # Открываем видео демки
            cap = cv2.VideoCapture(self.demo_video_path)
            if not cap.isOpened():
                self.debug_print("Не удалось открыть видео демонстрации")
                cap.release()
                return

            # Определяем длительность демки
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = DEMO_VIDEO_FPS  # 30 FPS
            segment_duration = 3  # Уменьшаем до 3 секунд для оптимизации
            segment_frames = segment_duration * fps  # 90 кадров

            # Выбираем случайный начальный кадр
            max_start_frame = max(0, total_frames - segment_frames)
            start_frame = random.randint(0, max_start_frame)
            start_time = start_frame / fps
            end_time = start_time + segment_duration

            # Находим действия в сегменте
            segment_actions = [
                action for action in self.demo_actions
                if start_time <= action['timestamp'] <= end_time
            ]

            if not segment_actions:
                self.debug_print("Нет действий в выбранном 3-секундном сегменте")
                cap.release()
                return

            # Читаем и кэшируем кадры
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for i in range(segment_frames):
                ret, frame = cap.read()
                if not ret:
                    self.debug_print(f"Не удалось прочитать кадр {i} из демки")
                    break
                # Ресайзим сразу до размера наблюдения (84x84)
                frame = cv2.resize(frame, (84, 84))
                frames.append(frame)

            cap.release()

            # Сохраняем в кэш (в память или файл)
            self.demo_cache = (frames, segment_actions)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.demo_cache, f)
            self.debug_print(f"Кэширован сегмент демки с кадра {start_frame}")

        # Определение функции маппинга действий
        def map_demo_action_to_index(demo_action):
            keys = demo_action.get('keys', [])
            mouse_buttons = demo_action.get('mouse_buttons', [])
            mouse_pos = demo_action.get('mouse_pos', [0, 0])

            for idx, action in enumerate(self.actions):
                action_name, action_keys = action
                if set(keys).issubset(action_keys):
                    return idx
                if "left" in mouse_buttons and action_name == "click_left":
                    return idx
                if "right" in mouse_buttons and action_name == "click_right":
                    return idx
                if mouse_pos[0] != 0 or mouse_pos[1] != 0:
                    if action_name in ["camera_left", "camera_right", "camera_up", "camera_down", "aim_at_person"]:
                        return idx
            return 0  # "wait" по умолчанию

        # Обработка сегмента
        fps = DEMO_VIDEO_FPS  # Определяем fps для использования в цикле
        for i, frame in enumerate(frames):
            # Находим ближайшее действие
            current_time = (start_frame + i) / fps
            closest_action = min(
                segment_actions,
                key=lambda x: abs(x['timestamp'] - current_time),
                default=None
            )

            if closest_action:
                try:
                    obs = self.get_state(frame)
                    action_idx = map_demo_action_to_index(closest_action)
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        logits = model.policy(obs_tensor.unsqueeze(0))
                    target = torch.tensor([action_idx], dtype=torch.long, device=self.device)
                    loss = nn.CrossEntropyLoss()(logits, target)
                    model.policy.optimizer.zero_grad()
                    loss.backward()
                    model.policy.optimizer.step()
                except Exception as e:
                    self.debug_print(f"Ошибка предобучения на кадре {i}: {e}")

        self.demo_access_count += 1
        self.debug_print(f"Предобучение завершено на 3-секундном сегменте" + (f" с кадра {start_frame}" if start_frame else ""))

    def detect_game_window_name(self):
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        return title if title else "UnknownGame"

    def set_game_description(self, desc):
        self.game_description = desc
        config = self.load_config()
        config["game_description"] = desc
        self.save_config(config)

    def fetch_game_description(self, lang="en"):
        try:
            game = self.game_name.replace(" ", "_")
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{game}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                desc = data.get("extract", "")
                if desc:
                    self.set_game_description(desc)
                    return desc
            # если ничего не нашли — запасной вариант
            self.set_game_description("Описание игры не найдено в Википедии.")
            return "Описание игры не найдено в Википедии."
        except Exception as e:
            self.set_game_description("Описание игры недоступно (ошибка API).")
            return "Описание игры недоступно (ошибка API)."

    def analyze_game_description(self):
        if not self.game_description:
            return []
        description = self.game_description.lower()
        shooter_keywords = ["shoot", "shooter", "first-person", "fps", "third-person", "tps", "kill", "eliminate", "aim", "weapon", "gun", "battle", "combat", "fire", "reload"]
    # Проверяем, является ли игра шутером
        is_shooter = any(kw in description for kw in shooter_keywords) or any(shooter_game in self.game_name.lower() for shooter_game in ["counter-strike", "call of duty", "battlefield", "valorant", "overwatch", "cs2"])
        keywords = ["collect", "build", "craft", "survive", "attack", "jump", "interact", "kill", "move", "delivering"]
        if "minecraft" in self.game_name.lower():
            keywords.extend(["ender dragon", "end portal", "craft", "ender pearl", "blaze rod"])
        if is_shooter:
            keywords = ["shoot", "kill", "eliminate", "plant", "defuse", "objective", "aim", "reload"]
        found_keywords = [kw for kw in keywords if kw in description]
        self.debug_print(f"Извлечённые ключевые слова из описания: {found_keywords}")
        return found_keywords, is_shooter

    def get_game_window_rect(self):
        hwnd = win32gui.GetForegroundWindow()  # Получаем активное окно
        if hwnd:
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            if not (style & win32con.WS_VISIBLE):
                self.debug_print("Активное окно не видимо")
                return None
            rect = win32gui.GetWindowRect(hwnd)
            title = win32gui.GetWindowText(hwnd)
            return rect
        self.debug_print("Активное окно не найдено")
        return None

    def restrict_cursor_to_window(self):
        if self.ui_mode:
            rect = self.get_game_window_rect()
            if rect:
                win32api.ClipCursor(rect)
            else:
                pass
        else:
            ctypes.windll.user32.ClipCursor(None)

    def _yolo_worker(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((640, 640))  # Уменьшаем размер для ускорения
        ])
        frame_counter = 0
        skip_frames = 4
        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                with self.lock:
                    if self.current_frame is None:
                        time.sleep(0.05)
                        continue
                    frame = self.current_frame.copy()
                if frame is None or frame.size == 0: 
                    time.sleep(0.05)
                    continue
                frame_counter += 1
                if frame_counter % skip_frames == 0:
                    frame_tensor = transform(frame).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        results = self.model_yolo(frame_tensor, classes=[0], conf=0.3, verbose=False)
                    with self.lock:
                        self.yolo_results = results
                        self.annotated_frame = results[0].plot() if results else frame.copy()
                        self.annotated_frame = cv2.resize(self.annotated_frame, (self.width, self.height))
                time.sleep(0.05)
            except Exception as e:
                self.debug_print(f"YOLO worker error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _send_worker(self):
        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
                   self.release_all_keys()  # Освобождаем клавиши при паузе
                   time.sleep(0.1)
                   continue
                item = self.send_queue.get(timeout=0.05)
                typ = item.get('type')
                rect = self.get_game_window_rect()
                if rect:
                    hwnd = win32gui.FindWindow(None, self.game_name)
                    if hwnd:
                        current_foreground = win32gui.GetForegroundWindow()
                        if current_foreground != hwnd:
                            self.debug_print(f"Текущее активное окно: {win32gui.GetWindowText(current_foreground)}")
                            win32gui.SetForegroundWindow(hwnd)
                            self.debug_print(f"Установлен фокус на окно: {self.game_name} (HWND: {hwnd})")
                            time.sleep(0.02)
                    else:
                        self.debug_print(f"Окно игры '{self.game_name}' не найдено для фокуса")
                        continue
                else:
                    self.debug_print("Не удалось получить координаты окна для фокуса")
                    continue
                if typ == 'press_keys':
                    vks = item.get('vks', [])
                    inputs = [make_key_input(vk, down=True) for vk in vks]
                    if inputs:
                        send_input(inputs)
                        for vk in vks:
                            key_name = vk_to_name(vk)
                            global_log_queue.put(f"Нажата клавиша: {key_name}")
                elif typ == 'release_keys':
                    vks = item.get('vks', [])
                    inputs = [make_key_input(vk, down=False) for vk in vks]
                    if inputs:
                        send_input(inputs)
                        for vk in vks:
                            global_log_queue.put(f"Отпущена клавиша: {vk}")
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
                        send_input([set_mouse_position(x, y)])
                        global_log_queue.put(f"Абсолютное движение мыши: x={x}, y={y}")
                elif typ == 'mouse_click':
                    btn = item.get('button', 'left')
                    down = item.get('down', True)
                    send_input([make_mouse_click(btn, down)])
                    click_type = "нажатие" if down else "отпускание"
                    global_log_queue.put(f"{click_type} кнопки мыши ({btn})")
                time.sleep(0.001)
            except queue.Empty:
                continue
            except Exception as e:
                self.debug_print(f"Send worker error: {e}")

    def _action_worker(self):
        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                action_item = self.action_queue.get(timeout=0.02)
                keys = action_item.get('keys', [])
                dx, dy = action_item.get('mouse_move', (0,0))
                click = action_item.get('mouse_click', None)
                duration = action_item.get('duration', 0.15)
                steps = max(1, int(duration / 0.03))
                vk_list = []
                for k in keys:
                    if k.startswith('hold_'):
                        base = k.split('hold_')[-1]
                        vk = key_to_vk(base)
                        if vk:
                            with self.held_keys_lock:
                                if base not in self.held_keys:
                                    self.held_keys.add(base)
                                    self.send_queue.put({'type':'press_keys','vks':[vk]})
                    else:
                        vk = key_to_vk(k)
                        if vk:
                            vk_list.append(vk)
                if vk_list:
                    self.send_queue.put({'type':'press_keys','vks':vk_list})
                for i in range(steps):
                    sub_dx = int(dx/steps)
                    sub_dy = int(dy/steps)
                    if sub_dx != 0 or sub_dy != 0:
                        self.send_queue.put({'type':'mouse_move','dx':sub_dx,'dy':sub_dy})
                    time.sleep(duration/steps)
                if action_item.get('human_mouse_smooth', False):
                    self.human_mouse_move_smooth()
                if click:
                    if click.startswith('hold_'):
                        btn = 'left' if 'left' in click else 'right'
                        if btn.upper() not in self.held_mouse:
                            self.held_mouse.add(btn.upper())
                            self.send_queue.put({'type':'mouse_click','button':btn,'down':True})
                    else:
                        btn = 'left' if 'left' in click else 'right'
                        self.send_queue.put({'type':'mouse_click','button':btn,'down':True})
                        time.sleep(0.02)
                        self.send_queue.put({'type':'mouse_click','button':btn,'down':False})
                if vk_list:
                    self.send_queue.put({'type':'release_keys','vks':vk_list})
            except queue.Empty:
                continue
            except Exception as e:
                self.debug_print(f"Action worker error:", {e})

    def frame_capture_worker(self):
        while not self.stop_event.is_set():
            with self.capture_thread.lock:
                if self.capture_thread.latest_frame is not None:
                    with self.lock:
                        # Копируем кадр всегда, чтобы YOLO и GUI видели его
                        self.current_frame = self.capture_thread.latest_frame.copy()
                        # Но если пауза — не кидаем в обучение/очередь
                        if not self.pause_event.is_set():
                            if self.frame_queue.full():
                                self.frame_queue.get_nowait()
                            self.frame_queue.put(self.current_frame, timeout=0.01)
            time.sleep(0.005)

    def hold_key(self, key_name):
        vk = key_to_vk(key_name)
        if not vk:
            return False
        with self.held_keys_lock:
            if key_name not in self.held_keys:
                self.held_keys.add(key_name)
                self.send_queue.put({'type':'press_keys','vks':[vk]})
        return True

    def release_key(self, key_name):
        vk = key_to_vk(key_name)
        if not vk:
            return False
        with self.held_keys_lock:
            if key_name in self.held_keys:
                self.held_keys.remove(key_name)
                self.send_queue.put({'type':'release_keys','vks':[vk]})
        return True

    def press_combo(self, keys_list, hold_time=0.05):
        vks = []
        for k in keys_list:
            vk = key_to_vk(k)
            if vk:
                vks.append(vk)
        if vks:
            self.send_queue.put({'type':'press_keys','vks':vks})
            if hold_time > 0:
                time.sleep(hold_time)
            self.send_queue.put({'type':'release_keys','vks':vks})

    def click(self, button='left'):
        btn = 'left' if button=='left' else 'right'
        self.send_queue.put({'type':'mouse_click','button':btn,'down':True})
        time.sleep(0.02)
        self.send_queue.put({'type':'mouse_click','button':btn,'down':False})

    def move_mouse(self, dx, dy):
        self.send_queue.put({'type':'mouse_move','dx':dx,'dy':dy})

    def release_all_keys(self):
        hwnd = win32gui.GetForegroundWindow()
        active_hwnd = hwnd
        if hwnd:
            title = win32gui.GetWindowText(hwnd)
            self.debug_print(f"Освобождение клавиш, активное окно: {title}")
            # Проверяем, что фокус на нужном окне
            game_hwnd = win32gui.GetForegroundWindow()
            if game_hwnd and game_hwnd != hwnd:
                try:
                    win32gui.SetForegroundWindow(game_hwnd)
                    self.debug_print("Фокус возвращён игровому окну перед освобождением клавиш")
                except:
                    self.debug_print("Не удалось вернуть фокус игровому окну")
        with self.held_keys_lock:
            for key in list(self.held_keys):
                vk = key_to_vk(key)
                if vk is not None:
                    self.send_queue.put({'type': 'release_keys', 'vks': [vk]})
                    self.debug_print(f"Отпущена клавиша: {key}")
            self.held_keys.clear()
        for button in list(self.held_mouse):
            self.send_queue.put({'type': 'mouse_click', 'button': button, 'down': False})
            self.debug_print(f"Отпущена кнопка мыши: {button}")
        self.held_mouse.clear()
        # Даём время на обработку команд отпускания
        time.sleep(0.1)

    def smooth_mouse_move(self, target_x, target_y, steps=20, total_time=0.3):
        import win32api
        cur_x, cur_y = win32api.GetCursorPos()
        dx = (target_x - cur_x) / steps
        dy = (target_y - cur_y) / steps
        for i in range(steps):
            new_x = int(cur_x + dx * (i + 1))
            new_y = int(cur_y + dy * (i + 1))
            self.send_queue.put({'type':'mouse_move_absolute', 'x':new_x, 'y':new_y})
            time.sleep(total_time / steps)
        global_log_queue.put(f"Плавное движение мыши к: {target_x}, {target_y}")

    def human_mouse_move_smooth(self, amp_x=200, amp_y=100, steps=20, total_time=0.3):
        rect = self.get_game_window_rect()
        if not rect:
            self.debug_print("Не удалось получить окно для human_mouse_move_smooth")
            return
        left, top, right, bottom = win32gui.GetWindowRect(win32gui.GetForegroundWindow())
        cur_x, cur_y = win32api.GetCursorPos()
        dx = random.randint(-amp_x, amp_x)
        dy = random.randint(-amp_y, amp_y)
        target_x = max(left, min(cur_x + dx, right - 1))
        target_y = max(top, min(cur_y + dy, bottom - 1))
        self.smooth_mouse_move(target_x, target_y, steps=steps, total_time=total_time)

    def aim_at_person(self, yolo_results):
        if not self.hwnd:
            self.debug_print("Ошибка: hwnd не определён")
            return
        if yolo_results and yolo_results[0].boxes.data.tolist():
            frame_height, frame_width = self.height, self.width
            center_x_frame = frame_width / 2
            center_y_frame = frame_height / 2
            closest_person = None
            min_distance = float('inf')
        
            for det in yolo_results[0].boxes.data:
                cls = int(det[5])
                if yolo_results[0].names.get(cls, "unknown").lower() == "person":
                    x1, y1, x2, y2 = det[:4].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    distance = ((center_x - center_x_frame) ** 2 + (center_y - center_y_frame) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_person = (center_x, center_y)
        
            if closest_person:
                center_x, center_y = closest_person
                dx = center_x - center_x_frame
                dy = center_y - center_y_frame

                window_rect = win32gui.GetWindowRect(self.hwnd)
                screen_width, screen_height = window_rect[2] - window_rect[0], window_rect[3] - window_rect[1]
                scale_x = screen_width / frame_width
                scale_y = screen_height / frame_height
                mouse_dx = int(dx * scale_x)
                mouse_dy = int(dy * scale_y)
                self.debug_print(f"Масштабирование: screen=({screen_width}, {screen_height}), mouse_dx={mouse_dx}, mouse_dy={mouse_dy}")
                self.action_queue.put({'keys': [], 'mouse_move': (mouse_dx, mouse_dy), 'mouse_click': 'left', 'duration': 0.05})

    def check_available_cameras(self):
        index = 0
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                break
            self.debug_print(f"Found camera index {index}")
            cap.release()
            index += 1
        self.debug_print("Try changing camera index (OBS virtual camera index).")

    def get_ocr_text(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang='eng+rus')
            return text.strip()
        except Exception as e:
            self.debug_print(f"OCR error:", {e})
            return ""

    def detect_ui(self, frame, ocr_text, yolo_results):
        text_low = (ocr_text or "").lower()
        for w in UI_KEYWORDS + ["пауза", "меню игры"]:
            if w in text_low:
                self.debug_print(f"UI обнаружен через OCR: {w}")
                return True
        if yolo_results:
            names = set()
            try:
                for det in yolo_results[0].boxes.data:
                    cls = int(det[5])
                    cls_name = yolo_results[0].names.get(cls, "") if hasattr(yolo_results[0],'names') else ""
                    names.add(cls_name.lower())
                ui_elements = {'inventory', 'menu', 'ui', 'search', 'pause'}
                if ui_elements.intersection(names):
                    self.debug_print(f"UI обнаружен через YOLO: {names & ui_elements}")
                    return True
            except Exception:
                pass
        return False

    def reset(self, **kwargs):
        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.current_frame = frame.copy()
                self.annotated_frame = frame.copy()
            self.current_state = self.get_state(frame)
            self.prev_frame = frame.copy()
        else:
            self.current_state = np.zeros((84,84,3),dtype=np.uint8)
            self.prev_frame = None
            self.annotated_frame = None
        self.prev_ocr_text = ""
        self.prev_objects = set()
        self.frame_count = 0
        self.frame_processed = 0
        self.release_all_keys()
        return self.current_state, {}

    def get_state(self, frame):
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((84, 84))
        ])
        frame_tensor = transform(frame).to(self.device)
        frame_np = frame_tensor.cpu().numpy().transpose(1, 2, 0) * 255
        return frame_np.astype(np.uint8)

    def step(self, action):
        start_time = time.time()
        new_state = np.zeros((84, 84, 3), dtype=np.uint8)
        reward = 0.0
        description_reward = 0.0
        done = False
        truncated = False
        info = {'ui_mode': self.ui_mode}
        new_state = self.current_state

        if self.pause_event.is_set():
            self.release_all_keys()  # Освобождаем все клавиши при паузе
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()
                new_state = self.get_state(frame)
                ocr_text = self.get_ocr_text(frame)
            else:
                new_state = np.zeros_like(self.current_state)
                ocr_text = ""
            self.ui_mode = self.detect_ui(frame, ocr_text, self.yolo_results)
            self.restrict_cursor_to_window()
            global_log_queue.put("Бот приостановлен")
            return new_state, 0.0, False, False, {'ui_mode': self.ui_mode}

        # Если не на паузе — увеличиваем счетчик
        self.frame_processed += 1

        if self.frame_processed % 2 == 0:
            try:
                img_tensor = torch.tensor(self.current_state.transpose(2,0,1)[None], dtype=torch.float32, device=self.device)
                current_feature = self.feature_net(img_tensor)
                action_onehot = torch.nn.functional.one_hot(torch.tensor(action, device=self.device), num_classes=self.action_space.n).float()
                input_tensor = torch.cat((current_feature.squeeze(0), action_onehot))
                pred_feature = self.forward_model(input_tensor)
                new_img_tensor = torch.tensor(new_state.transpose(2,0,1)[None], dtype=torch.float32, device=self.device)
                new_feature = self.feature_net(new_img_tensor)
                intrinsic = ((pred_feature - new_feature.squeeze(0)) ** 2).mean().item()
                intrinsic = min(intrinsic, 1000.0)
            
                # Обновление градиентов
                loss = ((pred_feature - new_feature.squeeze(0).detach()) ** 2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            except Exception as e:
                self.debug_print(f"Ошибка при вычислении внутренней награды: {e}")
                intrinsic = 0.0
        else:
            return new_state, 0.0, False, False, {}

        self.perform_action(action)

        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.current_frame = frame.copy()
            new_state = self.get_state(frame)
            ocr_text = self.prev_ocr_text if self.frame_processed % 30 != 0 else self.get_ocr_text(frame)
        else:
            new_state = np.zeros_like(self.current_state)
            ocr_text = ""

        try:
            new_img_tensor = torch.tensor(new_state.transpose(2,0,1)[None], dtype=torch.float32, device=self.device)
            new_feature = self.feature_net(new_img_tensor)
            intrinsic = ((pred_feature - new_feature.squeeze(0)) ** 2).mean().item()
            intrinsic = min(intrinsic, 1000.0)
        except Exception:
            intrinsic = 0.0

        yolo_results = None
        try:
            with self.lock:
                yolo_results = self.yolo_results
        except Exception as e:
            self.debug_print(f"Ошибка результатов YOLO: {e}")

        current_objects = set()
        exploration_reward = 0.0
        web_action_reward = 0.0
        enemy_reward = 0.0
        # Динамическое определение релевантных классов
        relevant_classes = ['person', 'item', 'object']  # Базовые классы для универсальности
        if self.is_shooter:
            relevant_classes = ['person', 'weapon', 'bomb']
        elif "minecraft" in self.game_name.lower():
            relevant_classes = ['person', 'animal', 'block', 'item']
        for kw in self.description_keywords:
            if kw in ['collect', 'craft', 'build']:
                relevant_classes.extend(['item', 'block'])
            elif kw in ['attack', 'shoot', 'kill', 'eliminate']:
                relevant_classes.extend(['person', 'weapon'])
            elif kw in ['plant', 'defuse']:
                relevant_classes.append('bomb')
        relevant_classes = list(set(relevant_classes))  # Убираем дубликаты
        self.debug_print(f"Релевантные классы YOLO для {self.game_name}: {relevant_classes}")

        if yolo_results:
            try:
                if not yolo_results[0].boxes.data.tolist():  # Проверяем, пустой ли тензор
                    self.debug_print("Обнаружения YOLO отсутствуют")
                else:
                    frame_height, frame_width = self.height, self.width
                    center_x_frame = frame_width / 2
                    center_y_frame = frame_height / 2
                    center_threshold = 0.1  # ~30 пикселей для 640x480

                    detected_classes = []
                    person_detected = False
                    for det in yolo_results[0].boxes.data:
                        cls = int(det[5])
                        cls_name = yolo_results[0].names.get(cls, "unknown").lower()
                        detected_classes.append(cls_name)
                        if cls_name in relevant_classes:
                            current_objects.add(cls_name)
                            if cls_name == "person" and self.is_shooter:
                                person_detected = True
                                x1, y1, x2, y2 = det[:4].cpu().numpy()
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                is_in_center = (abs(center_x - center_x_frame) < frame_width * center_threshold and
                                                abs(center_y - center_y_frame) < frame_height * center_threshold)
                                self.debug_print(f"Персонаж обнаружен, центр: ({center_x:.1f}, {center_y:.1f}), is_in_center: {is_in_center}")
                                if self.actions[action][0] == "aim_at_person":
                                    self.aim_at_person(yolo_results)
                                    self.action_queue.put({'keys': [], 'mouse_click': 'left', 'duration': 0.05})
                                if self.actions[action][0] in ["click_left", "hold_left"] and is_in_center:
                                    enemy_reward += 10.0
                                    self.debug_print(f"Вознаграждение за стрельбу по противнику в центре: {enemy_reward}")
                                elif self.actions[action][0] == "aim_at_person":
                                    if is_in_center:
                                        enemy_reward += 5.0  # Увеличенная награда за точное наведение
                                        self.debug_print(f"Вознаграждение за наведение на противника в центре: {enemy_reward}")
                                    else:
                                        enemy_reward += 1.0  # Бонус за выбор наведения при обнаружении
                                        self.debug_print(f"Бонус за выбор наведения на противника: {enemy_reward}")
                            elif cls_name in ["weapon", "bomb", "item", "block"] and self.actions[action][0] in ["click_left", "e"]:
                                enemy_reward += 0.5
                                self.debug_print(f"Вознаграждение за взаимодействие с {cls_name}: {enemy_reward}")

                    self.debug_print(f"Обнаруженные классы YOLO: {detected_classes}")

                    # Штраф за случайный выстрел в шутерах
                    if self.is_shooter and self.actions[action][0] in ["click_left", "hold_left"]: 
                        person_in_center = False
                        for det in yolo_results[0].boxes.data:
                            cls = int(det[5])
                            cls_name = yolo_results[0].names.get(cls, "unknown").lower()
                            if cls_name == "person":
                                x1, y1, x2, y2 = det[:4].cpu().numpy()
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                if (abs(center_x - center_x_frame) < frame_width * center_threshold and
                                    abs(center_y - center_y_frame) < frame_height * center_threshold):
                                    person_in_center = True
                                    break
                        if not person_in_center:
                            enemy_reward -= 3.0  # Увеличенный штраф за неточную стрельбу
                            self.debug_print("Штраф за случайный выстрел: -1.5")
                        elif not person_detected:
                            enemy_reward -= 1.5  # Штраф за стрельбу/зажатие, если персонажа вообще нет
                            self.debug_print(f"Штраф за {'случайный выстрел' if self.actions[action][0] == 'click_left' else 'зажатие кнопки'} без персонажа: -1.5")

                new_objects = current_objects - self.prev_objects
                exploration_reward = min(len(new_objects) * 0.5, 2.0)
                for obj in new_objects:
                    actions = self.search_object_info(obj)
                    self.debug_print(f"Результат вызова API для {obj}: {actions}")
                    if actions:
                        web_action_reward += 0.2
                self.prev_objects = current_objects
            except Exception as e:
                self.debug_print(f"Ошибка обработки YOLO: {e}")

        self.ui_mode = self.detect_ui(frame, ocr_text, yolo_results)
        self.restrict_cursor_to_window()
        if self.ui_mode:
            self.release_all_keys()

        change_reward = self.calculate_change_reward(self.prev_frame, frame) if ret else 0.0
        ocr_reward = self.calculate_ocr_reward(self.prev_ocr_text, ocr_text)

        if self.description_keywords:
            action_name = self.actions[action][0]
            if any(kw in self.description_keywords for kw in ["shoot", "kill", "eliminate"]) and action_name == "click_left":
                description_reward += 1.0
                self.debug_print(f"Вознаграждение за действие {action_name} (описание): {description_reward}")
            elif any(kw in self.description_keywords for kw in ["shoot", "kill", "eliminate"]) and action_name == "aim_at_person":
                description_reward += 0.5
                self.debug_print(f"Вознаграждение за действие {action_name} (описание): {description_reward}")
            elif "jump" in self.description_keywords and action_name == "space":
                description_reward += 0.3
            elif "objective" in self.description_keywords and action_name == "e":
                description_reward += 0.4
            elif "plant" in self.description_keywords and action_name == "e":
                description_reward += 0.5
            elif "defuse" in self.description_keywords and action_name == "e":
                description_reward += 0.5
            elif "collect" in self.description_keywords and action_name == "e":
                description_reward += 0.5
            elif "attack" in self.description_keywords and action_name in ["click_left", "click_right"]:
                description_reward += 0.4
            elif "craft" in self.description_keywords and action_name == "e" and self.ui_mode:
                description_reward += 0.6
            self.debug_print(f"Вознаграждение за действие {action_name} (описание): {description_reward}")

        reward = intrinsic * 0.001 + exploration_reward + change_reward + ocr_reward + web_action_reward + enemy_reward
        self.current_reward = reward
        self.total_reward += reward

        try:
            loss = ((pred_feature - new_feature.squeeze(0).detach()) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception:
            pass

        self.prev_frame = frame.copy() if ret else None
        self.prev_ocr_text = ocr_text
        self.current_state = new_state
        self.debug_print(f"Step FPS: {1/(time.time()-start_time+1e-9):.1f}, Reward: {reward:.2f}, Enemy Reward: {enemy_reward:.2f}, Description Reward: {description_reward:.2f}")
        # Логирование архетипа на ~шаге 1000
        if 990 <= self.frame_processed <= 1010 and self.behavior_archetype:
            self.debug_print(f"Архетип поведения на шаге {self.frame_processed}: {json.dumps(self.behavior_archetype, indent=2)}")
        terminated = False  # Установите True, если эпизод завершён по правилам (например, смерть персонажа)
        truncated = False  # Установите True, если завершён по лимиту времени или другой внешней причине
        info = {}
        return new_state, reward, terminated, truncated, info

    def calculate_change_reward(self, prev_frame, new_frame):
        if prev_frame is None:
            return 0.0
        diff = cv2.absdiff(prev_frame, new_frame)
        m = np.mean(diff)
        return 0.05 if m > 20 else -0.005

    def calculate_ocr_reward(self, prev_text, new_text):
        if prev_text != new_text:
            if any(x in new_text.lower() for x in ["log","stone","inventory","поиск"]):
                return 0.6
            return 0.15
        return 0.0

    def perform_action(self, action):
        action_dict = self.action_map[action]
        if self.ui_mode and 'e' not in action_dict.get('keys', []) and not action_dict.get('aim_at_person', False):
            return
        if action_dict.get('aim_at_person', False):
            with self.lock:
                self.aim_at_person(self.yolo_results)
        self.action_queue.put(action_dict)

    def search_object_info(self, obj_name, lang="en"):
        current_time = time.time()
        if obj_name in self.known_objects:
            print(f"Using cached object info for {obj_name}: {self.known_objects[obj_name]}")
            return self.known_objects[obj_name]

        if current_time - self.last_api_call < self.api_call_interval:
            print(f"API call skipped for {obj_name} due to rate limit")
            return []

        self.debug_print(f"Sending Wikipedia request for object: {obj_name}")
        try:
            page = obj_name.replace(" ", "_")
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{page}"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                self.debug_print(f"Wikipedia returned {response.status_code} for {obj_name}")
                return []

            data = response.json()
            content = data.get("extract", "").lower()
            actions = []
            
            if "press e" in content or "interact" in content:
                actions.append("e")
            if "click" in content or "attack" in content or "shoot" in content:
                actions.append("click_left")
            if "right click" in content:
                actions.append("click_right")

            if self.is_shooter and obj_name == "person":
                actions.extend(["click_left", "e", "aim_at_person"])

            actions = list(set(actions)) or ["unknown"]
            self.known_objects[obj_name] = actions or ["unknown"]
            self.save_config()
            self.last_api_call = current_time
            self.debug_print(f"Saved object info for {obj_name}: {actions}")
            return actions
        except Exception as e:
            self.debug_print(f"Search object info error for {obj_name}: {e}")
            return []

    def get_profile_dir(self):
        return f"profiles/{self.game_name}/gen_{self.generation}_agent_{self.profile_id}"

    def load_config(self):
        profile_dir = self.get_profile_dir()
        os.makedirs(profile_dir, exist_ok=True)
        config_path = f"{profile_dir}/config.json"
        default_config = {
            "actions": {str(i): act[0] for i, act in enumerate(self.actions)},
            "rewards": {},
            "known_objects": {},
            "total_reward": 0.0,
            "frame_processed": 0
        }
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        with open(config_path, 'r') as f:
            config = json.load(f)
            config = {**default_config, **config}
            self.known_objects = config.get("known_objects", {})
            self.total_reward = config.get("total_reward", 0.0)
            self.frame_processed = config.get("frame_processed", 0)
            return config
        
    def save_config(self, config=None):
        profile_dir = self.get_profile_dir()
        config_path = f"{profile_dir}/config.json"
        conf = config or{
            "actions": {str(i): act[0] for i, act in enumerate(self.actions)},
            "known_objects": self.known_objects,
            "total_reward": self.total_reward,
            "frame_processed": self.frame_processed
        }
        with open(config_path, 'w') as f:
            json.dump(conf, f, indent=4)


def gui_thread(env: GameEnv, stop_event: threading.Event):
    try:
        root = tk.Tk()
        env.active_gui = root
        root.title("Bot Overlay")
        root.overrideredirect(True)
        root.attributes('-topmost', True)
        root.attributes('-alpha', 0.8)
        root.attributes('-disabled', True)

        window_width = min(480, env.width // 2)
        window_height = int(window_width * (env.height / env.width)) + 120

        def update_position():
            if stop_event.is_set() or global_stop_event.is_set():
                try:
                    root.quit()
                    root.destroy()
                    env.debug_print("GUI закрыт в update_position")
                except tk.TclError:
                    pass
                return
            rect = env.get_game_window_rect()
            if rect:
                game_x, game_y, game_w, game_h = rect
                overlay_x = game_x + 10
                overlay_y = game_y + (game_h - window_height) // 2
                try:
                    root.geometry(f"{window_width}x{window_height}+{overlay_x}+{overlay_y}")
                    root.update()
                except tk.TclError:
                    pass
            else:
                monitor = get_monitors()[0]
                overlay_x, overlay_y = 10, (monitor.height - window_height) // 2
                try:
                    root.geometry(f"{window_width}x{window_height}+{overlay_x}+{overlay_y}")
                    root.update()
                except tk.TclError:
                    pass
            root.after(100, update_position)

        update_position()

        video_label = tk.Label(root, bg='black')
        video_label.pack(fill="both", expand=True)
        status_label = tk.Label(
            root,
            text="Gen: 0 | Agent: 0 | FPS: 0.0\nSteps: 0/20000 | Total Reward: 0.00",
            bg='black', fg='white', font=("Arial", 8)
        )
        status_label.pack(fill="x")
        demo_status_label = tk.Label(root, text="", bg='black', fg='red', font=("Arial", 8))
        demo_status_label.pack(fill="x")
        log_text = tk.Text(root, height=6, bg='black', fg='white', font=("Arial", 8))
        log_text.pack(fill="x")

        frame_count = 0
        start_time = time.time()
        last_update_time = 0
        update_interval = 1 / 20

        def update():
            nonlocal last_update_time, frame_count, start_time
            if stop_event.is_set() or global_stop_event.is_set():
                try:
                    root.quit()
                    root.destroy()
                    env.debug_print("GUI закрыт в update")
                except tk.TclError:
                    pass
                return

            now = time.time()
            if now - last_update_time < update_interval:
                root.after(5, update)  # Проверяем чаще, но обновляем только через интервал
                return
            last_update_time = now

            try:
                # Берём кадр для GUI без блокировки основной обработки
                with env.lock:
                    frame = env.annotated_frame if env.annotated_frame is not None else env.current_frame
                if frame is None:
                    frame = np.zeros((env.height, env.width, 3), dtype=np.uint8)

                # Изменяем размер один раз
                display_w = window_width
                display_h = int(display_w * (env.height / env.width))
                frame_resized = cv2.resize(frame, (display_w, display_h))

                # Конвертация для Tkinter
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)

                if video_label.winfo_exists():
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)

                # Обновление статуса
                steps = env.frame_processed
                reward = env.total_reward
                status_text = f"Gen: {env.generation} | Agent: {env.agent_id} | FPS: {frame_count/(time.time()-start_time+1e-9):.1f}\nSteps: {steps}/20000 | Total Reward: {reward:.2f}"
                if env.pause_event.is_set():
                    status_text += "\nPaused"
                status_label.config(text=status_text)

                # Demo статус
                demo_status_label.config(
                    text="Ведётся видеофиксация" if env.recording_active else 
                         ("Фрагмент игры: отсутствует" if not env.has_demo else "Фрагмент игры найден (нажмите '=')"),
                    fg='yellow' if env.recording_active else ('red' if not env.has_demo else 'green')
                )

                # Логи
                max_logs = 5
                for _ in range(max_logs):
                    if global_log_queue.empty():
                        break
                    log_text.insert(tk.END, global_log_queue.get_nowait() + "\n")
                    log_text.see(tk.END)

            except Exception as e:
                env.debug_print(f"GUI update error: {e}")

            frame_count += 1
            if frame_count % 30 == 0:
                start_time = time.time()
                frame_count = 0

            if not stop_event.is_set() and not global_stop_event.is_set():
                root.after(5, update)

        update()
        root.mainloop()
    except Exception as e:
        env.debug_print(f"Ошибка потока GUI: {e}")
        traceback.print_exc()
    finally:
        try:
            root.quit()
            root.destroy()
            env.active_gui = None
            env.debug_print("GUI закрыт в finally")
        except (tk.TclError, NameError) as e:
            env.debug_print(f"Ошибка при закрытии GUI в finally: {e}")
    

class StopCallback(BaseCallback):
    def __init__(self, stop_event):
        super(StopCallback, self).__init__()
        self.stop_event = stop_event
    def _on_step(self) -> bool:
        return not self.stop_event.is_set()

class SaveCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int, env: GameEnv):
        super(SaveCallback, self).__init__()
        self.save_path = save_path
        self.save_freq = save_freq
        self.env = env

    def _on_step(self) -> bool:
        try:
            # Сохраняем при достижении save_freq или при остановке
            if self.n_calls % self.save_freq == 0 or self.env.stop_event.is_set() or global_stop_event.is_set():
                self.model.save(self.save_path)
                self.env.save_training_state(save_path=os.path.dirname(self.save_path))
                self.env.debug_print(f"Model saved at step {self.n_calls}: {self.save_path}, Total Reward: {self.env.total_reward}, Frames Processed: {self.env.frame_processed}")
        except Exception as e:
            self.env.debug_print(f"Save error at step {self.n_calls}: {e}")
        # Прерываем обучение, если установлен stop_event или global_stop_event
        if self.env.stop_event.is_set() or global_stop_event.is_set():
            self.env.debug_print("SaveCallback: обучение прервано по событию остановки")
            return False
        return True


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
        "num_agents": num_agents
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
    default_generations = 1000
    default_num_agents = 100
    timesteps_per_agent = 20000
    elite_fraction = 0.2
    random_fraction = 0.1
    profile_paths = []
    
    os.makedirs(f"profiles/{game_name}", exist_ok=True)
    print(f"Создана директория profiles/{game_name}")
    
    state = load_global_state(game_name)
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
        save_global_state(game_name, current_gen, current_agent, profile_paths, generations, num_agents)

    os.makedirs(f"profiles/{game_name}/shared_demo", exist_ok=True)

    try:
        while current_gen < generations:
            if global_stop_event.is_set():
                print("Глобальный стоп-событие установлено, выход из основного цикла")
                break  # Прерываем главный цикл полностью
            print(f"Поколение {current_gen + 1}/{generations}")
            current_profile_paths = []
                

            for agent_id in range(current_agent, num_agents):
                profile_dir = f"profiles/{game_name}/gen_{current_gen}_agent_{agent_id}"
                os.makedirs(profile_dir, exist_ok=True)
                model_path = os.path.join(profile_dir, "model.zip")

                env = GameEnv(game_name, profile_id=agent_id, is_final_profile=False, generation=current_gen)
                env.debug_print(f"Создано окружение для gen={current_gen}, agent_id={agent_id}")

                current_profile_paths.append(profile_dir)
                
                if os.path.exists(model_path):
                    try:
                        model = PPO.load(model_path, env=env)
                        config = env.load_config()
                        env.frame_processed = config.get("frame_processed", 0)
                        env.total_reward = config.get("total_reward", 0.0)
                        env.debug_print(f"Загружена существующая модель для агента {agent_id} из {model_path}, обработано кадров: {env.frame_processed}")
                    except Exception as e:
                        env.debug_print(f"Не удалось загрузить модель {model_path}: {e}")
                        model = PPO("CnnPolicy", env, n_steps=4096, verbose=1, batch_size=1024, device='cuda' if torch.cuda.is_available() else 'cpu')
                        env.frame_processed = 0
                        env.total_reward = 0.0
                else:
                    model = PPO("CnnPolicy", env, n_steps=4096, batch_size=1024, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')
                    env.frame_processed = 0
                    env.total_reward = 0.0

                if profile_paths:
                    selected_parent = random.choice(profile_paths)
                    env.evolve_model([selected_parent])
                        
                if env.has_demo and env.demo_access_count < 5:
                    try:
                        env.pretrain_with_demo(model)
                    except Exception as e:
                        env.debug_print(f"Ошибка в pretrain_with_demo для агента {agent_id}: {e}")
                        traceback.print_exc()
                            
                env.model = model
                callbacks = [StopCallback(env.stop_event), SaveCallback(model_path, save_freq=5000, env=env)]
                remaining_timesteps = max(0, timesteps_per_agent - env.frame_processed)
                    
                while remaining_timesteps > 0 and not env.stop_event.is_set() and not global_stop_event.is_set():
                    steps_to_run = min(1000, remaining_timesteps)
                    try:
                        model.learn(total_timesteps=steps_to_run, callback=callbacks)
                    except Exception as e:
                        env.debug_print(f"Ошибка обучения модели на шаге {env.frame_processed}: {e}")
                        traceback.print_exc()
                        break
                            
                    remaining_timesteps -= steps_to_run
                        
                    if env.has_demo and env.demo_access_count < 5 and env.frame_processed % 5000 == 0:
                        try:
                            env.pretrain_with_demo(model)
                        except Exception as e:
                            env.debug_print(f"Ошибка в pretrain_with_demo на шаге {env.frame_processed}: {e}")
                            traceback.print_exc()
                    
                env.save_training_state()
                config = env.load_config()
                config["frame_processed"] = env.frame_processed
                config["total_reward"] = env.total_reward
                env.save_config(config)

                save_global_state(game_name, current_gen, current_agent, profile_paths, generations, num_agents)

                env.cleanup()
                time.sleep(1)

                if global_stop_event.is_set():
                    print("Глобальный стоп-событие установлено, пропуск дальнейших агентов")
                    break

            # Отбор для следующего поколения
            performances = []
            for profile_path in current_profile_paths:
                config_path = os.path.join(profile_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        cfg  = json.load(f)
                        performances.append((profile_path, cfg.get("total_reward", 0.0)))


            if performances:
                performances.sort(key=lambda x: x[1], reverse=True)
                num_elite = int(num_agents * elite_fraction)
                num_random = int(num_agents * random_fraction)
                elite_paths = [p[0] for p in performances[:num_elite]]
                remaining = performances[num_elite:]
                random_paths = [p[0] for p in random.sample(remaining, min(num_random, len(remaining)))] if remaining else []
                profile_paths = elite_paths + random_paths
                print(f"Отобрано {len(profile_paths)} профилей для следующего поколения: {len(elite_paths)} элитных, {len(random_paths)} случайных")

            current_agent = 0
            current_gen += 1
            save_global_state(game_name, current_gen, current_agent, profile_paths, generations, num_agents)

        # В конце выбираем чемпиона
        if current_agent >= num_agents and performances:
            champion_path = performances[0][0]
            champion_reward = performances[0][1]
            print(f"Чемпион найден в поколении {current_gen-1}: {champion_path} с наградой {champion_reward}")
            final_profile_id = f"champion_gen_{current_gen-1}"
            final_profile_dir = f"profiles/{game_name}/gen_{current_gen-1}_agent_{final_profile_id}"
            os.makedirs(final_profile_dir, exist_ok=True)

            final_env = GameEnv(game_name, profile_id=final_profile_id, is_final_profile=True, generation=current_gen-1)
            final_env.evolve_model([champion_path])
            final_env.save_training_state(save_path=final_profile_dir)
            save_global_state(game_name, current_gen-1, final_profile_id, [final_profile_dir], generations, num_agents)
            final_env.cleanup()
            print(f"Чемпион сохранён в {final_profile_dir}")

    except Exception as e:
        print(f"Ошибка в main: {e}")
        traceback.print_exc()
    finally:
        global_stop_event.set()
        if 'env' in locals() and env is not None:
            env.cleanup()
        if 'final_env' in locals() and final_env is not None:
            final_env.cleanup()
        state_path = f"profiles/{game_name}/training_state.json"
        if current_gen >= generations:
            if os.path.exists(state_path):
                os.remove(state_path)
        else:
            save_global_state(game_name, current_gen, current_agent, profile_paths, generations, num_agents)
        print("Программа полностью остановлена")

def evolve_model(self, profile_paths, mutate=True):
    """
    Эволюция модели: выбирает случайную модель из списка и применяет мутации, если mutate=True.
    profile_paths: список путей к профилям для эволюции.
    """
    if not profile_paths:
        self.debug_print("Нет профилей для эволюции")
        return None

    selected_path = random.choice(profile_paths)
    self.debug_print(f"Выбран профиль для эволюции: {selected_path}")

    # Загружаем модель из выбранного профиля
    ppo_files = [f for f in os.listdir(selected_path) if f.endswith(".zip")]
    feature_net_files = [f for f in os.listdir(selected_path) if f.startswith("feature_net_") and f.endswith(".pth")]
    forward_model_files = [f for f in os.listdir(selected_path) if f.startswith("forward_model_") and f.endswith(".pth")]

    if not (ppo_files and feature_net_files and forward_model_files):
        self.debug_print("Не найдены файлы модели в выбранном профиле")
        return None

    latest_ppo = max(ppo_files, key=lambda f: os.path.getmtime(os.path.join(selected_path, f)))
    latest_feature_net = max(feature_net_files, key=lambda f: os.path.getmtime(os.path.join(selected_path, f)))
    latest_forward_model = max(forward_model_files, key=lambda f: os.path.getmtime(os.path.join(selected_path, f)))

    self.load_training_state(
        ppo_path=os.path.join(selected_path, latest_ppo),
        feature_net_path=os.path.join(selected_path, latest_feature_net),
        forward_model_path=os.path.join(selected_path, latest_forward_model)
    )

    if mutate:
        # Применяем мутации к feature_net и forward_model
        with torch.no_grad():
            for param in self.feature_net.parameters():
                noise = torch.randn_like(param) * 0.05  # Небольшая мутация (5% шума)
                param.add_(noise)
            for param in self.forward_model.parameters():
                noise = torch.randn_like(param) * 0.05
                param.add_(noise)

    self.debug_print(f"Эволюция модели завершена: загружен профиль {'с мутацией' if mutate else 'без мутации'}")
    return selected_path

    print("Програма полностью остановлена")

if __name__ == "__main__":
    main()
