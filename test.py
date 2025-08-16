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


import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

import torch
import torch.nn as nn

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

global_log_queue = queue.Queue(maxsize=100)

# Конфигурация

OBS_CAMERA_INDEX = Insert your OBS VIrtual Camera
YOLO_MODEL_PATH = "yolov8l.pt"
TAVILY_API_KEY = "Insert your Tavily API"
ENABLE_GUI = True
DEFAULT_FRAME_RESIZE = (640, 480)

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
    # attempt single char
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

# Основной класс среды

class GameEnv(gym.Env):
    def __init__(self, game_name=""):
        super(GameEnv, self).__init__()
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=20)
        self.action_queue = queue.Queue()
        self.send_queue = queue.Queue()
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
        self.api_call_interval = 2.0
        self.ui_mode = False
        self.current_frame = None
        self.game_name = game_name or self.detect_game_window_name()
        self.cap = cv2.VideoCapture(OBS_CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Не удалось открыть виртуальную камеру OBS. Попробуй другой индекс.")
            self.check_available_cameras()
            raise Exception("Camera open failed")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or DEFAULT_FRAME_RESIZE[0]
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or DEFAULT_FRAME_RESIZE[1]
        self.debug_print(f"Camera: {self.width}x{self.height}")
        self.model_yolo = YOLO(YOLO_MODEL_PATH)
        self.debug_print(f"YOLO device: {self.model_yolo.device}")
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,3), dtype=np.uint8)
        self.actions = [
            ("wait", []),
            ("w", ["w"]), ("a", ["a"]), ("s", ["s"]), ("d", ["d"]),
            ("space", ["space"]), ("shift", ["shift"]), ("e", ["e"]), ("ctrl", ["ctrl"]),
            ("click_left", ["click_left"]), ("click_right", ["click_right"]),
            ("camera_left", ["camera_left"]), ("camera_right", ["camera_right"]),
            ("camera_up", ["camera_up"]), ("camera_down", ["camera_down"])
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
            {'keys': [], 'mouse_move': (-500,0), 'mouse_click': None, 'duration': 0.12},
            {'keys': [], 'mouse_move': (500,0), 'mouse_click': None, 'duration': 0.12},
            {'keys': [], 'mouse_move': (0,-300), 'mouse_click': None, 'duration': 0.12},
            {'keys': [], 'mouse_move': (0,300), 'mouse_click': None, 'duration': 0.12},
        ]
        config = self.load_config()
        self.game_description = config.get("game_description", "")
        if not self.game_description:
            self.debug_print("Получаю описание игры в интернете...")
            try:
                self.fetch_game_description()
                self.debug_print(f"Описание игры:, {self.game_description}")
            except Exception as e:
                self.debug_print(f"Ошибка при получении описания:, {e}")
        self.description_keywords = self.analyze_game_description()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_print(f"Device:, {self.device}")
        self.feature_net = FeatureNet().to(self.device).float()
        self.forward_model = nn.Sequential(nn.Linear(512 + len(self.actions), 512), nn.ReLU(), nn.Linear(512,512)).to(self.device).float()
        self.optimizer = torch.optim.Adam(list(self.feature_net.parameters()) + list(self.forward_model.parameters()), lr=1e-3)
        self.model = None
        self.send_worker_thread = threading.Thread(target=self._send_worker, daemon=True)
        self.send_worker_thread.start()
        self.action_worker_thread = threading.Thread(target=self._action_worker, daemon=True)
        self.action_worker_thread.start()
        self.frame_capture_thread = threading.Thread(target=self.frame_capture_worker, daemon=True)
        self.frame_capture_thread.start()
        self.yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self.yolo_thread.start()
        if ENABLE_GUI:
            threading.Thread(target=gui_thread, args=(self, self.stop_event), daemon=True).start()
        threading.Thread(target=self.console_listener, args=(self.stop_event, self.pause_event), daemon=True).start()
        self.restrict_cursor_to_window()
        self.actions.append(("human_mouse_move_smooth", ["human_mouse_move_smooth"]))
        self.action_map.append({'keys': [], 'mouse_move': (0,0), 'mouse_click': None, 'duration': 0.3, 'human_mouse_smooth': True})

    def debug_print(self, message):
        if not self.pause_event.is_set():
            print(message)

    def save_training_state(self, save_path=None):
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path or f"profiles/{self.game_name}"
        os.makedirs(save_path, exist_ok=True)
        try:
            if hasattr(self, 'model') and self.model is not None:
                ppo_save_path = os.path.join(save_path, f"ppo_model_{timestamp}.zip")
                self.model.save(ppo_save_path)
                self.debug_print(f"PPO модель сохранена в {ppo_save_path}")
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

    def console_listener(self, stop_event, pause_event):
        import keyboard
        print("Консольный слушатель запущен. Нажмите '=' для паузы/возобновления, 'Caps Lock' для остановки и сохранения")
        last_pressed = None
        while not stop_event.is_set():
            try:
                event = keyboard.read_event(suppress=False)
                if event.event_type == keyboard.KEY_DOWN:
                    if event.name == '=' and last_pressed != '=':
                        if pause_event.is_set():
                            pause_event.clear()
                            msg = "Bot возобновил работу"
                        else:
                            pause_event.set()
                            msg = "Bot приостановлен."
                        print(msg)
                        global_log_queue.put(msg)
                        last_pressed = '='
                    elif event.name == 'caps lock' and last_pressed != 'caps lock':
                        print("Обнаружено 'Caps Lock' key, сохранение состояния и остановка бота...")
                        if hasattr(self, 'save_training_state'):
                            self.save_training_state()
                        stop_event.set()
                        break
                elif event.event_type == keyboard.KEY_UP:
                    if event.name in ('=', 'caps lock'):
                        last_pressed = None  # Сбрасываем, когда клавиша отпущена
                    time.sleep(0.01)
            except Exception as e:
                self.debug_print(f"ошибка консольного слушателя: {e}")
        print("Консольный слушатель остановлен")

    def detect_game_window_name(self):
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        return title if title else "UnknownGame"

    def set_game_description(self, desc):
        self.game_description = desc
        config = self.load_config()
        config["game_description"] = desc
        self.save_config(config)

    def fetch_game_description(self):
        url = "https://api.tavily.com/search"
        query = f"What is the main objective and how to win in {self.game_name}?"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 1
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        desc = data.get("results", [{}])[0].get("content", "")
        self.set_game_description(desc)
        return desc

    def analyze_game_description(self):
        if not self.game_description:
            return []
        description = self.game_description.lower()
        keywords = ["collect", "build", "craft", "survive", "attack", "jump", "interact", "kill", "move", "delivery"]
        if "minecraft" in self.game_name.lower():
            keywords.extend(["ender dragon", "end portal", "craft", "ender pearl", "blaze rod"])
        if "counter-strike" in self.game_name.lower():
            keywords = ["shoot", "kill", "eliminate", "plant", "defuse", "objective", "aim", "reload"]
        found_keywords = [kw for kw in keywords if kw in description]
        self.debug_print(f"Извлечённые ключевые слова из описания: {found_keywords}")
        return found_keywords

    def get_game_window_rect(self):
        hwnd = win32gui.FindWindow(None, self.game_name)
        if hwnd:
            rect = win32gui.GetWindowRect(hwnd)
            self.debug_print(f"Окно игры найдено: {rect}")
            return rect
        self.debug_print(f"Окно игры '{self.game_name}' не найдено.")
        return None

    def restrict_cursor_to_window(self):
        if self.ui_mode:
            rect = self.get_game_window_rect()
            if rect:
                win32api.ClipCursor(rect)
                self.debug_print(f"Курсор ограничен в окне (UI-режим): {rect}")
            else:
                self.debug_print("Не удалось ограничить курсор: окно не найдено.")
        else:
            ctypes.windll.user32.ClipCursor(None)
            self.debug_print("Курсор освобождён (игровой режим).")
        
    # YOLO worker

    def _yolo_worker(self):
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
                frame_small = cv2.resize(frame, (320, 240))
                results = self.model_yolo(frame_small, conf=0.25, verbose=False)
                with self.lock:
                    self.yolo_results = results
                    self.annotated_frame = results[0].plot() if results else frame.copy()
                    self.annotated_frame = cv2.resize(self.annotated_frame, (self.width, self.height))
                time.sleep(0.05)
            except Exception as e:
                self.debug_print(f"YOLO worker error:", {e})
                traceback.print_exc()
                time.sleep(0.1)

    # Send worker

    def _send_worker(self):
        while not self.stop_event.is_set():
            try:
                if self.pause_event.is_set():
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
                            self.debug_print(f"Отправлено нажатие клавиши: {key_name}")
                elif typ == 'release_keys':
                    vks = item.get('vks', [])
                    inputs = [make_key_input(vk, down=False) for vk in vks]
                    if inputs:
                        send_input(inputs)
                        for vk in vks:
                            global_log_queue.put(f"Отпущена клавиша: {vk}")
                            self.debug_print(f"Отправлено отпускание клавиши: {vk}")
                elif typ == 'mouse_move':
                    dx = item.get('dx', 0)
                    dy = item.get('dy', 0)
                    if dx != 0 or dy != 0:
                        send_input([make_mouse_move(dx, dy)])
                        global_log_queue.put(f"Движение мыши: dx={dx}, dy={dy}")
                        self.debug_print(f"Отправлено движение мыши: dx={dx}, dy={dy}")
                elif typ == 'mouse_move_absolute':
                    x = item.get('x')
                    y = item.get('y')
                    if rect and x is not None and y is not None:
                        new_x = max(rect[0], min(rect[2] - 1, x))
                        new_y = max(rect[1], min(rect[3] - 1, y))
                        send_input([set_mouse_position(new_x, new_y)])
                        global_log_queue.put(f"Абсолютное движение мыши: x={new_x}, y={new_y}")
                        self.debug_print(f"Отправлено абсолютное движение мыши: x={new_x}, y={new_y}")
                    else:
                        send_input([set_mouse_position(x, y)])
                        global_log_queue.put(f"Абсолютное движение мыши: x={x}, y={y}")
                        self.debug_print(f"Отправлено абсолютное движение мыши: x={x}, y={y}")
                elif typ == 'mouse_click':
                    btn = item.get('button', 'left')
                    down = item.get('down', True)
                    send_input([make_mouse_click(btn, down)])
                    click_type = "нажатие" if down else "отпускание"
                    global_log_queue.put(f"{click_type} кнопки мыши ({btn})")
                    self.debug_print(f"Отправлено {click_type} кнопки мыши: {btn}")
                time.sleep(0.001)
            except queue.Empty:
                continue
            except Exception as e:
                self.debug_print(f"Send worker error:", {e})

    # Action worker

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

    # Frame capture worker

    def frame_capture_worker(self):
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()
            time.sleep(0.03)

    # Утилиты: press/hold/release

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
        with self.held_keys_lock:
            for k in list(self.held_keys):
                vk = key_to_vk(k)
                if vk:
                    self.send_queue.put({'type':'release_keys','vks':[vk]})
                try:
                    self.held_keys.remove(k)
                except:
                    pass
        for btn in list(self.held_mouse):
            try:
                if btn.lower() == 'left':
                    self.send_queue.put({'type':'mouse_click','button':'left','down':False})
                else:
                    self.send_queue.put({'type':'mouse_click','button':'right','down':False})
            except:
                pass
        self.held_mouse.clear()

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

    # Camera / OCR / YOLO helpers

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
        self.debug_print("UI не обнаружен")
        return False

    # RL env methods

    def reset(self, **kwargs):
        for _ in range(2):
            self.cap.grab()
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
        frame_small = cv2.resize(frame, (84,84))
        return frame_small

    def step(self, action):
        start_time = time.time()
        self.frame_processed += 1
        reward = 0.0
        enemy_reward = 0.0
        description_reward = 0.0
        done = False
        truncated = False
        info = {'ui_mode': self.ui_mode}
        new_state = self.current_state

        if self.pause_event.is_set():
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

        try:
            img_tensor = torch.tensor(self.current_state.transpose(2,0,1)[None], dtype=torch.float32, device=self.device)
            current_feature = self.feature_net(img_tensor)
            action_onehot = torch.nn.functional.one_hot(torch.tensor(action, device=self.device), num_classes=self.action_space.n).float()
            input_tensor = torch.cat((current_feature.squeeze(0), action_onehot))
            pred_feature = self.forward_model(input_tensor)
        except Exception as e:
            self.debug_print(f"Feature net error: {e}")
            pred_feature = torch.zeros(512, device=self.device)

        self.perform_action(action)

        for _ in range(2):
            self.cap.grab()
        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.current_frame = frame.copy()
            new_state = self.get_state(frame)
            ocr_text = self.get_ocr_text(frame)
        else:
            new_state = np.zeros_like(self.current_state)
            ocr_text = ""

        try:
            new_img_tensor = torch.tensor(new_state.transpose(2,0,1)[None], dtype=torch.float32, device=self.device)
            new_feature = self.feature_net(new_img_tensor)
            intrinsic = min(intrinsic, 1000.0)
        except Exception:
            intrinsic = 0.0

        yolo_results = None
        try:
            with self.lock:
                yolo_results = self.yolo_results
        except Exception as e:
            self.debug_print(f"YOLO results error:", {e})

        current_objects = set()
        exploration_reward = 0.0
        web_action_reward = 0.0
        enemy_reward = 0.0
        relevant_classes = ['person', 'animal', 'block', 'item']
        if "counter-strike" in self.game_name.lower():
            relevant_classes = ['person', 'weapon', 'bomb']
        elif "minecraft" in self.game_name.lower():
            relevant_classes = ['person', 'animal', 'block', 'item']
        for kw in self.description_keywords:
            if kw in ['collect', 'craft', 'build']:
                relevant_classes.extend(['item', 'block'])
            elif kw in ['attack', 'shoot', 'kill', 'eliminate']:
                relevant_classes.append('person')
            elif kw in ['plant', 'defuse']:
                relevant_classes.append('bomb')
        relevant_classes = list(set(relevant_classes))
        self.debug_print(f"Релевантные классы YOLO для {self.game_name}: {relevant_classes}")
    
        if yolo_results:
            try:
                for det in yolo_results[0].boxes.data:
                    cls = int(det[5])
                    cls_name = yolo_results[0].names.get(cls, "unknown").lower()
                    if cls_name in relevant_classes:
                        current_objects.add(cls_name)
                        if cls_name == "person" and self.actions[action][0] == "click_left" and "counter-strike" in self.game_name.lower():
                            enemy_reward += 1.0
                            self.debug_print(f"Вознаграждение за стрельбу по противнику: {enemy_reward}")
                        if cls_name in ["animal", "block", "item"] and self.actions[action][0] in ["click_left", "e"]:
                            enemy_reward += 0.5
                            self.debug_print(f"Вознаграждение за взаимодействие с {cls_name}: {enemy_reward}")
                new_objects = current_objects - self.prev_objects
                exploration_reward = min(len(new_objects) * 0.5, 2.0)
                self.debug_print(f"New objects detected: {new_objects}")
                for obj in new_objects:
                    actions = self.search_object_info(obj)
                    self.debug_print(f"API call result for {obj}: {actions}")
                    if actions:
                        web_action_reward += 0.2
                self.prev_objects = current_objects
            except Exception as e:
                self.debug_print(f"YOLO processing error:", {e})

        self.ui_mode = self.detect_ui(frame, ocr_text, yolo_results)
        self.restrict_cursor_to_window()
        if self.ui_mode:
            self.release_all_keys()

        change_reward = self.calculate_change_reward(self.prev_frame, frame) if ret else 0.0
        ocr_reward = self.calculate_ocr_reward(self.prev_ocr_text, ocr_text)
    
        if self.description_keywords:
            action_name = self.actions[action][0]
            if "counter-strike" in self.game_name.lower():
                if any(kw in self.description_keywords for kw in ["shoot", "kill", "eliminate"]) and action_name == "click_left":
                    description_reward += 1.0
                    self.debug_print(f"Вознаграждение за действие {action_name} (описание): {description_reward}")
                elif "jump" in self.description_keywords and action_name == "space":
                    description_reward += 0.3
                elif "objective" in self.description_keywords and action_name == "e":
                    description_reward += 0.4
                elif "plant" in self.description_keywords and action_name == "e":
                    description_reward += 0.5
                elif "defuse" in self.description_keywords and action_name == "e":
                    description_reward += 0.5
            else:
                if "collect" in self.description_keywords and action_name == "e":
                    description_reward += 0.5
                elif "jump" in self.description_keywords and action_name == "space":
                    description_reward += 0.3
                elif "attack" in self.description_keywords and action_name in ["click_left", "click_right"]:
                    description_reward += 0.4
                elif "craft" in self.description_keywords and action_name == "e" and self.ui_mode:
                    description_reward += 0.6
                self.debug_print(f"Вознаграждение за действие {action_name} (описание): {description_reward}")

        reward = intrinsic * 0.001 + exploration_reward + change_reward + ocr_reward + web_action_reward + enemy_reward
        self.debug_print(f"Reward components: intrinsic={intrinsic*0.1:.2f}, exploration={exploration_reward:.2f}, change={change_reward:.2f}, ocr={ocr_reward:.2f}, web={web_action_reward:.2f}, enemy={enemy_reward:.2f}")

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
        return new_state, reward, done, truncated, {'ui_mode': self.ui_mode}

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
        if self.ui_mode:
            if 'e' in action_dict.get('keys', []):
                pass
            else:
                return
        self.action_queue.put(action_dict)

    def search_object_info(self, obj_name):
        current_time = time.time()
        #if obj_name in self.known_objects:
         #   print(f"Using cached object info for {obj_name}: {self.known_objects[obj_name]}")
          #  return self.known_objects[obj_name]
        if current_time - self.last_api_call < self.api_call_interval:
            print(f"API call skipped for {obj_name} due to rate limit")
            return []
        self.debug_print(f"Sending API request for object: {obj_name}")
        try:
            url = "https://api.tavily.com/search"
            query = f"What is a {obj_name} in {self.game_name} ? What am I need to did?  Actions like shoot, attack, or press keys"
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            self.debug_print(f"API response for {obj_name}: {data}")
            actions = []
            for result in data.get("results", []):
                content = result.get("content", "").lower()
                if "press" in content or "click" in content or "interact" in content:
                    if "e" in content:
                        actions.append("e")
                    if "click" in content:
                        actions.append("click_left")
                    if "right click" in content:
                        actions.append("click_right")
            if "counter-strike" in self.game_name.lower() and obj_name == "person":
                actions.extend(["click_left", "e"])
            actions = list(set(actions))
            self.known_objects[obj_name] = actions
            self.save_config()
            self.last_api_call = current_time
            self.debug_print(f"Saved object info for {obj_name}: {actions}")
            return actions
        except Exception as e:
            self.debug_print(f"Search object info error for {obj_name}: {e}")
            return []

    def load_config(self):
        safe_name = re.sub(r'[:\\/*?"<>|]', "_", self.game_name)
        profile_dir = f"profiles/{self.game_name}"
        os.makedirs(profile_dir, exist_ok=True)
        config_path = f"{profile_dir}/config.json"
        default_config = {
            "actions": {str(i): act[0] for i, act in enumerate(self.actions)},
            "rewards": {},
            "known_objects": {}
        }
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        with open(config_path, 'r') as f:
            config = json.load(f)
            config = {**default_config, **config}
            self.known_objects = config.get("known_objects", {})
            return config
        
    def save_config(self, config=None):
        profile_dir = f"profiles/{self.game_name}"
        config_path = f"{profile_dir}/config.json"
        conf = {
            "actions": {str(i): act[0] for i, act in enumerate(self.actions)},
            "known_objects": self.known_objects
        }
        with open(config_path, 'w') as f:
            json.dump(conf, f, indent=4)

# GUI

def gui_thread(env: GameEnv, stop_event: threading.Event):
    try:
        root = tk.Tk()
        root.title("Bot View")
        monitors = get_monitors()
        monitor = monitors[0] if monitors else None
        window_width = min(960, env.width)
        window_height = int(window_width * (env.height / env.width)) + 120
        geom = f"{window_width}x{window_height}"
        if monitor:
            geom += f"+{monitor.x}+{monitor.y}"
        root.geometry(geom)
        video_label = tk.Label(root)
        video_label.pack(fill="both", expand=True)
        info_label = tk.Label(root, text="FPS: 0.0 | Held: [] | UI: False")
        info_label.pack()
        log_text = tk.Text(root, height=6)
        log_text.pack(fill="x")
        frame_count = 0
        start_time = time.time()
        def update():
            nonlocal frame_count, start_time
            if stop_event.is_set():
                try:
                    root.destroy()
                except:
                    pass
                return
            if env.pause_event.is_set():
                info_label.config(text="Bot paused")
                root.after(100, update)
                return
            try:
                with env.lock:
                    frame = env.annotated_frame if env.annotated_frame is not None else env.current_frame
                if frame is None:
                    frame = np.zeros((env.height, env.width, 3), dtype=np.uint8)
                display_w = window_width // 2
                display_h = int(display_w * (env.height / env.width))
                frame_resized = cv2.resize(frame, (display_w, display_h))
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                if not video_label.winfo_exists():
                    return
                try:
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                except tk.TclError:
                    return
                with env.held_keys_lock:
                    held = [env.actions[i][0] if isinstance(i, int) else str(i) for i in env.held_keys]
                info_label.config(text=f"FPS: {frame_count/(time.time()-start_time+1e-9):.1f} | Held: {held} | UI: {env.ui_mode}")
                while not global_log_queue.empty() and not env.pause_event.is_set():
                    log_text.insert(tk.END, global_log_queue.get_nowait() + "\n")
                    log_text.see(tk.END)
            except Exception as e:
                print("GUI update error:", e)
                import traceback; traceback.print_exc()
            frame_count += 1
            if frame_count % 30 == 0:
                start_time = time.time()
                frame_count = 0
            root.after(50, update)  # 20 FPS

        update()
        root.mainloop()
    except Exception as e:
        print("GUI thread crashed:", e)
        traceback.print_exc()
        time.sleep(3)
        if not stop_event.is_set():
            gui_thread(env, stop_event)

# Callbacks for RL

class StopCallback(BaseCallback):
    def __init__(self, stop_event):
        super(StopCallback, self).__init__()
        self.stop_event = stop_event
    def _on_step(self) -> bool:
        return not self.stop_event.is_set()

class SaveCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int):
        super(SaveCallback, self).__init__()
        self.save_path = save_path
        self.save_freq = save_freq
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            try:
                self.model.save(self.save_path)
                self.debug_print("Model saved:", self.save_path)
            except Exception as e:
                self.debug_print("Save error:", e)
        return True

# Main

def get_game_name():
    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd)
    print(f"Detected game window:", {title})
    return title if title else "UnknownGame"

def main():
    game_name = get_game_name()
    env = GameEnv(game_name)
    profile_dir = f"profiles/{game_name}"
    os.makedirs(profile_dir, exist_ok=True)
    model_path = f"{profile_dir}/model"

    try:
        model = PPO.load(model_path, env=env)
        env.model = model
        env.debug_print(f"Loaded existing model:, {model_path}")
    except Exception as e:
        env.debug_print(f"Model load error, creating new PPO:, {e}")
        model = PPO("CnnPolicy", env, verbose=1)
        env.model = model

    callbacks = [StopCallback(env.stop_event), SaveCallback(model_path, save_freq=10000)]
    try:
        model.learn(total_timesteps=1000000, callback=callbacks)
    except KeyboardInterrupt:
        env.debug_print("KeyboardInterrupt - stopping")
    except Exception as e:
        env.debug_print("Training error:, {e}")

    try:
        env.model.save(model_path)
        env.debug_print("Model saved at end:, {model_path}")
    except Exception as e:
        env.debug_print("Final save error:, {e}")

    env.release_all_keys()
    try:
        env.cap.release()
    except:
        pass
    env.stop_event.set()
    print("Exiting.")

if __name__ == "__main__":
    main()