# AI Game Bot

## English Version

This bot is a Python-based system that uses computer vision, keyboard/mouse control, and artificial intelligence to interact with games. The bot was written using the AI **Grok** .
Its structure and functionality include:

1. Image capture
- Uses OpenCV to capture frames from the screen.
 - Can work in streaming mode: processes each frame in turn.
2. Frame processing
 - Compresses, converts, and sends images to the queue for analysis.
 - Multithreading is supported so that capture and processing do not interfere with each other.
3. Object recognition (YOLOv8)
 - The YOLOv8 model is used to detect objects on the screen.
 - Determines what is in the frame (e.g., enemies, objects, interface elements).
4. OCR (text recognition)
 - Tesseract is built in for text analysis (e.g., chat, inscriptions, hints).
5. Game control
 - Using pydirectinput and keyboard, the bot can send commands to the game.
 - It simulates keystrokes and mouse movements.
6. Decision-making logic
 - Based on data from YOLO and OCR, the bot selects actions.
 - Can learn with reinforcement: analyzes its actions and improves its strategy.
7. Debugging and monitoring
 - Built-in logger for tracking events.
 - Queuing system prevents overload.

Also, for a correct behavior profile, you need to record a demo for it at the beginning.

After recording a 15-minute video, the bot will learn. Training consists of 100 agents with 20,000 steps per 1,000 generations.
The speed of the bot's learning will depend on the power of the video card.
On average, the bot will be fully trained in ~7.9 years (SPS=40) to decades (SPS=10). So the feasibility is up to you to decide.

The main testing was conducted on **Counter-Strike 2 (CS2)** and **Minecraft**, but the bot is also compatible with other games.

The bot determines which game is running, requests information about it and its objectives from Wikipedia, and begins training based on the keywords found. If no keywords are found, training proceeds by trial and error.

The bot was developed in **four days**, so it is considered incomplete and is provided “as is.”

---

## Requirements
1. Python 3.12.0
2. OBS Studio
3. Tesseract OCR

---

## Installation
1. Uninstall all installed versions of Python and install Python 3.12.0 (the bot conflicts with version 3.13.0):  
   👉 [Download Python 3.12.0](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe)
2. Install OBS Studio:  
   👉 [Download OBS Studio](https://obsproject.com/download)
3. Install Tesseract OCR:  
   👉 [Download Tesseract](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
4. Install the necessary Python libraries using the command:
```bash
   pip install numpy opencv-python pillow torch ultralytics pytesseract requests gymnasium stable-baselines3 screeninfo pywin32 keyboard
   ```
   Or, if you have a `requirements.txt` file:
```bash
   pip install -r requirements.txt
   ```
5. Restart your computer for Tesseract to work correctly, and check the path:
6. Press `Win + R`, type `sysdm.cpl`, and press Enter.
7. Go to the **Advanced** tab.
8. Click **Environment Variables**.
9. In the **System Variables** section, find the `PATH` variable and make sure it contains `C:\Program Files\Tesseract-OCR`. If the path is missing, add it manually.

## Launch
1. Edit the bot, line 51 and below:
- OBS_CAMERA_INDEX = 2 #OBS virtual camera number (0-3)
- YOLO_MODEL_PATH = “yolov8m.pt” #AI model (8n, 8s, 8m, 8l, 8x)
- “yolov8n.pt” - nano (Minimal model, very fast)
-	“yolov8s.pt” - small (higher accuracy, but still fast)
- “yolov8m.pt” - medium (Balance between speed and accuracy. Well suited for medium GPUs (RTX 2060–3060).)
- “yolov8l.pt” - large (More accurate, but requires more resources.)
- “yolov8x.pt” - extra-large (The most powerful and accurate model. High GPU requirements (preferably RTX 3090/4090 and similar))
- ENABLE_GUI = True #Overlay
- DEMO_RECORD_DURATION = 900  # Demo recording time in seconds
- DEMO_VIDEO_FPS = 30  # FPS for demo recording
- MAX_PROFILES = 10  # Maximum number of profiles, minimum 500 for 3D games, 200 for 2D
- TIMESTEPS_PER_AGENT = 120 #Number of steps for one agent, minimum 15,000 for 3D games, 10,000 for 2D games
- MAX_GENS = 10 #Number of generations, minimum 80 for 3D games, 40 for 2D games
- BIND_PAUSE = “f1” #pause button
- BIND_STOP = “f3” #program exit button
- BIND_RECORD = “f2” #demo recording button

2. Launch OBS Studio, find the “Launch Virtual Camera” button, click on the gear next to it, and configure:  
   - **Output Type**: Source  
   - **Select Output**: Screen Capture
3. Launch the game that the bot should play.
4. Run `Start.bat`. The bot will download YOLOv8
5. When the bot turns on the overlay, press F2 to the demo recording button and play yourself. After a certain amount of time, the bot will stop recording.
6. To start training each agent, press the F1.
7. To close application press F3
---

## Russian Version

Этот бот представляет собой систему на Python, которая использует компьютерное зрение, управление клавиатурой/мышью и искусственный интеллект для взаимодействия с играми. Бот написан с помощью ИИ **Grok** 
Его структура и функциональность включают:

1. Захват изображения
 - Использует OpenCV для получения кадров с экрана.
 - Может работать в режиме потока: обрабатывает каждый кадр по очереди.
2. Обработка кадров
 - Сжимает, преобразует и отправляет изображения в очередь для анализа.
 - Поддерживается многопоточность, чтобы захват и обработка не мешали друг другу.
3. Распознавание объектов (YOLOv8)
 - Модель YOLOv8 используется для детекции объектов на экране.
 - Определяет, что находится на кадре (например, враги, предметы, интерфейсные элементы).
4. OCR (распознавание текста)
 - Встроен Tesseract для анализа текста (например, чат, надписи, подсказки).
5. Управление игрой
 - С помощью pydirectinput и keyboard бот может отправлять команды в игру.
 - Имитирует нажатия клавиш и движение мыши.
6. Логика принятия решений
 - На основе данных от YOLO и OCR бот выбирает действия.
 - Может обучаться с подкреплением: анализирует свои действия и улучшает стратегию.
7. Отладка и мониторинг
 - Встроенный логгер для отслеживания событий.
 - Система очередей предотвращает перегрузку.

Также, для коректного профиля поведения, в начале требуется записать для него Демо.

После записи 15 минутного видео, бот будет обучаться. Обучение составляет 100 агентов по 20 000 шагов на 1000 поколений.
От мощности видеокарты будет зависеть скорость обучения бота.
В среднем, бот полностью обучится за ~7.9 лет (SPS=40) до десятков лет (SPS=10). Так что целесообразность - вывод за вами

Основное тестирование проводилось на **Counter-Strike 2 (CS2)** и **Minecraft**, но бот также совместим с другими играми.

Бот определяет, какая игра запущена, запрашивает информацию о ней и ее целях в Wikipedia и начинает обучение на основе найденных ключевых слов. Если ключевые слова не найдены, обучение происходит методом проб и ошибок.

Бот был разработан за **четыре дня**, поэтому считается незавершенным и предоставляется "как есть".

---

## Требования
1. Python 3.12.0
2. OBS Studio
3. Tesseract OCR

---

## Установка
1. Удалите все установленные версии Python и установите Python 3.12.0 (с версией 3.13.0 - бот конфликтует):  
   👉 [Скачать Python 3.12.0](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe)
2. Установите OBS Studio:  
   👉 [Скачать OBS Studio](https://obsproject.com/download)
3. Установите Tesseract OCR:  
   👉 [Скачать Tesseract](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
4. Установите необходимые библиотеки Python с помощью команды:  
   ```bash
   pip install numpy opencv-python pillow torch ultralytics pytesseract requests gymnasium stable-baselines3 screeninfo pywin32 keyboard
   ```
   Или, если у вас есть файл `requirements.txt`:  
   ```bash
   pip install -r requirements.txt
   ```
5. Перезагрузите компьютер, чтобы Tesseract корректно работал, и проверьте путь:
6. Нажмите `Win + R`, введите `sysdm.cpl` и нажмите Enter.
7. Перейдите на вкладку **Дополнительно**.
8. Нажмите **Переменные среды**.
9. В разделе **Системные переменные** найдите переменную `PATH` и убедитесь, что в ней указан `C:\Program Files\Tesseract-OCR`. Если путь отсутствует, добавьте его вручную.

## Запуск
1. Отредактируйте бота, строка 51 и ниже:
 - OBS_CAMERA_INDEX = 2 #Номер виртуальной камеры OBS (0-3)
 - YOLO_MODEL_PATH = "yolov8m.pt" #Модель ИИ (8n, 8s, 8m, 8l, 8х)
- "yolov8n.pt" - nano (Минимальная модель, очень быстрая)
- "yolov8s.pt" - small (выше точность, но всё ещё быстрый)
- "yolov8m.pt" - medium (Баланс между скоростью и точностью. Хорошо подходит для средних GPU (RTX 2060–3060).)
- "yolov8l.pt" - large (Более точная, но требует больше ресурсов.)
- "yolov8x.pt" - extra-large (Самая мощная и точная модель. Высокие требования к GPU (лучше RTX 3090/4090 и аналогичные))
- ENABLE_GUI = True #Оверлей
- DEMO_RECORD_DURATION = 900  # Время записи демки в секундах
- DEMO_VIDEO_FPS = 30  # FPS для записи демки 
- MAX_PROFILES = 10  # максимальное количество профилей, для 3D игры минимум 500, для 2D - 200
- TIMESTEPS_PER_AGENT = 120 #Кол-во шагов для одного агента, для 3D игры минимум 15 000, для 2D - 10 000
- MAX_GENS = 10 #Кол-во поколений, для 3D игры минимум 80, для 2D - 40
- BIND_PAUSE = "f1" #кнопка паузы
- BIND_STOP = "f3" #кнопка завершения программы
- BIND_RECORD = "f2" #кнопка записи демо
2. Запустите OBS Studio, найдите кнопку «Запуск виртуальной камеры», нажмите на шестеренку рядом и настройте:  
   - **Тип вывода**: Источник  
   - **Выбрать вывод**: Захват экрана
3. Запустите игру, в которую должен играть бот.
4. Запустите `Start.bat`. Бот скачает YOLOv8
5. Когда бот включит оверлей, нажмите кнопку F2 для записи демо и поиграйте сами. Через н-ное время бот остановит запись.
6. Для начала обучения каждого агента - нажмите кнопку F1.
7. Чтоб закрыть приложение, нажмите F3

---

## Technologies

- **YOLOv8**: Computer vision
- **Stable Baselines3**: Reinforcement learning
- **Gymnasium**: Reinforcement learning environment
- **OBS Studio**: Virtual camera
- **Tesseract OCR**: Text recognition

---

## License

MIT — Free to use and modify.
