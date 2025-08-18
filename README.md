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

After recording a 30-minute video, the bot will learn. Training consists of 100 agents with 20,000 steps per 1,000 generations.
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
1. Launch OBS Studio, find the “Launch Virtual Camera” button, click on the gear next to it, and configure:
- **Output Type**: Source
- **Select Output**: Screen Capture
2. Launch the game that the bot should play.
3. Run `Start.bat`. The bot will download YOLOv8
4. When the bot turns on the overlay, press F12 and play yourself. After half an hour, the bot will stop recording and start training.

### Bot control
- **F12** : Start/stop recording.
- **=**: Pause the bot.
- **Caps Lock**: Stop the bot and save the training progress.

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

После записи 30 минутного видео, бот будет обучаться. Обучение составляет 100 агентов по 20 000 шагов на 1000 поколений.
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
1. Запустите OBS Studio, найдите кнопку «Запуск виртуальной камеры», нажмите на шестеренку рядом и настройте:  
   - **Тип вывода**: Источник  
   - **Выбрать вывод**: Захват экрана
2. Запустите игру, в которую должен играть бот.
3. Запустите `Start.bat`. Бот скачает YOLOv8
4. Когда бот включит оверлей, нажмите F12 и поиграйте сами. Через пол часа бот остановит запись и начнет обучение.

### Управление ботом
- **F12** : Начать/остановить запись.
- **=**: Приостановить бота.
- **Caps Lock**: Остановить бота и сохранить прогресс обучения.

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
