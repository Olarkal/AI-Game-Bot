# AI Game Bot

## English Version

The AI Game Bot was created using **ChatGPT** and **Grok**. It is designed for neural network-based learning to play video games. The bot utilizes **YOLOv8** for computer vision and captures game footage via the **OBS virtual camera**.

It was primarily tested on **Counter-Strike 2 (CS2)** and **Minecraft**, but it is compatible with other games as well.

The bot detects the game being played, queries Wikipedia to understand the game and its objectives, and begins training based on identified keywords. If no matching keywords are found, it learns through trial and error.

The bot was developed in just **two days**, so it is considered a work in progress and is shared "as is."

---

## Requirements
1. Python 3.12.0
2. OBS Studio
3. Tesseract OCR

---

## Installation
1. Uninstall all existing Python versions and install Python 3.12.0:  
   👉 [Download Python 3.12.0](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe)
2. Install OBS Studio:  
   👉 [Download OBS Studio](https://obsproject.com/download)
3. Install Tesseract OCR:  
   👉 [Download Tesseract](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
4. Install required Python libraries using the following command:  
   ```bash
   pip install numpy opencv-python pillow torch ultralytics pytesseract requests gymnasium stable-baselines3 screeninfo pywin32 keyboard
   ```
   Alternatively, if you have a `requirements.txt` file:  
   ```bash
   pip install -r requirements.txt
   ```
5. Launch OBS Studio, locate the “Start Virtual Camera” button, click the gear icon next to it, and set:  
   - **Output Type**: Source  
   - **Select Output**: Screen Capture
6. In the bot’s code (line 41), specify your virtual camera number (1–3).
7. Restart your PC to ensure Tesseract works correctly and verify the system path:
8. Press `Win + R`, type `sysdm.cpl`, and press Enter.
9. Go to the **Advanced** tab.
10. Click **Environment Variables**.
11. In the **System Variables** section, locate the `PATH` variable and ensure `C:\Program Files\Tesseract-OCR` is included. If not, add it manually.
12. Launch the game you want the bot to play.
13. Run `Start.bat`.

The bot will download YOLOv8 and begin training.

You can open the bot’s GUI window to monitor its actions and view training progress in the command line.

### Bot Controls
- **=**: Pause the bot.
- **Caps Lock**: Stop the bot and save training progress.

---

## Russian Version

Бот создан с использованием **ChatGPT** и **Grok**. Он предназначен для нейронного обучения игре в видеоигры. Бот построен на основе **YOLOv8** и получает изображение с виртуальной камеры OBS.

Основное тестирование проводилось на **Counter-Strike 2 (CS2)** и **Minecraft**, но бот также совместим с другими играми.

Бот определяет, какая игра запущена, запрашивает информацию о ней и ее целях в Wikipedia и начинает обучение на основе найденных ключевых слов. Если ключевые слова не найдены, обучение происходит методом проб и ошибок.

Бот был разработан за **два дня**, поэтому считается незавершенным и предоставляется "как есть".

---

## Требования
1. Python 3.12.0
2. OBS Studio
3. Tesseract OCR

---

## Установка
1. Удалите все установленные версии Python и установите Python 3.12.0:  
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
5. Запустите OBS Studio, найдите кнопку «Запуск виртуальной камеры», нажмите на шестеренку рядом и настройте:  
   - **Тип вывода**: Источник  
   - **Выбрать вывод**: Захват экрана
6. В коде бота (строка 41) укажите номер вашей виртуальной камеры (1–3).
7. Перезагрузите компьютер, чтобы Tesseract корректно работал, и проверьте путь:
8. Нажмите `Win + R`, введите `sysdm.cpl` и нажмите Enter.
9. Перейдите на вкладку **Дополнительно**.
10. Нажмите **Переменные среды**.
11. В разделе **Системные переменные** найдите переменную `PATH` и убедитесь, что в ней указан `C:\Program Files\Tesseract-OCR`. Если путь отсутствует, добавьте его вручную.
12. Запустите игру, в которую должен играть бот.
13. Запустите `Start.bat`.

Бот скачает YOLOv8 и начнет обучение.

Вы можете открыть графический интерфейс бота, чтобы следить за его действиями, а также наблюдать за процессом обучения в командной строке.

### Управление ботом
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
