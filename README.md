# AI Game Bot

## English Version

The bot was created using **ChatGPT** and **Grok**.  
The bot is based on neural learning to play games.  
It is built on **YOLOv8** and takes images from the **OBS virtual camera**.  

It was mainly tested on **CS2** and **Minecraft**, but it also works in other games.

The bot detects which game is being played, asks Wikipedia what the game is and what to do in it, and, depending on the keywords, begins training.  
If the keywords do not match, it learns by trial and error.  

The bot was written in just **two days**, so it can be considered raw, and I am posting it “as is.”

---

## Requirements
1. Python 3.12.0  
2. OBS Studio  
3. Tesseract  

---

## Installation
1. Remove all versions of Python. Install Python 3.12.0:  
   👉 https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe  
2. Install OBS Studio:  
   👉 https://obsproject.com/download  
3. Install Tesseract:  
   👉 https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe  
4. Install the libraries using the command:  
   ```bash
   pip install numpy opencv-python pillow torch ultralytics pytesseract requests gymnasium stable-baselines3 screeninfo pywin32 keyboard

Or, type:

    pip install -r requirements.txt

5. Launch OBS Studio, to the right of the “Start Virtual Camera” button, click the gear icon - Output Type: Source, Select Output: Screen Capture.
6. In line 41, enter your virtual camera number (1-3).
7. Restart your PC for Tesseract to start working and check the path:
8. Press Win+R, enter sysdm.cpl
9. Go to the “Advanced” tab
10. Click the “Environment Variables” button
11. In the system variables, find the “PATH” line, open it, and you should see C:\Program Files\Tesseract-OCR. If it is not there, enter it manually.
12. Start the game.
13. Run Start.bat

The bot will download YOLO and start learning.

You can bring up the bot's GUI window to see what it is doing, as well as observe its actions and training in the command line.

Bot control:
The “=” button pauses the bot.
The “Caps Lock” button turns off the bot and saves the training progress.

## Русская версия

Бот создат с помощью **ChatGPT** и **Grok**.

Основа бота - нейронное обучение играть в игры. Бот построен на YOLOv8, берет картинку с виртуальной камеры OBS.  Тестировал в основном на КС2 и Майнкрафт. Но также он работает в других играх.

Бот видит, что за игра написана, спрашивает у Wikipedia, что за игра и что в ней делать. и в зависимости от ключевых слов, начинает обучение. если ключевые слова не совпадают - он обучается методом иследования. Бот написан за 2 дня, по этому считать можно его сырым и выкладываю "как есть".

Требования:
1. Python 3.12.0
1. OBS Studio
2. Tesseract

Установка:
1. Удалите все версии python. Установите python 3.12.0:
     👉 https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe
3. Установите OBS Studio:
     👉 https://obsproject.com/download
5. Установите Tesseract:
     👉 https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe
7. Установите библиотеки командой: 
   ```bash
   pip install numpy opencv-python pillow torch ultralytics pytesseract requests gymnasium stable-baselines3 screeninfo pywin32 keyboard

Или жеЖ

    pip install -r requirements.txt
8. Запустите OBS Studio, справа от кнопки "Запуск виртуальной камеры", нажмите шестеренку - Тип вывода: источник, Выбрать вывод: Захват экрана.
9. В строке 41 - вставьте номер вашей виртуальной камеры (1-3).
10. Перезагрузите ПК, чтоб Tesseract начал работать и проверьте путь:
11. Нажмите Win+R, введите sysdm.cpl
12. Вкладка "Дополнительно"
13. Нажмите кнопку "Переменные среды"
14. В системных переменных найдите строку "PATH", откройте ее, вы должны видеть C:\Program Files\Tesseract-OCR, если ее нет - введите вручную.
15. Запустите игру.
16. Запустите Start.bat

Бот докачает YOLO и начнет обучаться.

Можно вывести окно GUI бота, чтоб видеть что он делает, а также наблюдать его действия и обучения в командной строке.

Управление ботом:
Кнопка "=" - пауза бота.
Кнопка "Caps Lock" - выключение бота и сохранения прогреса обучения.
