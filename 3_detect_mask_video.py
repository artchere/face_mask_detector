# Импорт модулей:
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Функция для детекции лица и классификации маска/не маска:
def detect_and_predict_mask(frame, faceNet, maskNet):
	# Создание блоба, вычитание mean (борьба с плохой освещённостью/экспозицией):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	# Передача блоба на вход сети:
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# Инициализируем списки с лицами,
    # соответствующими координатами и предсказаниями (маска/не маска):
	faces = [] # Лица
	locs = [] # Координаты
	preds = [] # Предсказание
    
    # Проход по обнаружениям:
	for i in range(0, detections.shape[2]):
		# Извлечение уверенности (вероятности) классификатора:
		confidence = detections[0, 0, i, 2]
		# Фильтрация слабых обнаружений по заданному порогу вероятности:
		if confidence > args["confidence"]:
			# Вычисление координат ограничивающей рамки:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# Гарантируем попадание прямоугольника в границы изображения:
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Извлечение ROI (интересующих областей), изменение цветового
            # профиля, resize и препроцессинг:
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			# Обновляем списки лиц и координат:
			faces.append(face)
			locs.append((startX, startY, endX, endY))
            
    # Предсказание делается только в случае обнаружения 1 и более лиц:
	if len(faces) > 0:
		# Для улучшения быстродействия реализуется пакетный прогноз
        # для всех лиц, а не поочередный для каждого лица в цикле:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=8) #32 by default
	# Возвращаем набор векторов с лицами и предиктами:
	return (locs, preds)

# Снова распарсим аргументы:
ap = argparse.ArgumentParser()
# Путь к модели детектора лиц:
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
# Путь к обученному классификатору масок:
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
# Порог вероятности (уверенность классификатора):
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Загрузка модели детектора лиц и его конфигурации:
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# Загрузка модели обученного классификатора масок:
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Инициализация видеопотока:
print("[INFO] starting video stream...")
vs = VideoStream(#src=1, # Номер девайса (если камера не одна)
                 framerate=20, # Частота кадров
                 #resolution=(640, 480) # Разрешение
                 ).start()
time.sleep(2.0)

# Перебор кадров в видеопотоке:
while True:
	# Захват кадра и resize до заданного размера:
	frame = vs.read()
	frame = imutils.resize(frame, width=360)
	# Обнаружение лица в кадре и предсказание маска/не маска:
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Проход по координатам и предсказаням:
	for (box, pred) in zip(locs, preds):
		# Распаковка переменных:
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# Определим метку класса и цвет рамки/текста:
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 245, 0) if label == "Mask" else (0, 0, 245)
		# Добавим к метке вероятность в %:
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# Отображение рамки и текста на кадре:
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
    # Отобразить выходной кадр:
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# Если была нажата клавиша "q", выйти из цикла:
	if key == ord("q"):
		break
    
# Сделать очистку:
cv2.destroyAllWindows()
vs.stop()
    
