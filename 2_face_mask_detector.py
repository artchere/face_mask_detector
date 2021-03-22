# Импорт модулей:
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# Распарсим аргументы для командной строки:
ap = argparse.ArgumentParser()
# Путь к изображению:
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
# Путь к модели детектора лиц:
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
# Путь к обученной модели детектора масок:
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
# Порог вероятности:
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
# Инициализация парсера:
args = vars(ap.parse_args())

# Загрузка модели детектора лиц:
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
# Загрузка обученной модели детектора масок:
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# Загрузка изображения, создание копии, получение размеров изображения:
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]
# Создание блоба, вычитание mean (борьба с плохой освещённостью/экспозицией):
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))
# Обнаружение лиц:
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# Проход по обнаружениям:
for i in range(0, detections.shape[2]):
	# Уверенность/вероятность:
	confidence = detections[0, 0, i, 2]
	# Фильтрация обнаружений по отношению к заданному порогу:
	if confidence > args["confidence"]:
		# Вычисление координат прямоугольника для объекта:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# Гарантируем попадание прямоугольника в границы изображения:
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
        # Извлечение интересующей области изображения (ROI),
        # стандартный препроцессинг и повышение размерности:
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		# Предсказание наличия маски:
		(mask, withoutMask) = model.predict(face)[0]
        
        # Определяем метку класса и цвет рамки:
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# Добавляем уверенность классификатора в метку класса:
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# Отображение рамки и текста на кадре:
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Показать изображение:
cv2.imshow("Output", image)
cv2.waitKey(0)