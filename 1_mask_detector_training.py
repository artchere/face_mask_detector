# Импорт модулей:
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import cv2

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Распарсим аргументы для командной строки:
ap = argparse.ArgumentParser()
# Путь ко входному датасету:
ap.add_argument("-d", "--dataset", #required=True,
    help="path to input dataset")
# Опциональный путь к выходному графику истории обучения.
# По умолчанию график называется plot.png,
# если в командной строке не задано другое имя:
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
# Опциональный путь к выходной модели.
# По умолчанию она называется covid19.model:
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
    help="path to output loss/accuracy plot")
# Во избежание 'An exception has occurred, use %tb to see the full traceback':
ap.add_argument('-f') 
# Инициализация парсера:
args = vars(ap.parse_args())

# Инициализируем гиперпараметры - скорость обучения, число эпох и размер батча:
INIT_LR = 1e-4
EPOCHS = 20
BS = 40

# Создание списка путей к изображениям,
# списка данных и соответствующих меток классов:
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Перебор путей к изображениям:
for imagePath in imagePaths:
    # Извлечение метки класса из имени:
    label = imagePath.split(os.path.sep)[-2]
    
    # OpenCV на некоторых картинках может выдавать
    # 'chrm invalid chromaticities' и т.д.,
    # что, вероятно, связано с несоответствием некоторых
    # типов цветовых пространств входных изоражений ожидаемому 'RGB'.
    # Чтение, преобразование в массив, предобработка:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Обновление списка данных и меток классов:
    data.append(image)
    labels.append(label)

# В массивы numpy + масштабирование:
data = np.array(data, dtype='float32') / 255.
labels = np.array(labels)

# Бинарное кодирование для меток классов:
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print(labels)

# Разделение данных на обучающую и валидационную части:
x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.20,
                                                    stratify=labels,
                                                    random_state=0)

# Генератор аугментаций:
train_data_gen = ImageDataGenerator(rotation_range=20,
                                    horizontal_flip=True,
                                    fill_mode="nearest")

# Загрузка предобученной на imagenet сети MobileNetV2 в качестве базовой:
baseModel = MobileNetV2(weights="imagenet",
                        input_shape=(224, 224, 3),
                        input_tensor=Input(shape=(224, 224, 3)),
                        include_top=False)

# Заморозка слоёв базовой сети:
baseModel.trainable = False

# Наращивание сети поверх базовой MobileNetV2:
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # Или 1 + sigmoid

# Объединение двух архитектур:
model = Model(inputs=baseModel.input,
              outputs=headModel)

print("[INFO] compiling model...")
# Оптимизатор:
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# Компиляция модели:
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# Обучение полученной сети:
print("[INFO] training head...")
H = model.fit_generator(train_data_gen.flow(x_train, y_train,
                                            batch_size=BS),
                        steps_per_epoch=len(x_test) // BS,
                        validation_data=(x_test, y_test),
                        validation_steps=len(x_test) // BS,
                        epochs=EPOCHS)

# Предсказания на валидационной части данных:
print("[INFO] evaluating network...")
y_pred = model.predict(x_test, batch_size=BS)

y_pred = np.argmax(y_pred, axis=1)

# Отчёт о качестве классификации:
print(classification_report(y_test.argmax(axis=1), y_pred,
                            target_names=lb.classes_))

# Построение графика функции потерь и точности модели на обучении и валидации:
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Data")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
#plt.savefig(args["plot"])

# Serialize the model to disk
print("[INFO] saving model...")
model.save(args["model"], save_format="h5")
