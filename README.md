# Face & mask detection

![Image alt](https://github.com/artchere/mask_detector/blob/main/test_animation.gif)

## Задача

В данном проекте реализован детектор лиц и классификатор медицинских масок в видеопотоке с помощью OpenCV, Keras и Python.

## Данные

В итоговый датасет вошли по 2600 картинок каждого класса. Скачать:
https://disk.yandex.ru/d/xzAGIUhKNMtkHg

Изображения взяты из открытых источников и предобработаны, цветовые профили приведены к RGB 8 bit с помощью PS.

## Реализация

__Требуются imutils, tensorflow/keras 2.4.1, OpenCV__

__Проект включает в себя 3 скрипта:__
- Обучение классификатора масок с использованием MobileNetV2, сериализация классификатора;
- Реализация детектора лиц и классификатора масок, отрисовка рамки и лейбла в кадре;
- Настройка детектора на видеопоток с веб-камеры.

Веса используемой во 2-м скрипте resnet-10:

https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel

https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/deploy.prototxt

Структура проекта:

![Image alt](https://github.com/artchere/mask_detector/blob/main/tree.png)

Метрика - __accuracy__

Качество полученного классификатора:

![Image alt](https://github.com/artchere/mask_detector/blob/main/report.png)

__Запуск__
1. Можно запустить скрипт _3_detect_mask_video.py_, который использует уже обученный классификатор масок _mask_detector.model_ (полученный скриптом _1_mask_detector_training.py_) и посмотреть на работу детектора;
2. Можно пройти заново по всем скриптам, то есть с помощью _1_mask_detector_training.py_ обучить классификатор масок на датасете и сохранить его как _mask_detector.model_,
затем _2_face_mask_detector.py_ - здесь можно запустить проверку детектора на одиночном изображении, и уже с помощью _3_detect_mask_video.py_ подключиться к веб-камере своего ПК и проверить работу в realtime.

В скриптах есть парсер аргументов, можно работать с командной строкой, например:
- Открыть anaconda prompt, перейти в папку с проектом с помощью команды __cd C:\Users\xxxx\mask_detector__
- Передать команду __python 1_mask_detector_training.py --dataset dataset__ (запустится первый скрипт)
- Затем можно запустить второй скрипт командой __python 2_face_mask_detector.py --image examples/0.png__ (проверка на изображении)
- Далее - третий скрипт командой __python 3_detect_mask_video.py__


## Источники данных

__В данной работе использованы идеи следующего проекта:__

https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

__Датасеты:__

https://github.com/StephenMilborrow/muct/blob/master/

https://www.kaggle.com/dhruvmak/face-mask-detection

https://www.kaggle.com/shreyashwaghe/face-mask-dataset

https://www.kaggle.com/rakshana0802/face-mask-detection-data

https://www.kaggle.com/andrewmvd/face-mask-detection
