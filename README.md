# Face & mask detection

## Описание

__Данный проект я подсмотрел на этом ресурсе:__

https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

Возжелав быстренько скачать готовое решение для TF/Keras (что реализовано в самом тексте статьи),
я совершил вынужденную остановку - по ссылкам на гитхаб, указанным в статье, лежали скрипты для Pytorch,
также радости доставил кривой автоперевод и прочие ништяки.
Соответственно, последовали изучение/разбор и адаптация этого проекта.

__Датасеты качал по следующим ссылкам:__

https://github.com/StephenMilborrow/muct/blob/master/

https://www.kaggle.com/dhruvmak/face-mask-detection

https://www.kaggle.com/shreyashwaghe/face-mask-dataset

https://www.kaggle.com/rakshana0802/face-mask-detection-data

https://www.kaggle.com/andrewmvd/face-mask-detection

Изображения выбирал, обрезал и миксовал интуитивно.
Цветовые профили привёл к RGB 8 bit с помощью PS.

В итоговый датасет вошли по 2600 картинок каждого класса. Скачать:

https://disk.yandex.ru/d/xzAGIUhKNMtkHg

__Проект включает в себя 3 скрипта:__
- Обучение классификатора маска/не маска с использованием предобученной MobileNetV2, сериализация классификатора;
- Реализация детекции лица и классификации маска/не маска, отрисовка рамки и лейбла (нужны resnet-10 в формате caffemodel и наш обученный классификатор);
- Настройка детектора на видеопоток с веб-камеры.

Файлы предобученной resnet-10 (для распознавания лиц) скачал отсюда:

https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel

https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/deploy.prototxt

Структура проекта:

![Image alt](https://github.com/artchere/mask_detector/blob/main/tree.png)

Качество полученного классификатора:

![Image alt](https://github.com/artchere/mask_detector/blob/main/report.png)

## Запуск
__Требуются imutils, tensorflow/keras 2.4.1, OpenCV__

1. Можно сразу запустить скрипт _3_detect_mask_video.py_, который использует уже обученный классификатор масок _mask_detector.model_ (полученный скриптом _1_mask_detector_training.py_) и посмотреть на работу детектора;
2. Можно пройти заново по всем скриптам, то есть с помощью _1_mask_detector_training.py_ обучить классификатор масок на датасете и сохранить его как _mask_detector.model_,
затем _2_face_mask_detector.py_ - здесь можно запустить проверку детектора на изображении, и уже с помощью _3_detect_mask_video.py_ подключиться к веб-камере своего ПК и проверить работу в realtime.

Так как в скриптах есть парсер аргументов, то можно работать с командной строкой:
- Открыть anaconda prompt, перейти в папку с проектом, например, с помощью команды __cd C:\Users\xxxx\mask_detector__
- Передать команду __python 1_mask_detector_training.py --dataset dataset__ (запустится первый скрипт)
- Затем можно запустить второй скрипт командой __python 2_face_mask_detector.py --image examples/0.png__ (проверка на изображении)
- Далее - третий скрипт командой __python 3_detect_mask_video.py__
