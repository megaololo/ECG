# Интерпретация ЭКГ с помощью CNN
Проект свёрточной нейросети для интерпретации ЭКГ. Включает диагностику нарушений ритма, ифаркта миокарда.

## Структура проекта:
- images_test_cnn
  - Arrythmia
  - Myocardial infarction
  - Normal
- dataset
  - dataset.rar
- ECG.pptx
- model_num.json
- my_model.h5
- Prediction_cnn.py
- Qt5_cnn.py
- README.md
- Qt5_cnn(.exe)
  - Qt5_cnn.rar

## Необходимые бибиотеки и порядок работы:
- pip install [keras](https://keras.io)
- pip install [tensorflow](https://www.tensorflow.org)
- pip install [matplotlib](https://matplotlib.org)
- pip install [numpy](https://numpy.org)
- pip install [pyinstaller](https://www.pyinstaller.org)
- pip install [pyQt5](https://pypi.org/project/PyQt5/)

При желании переобучить модель разархивируйте файл **dataset.rar** в папке dataset, запустите **ECG_cnn.py**. После запуска создаст автоматически дирректории *val*, *train*, *test*. При переобучении рекомендуется создать *.h5* и *.json* с другими именеами

Для тестирования работы нейросети запустите **Qt5_cnn.py** (убедитесь что в файле указан путь к *my_model.h5*), и импортируйте изображения из папки *images_test_cnn*

Также можно воспользоваться Qt5_cnn.exe(**pyinstaller -w Qt5_cnn.py**). Для этого разархивируйте *Qt5_cnn.rar* в папке Qt5_cnn(.exe), откройте **.exe** файл в папке *dist*

Более потробное описание проекта **ECG.pptx**.

**Prediction_cnn.py** - возможно загружать весы нейросети и тестовое изображения для предсказаний


