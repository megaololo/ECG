from tensorflow.python.keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # вычисления через ЦП

model=load_model('my_model.h5') # Загружаем весы нейросети
# Путь к файлу. Препроцессинг.
img = image.load_img('dataset/ECG/train/MI(200).jpg', target_size = (150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
print(model.predict(img))
def Predictions():
     if model.predict(img)[0][0]==1:
         print('Аритмия')
     elif model.predict(img)[0][1]==1:
         print('Инфаркт имиокарда')
     else:
         print('Патологии не обнаружено')
Predictions() # Вывод предсказания