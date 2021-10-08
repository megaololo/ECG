from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # чтобы tensorflow не использовал GPU

# скрипт для создания дирректорий, копирования изображений, созлания train val test
# train - обучение val - проверка test  - тестирование
data_dir = 'dataset/ECG/train' # папка с датасетом

train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
# Часть набора данных для тестирования
test_data_portion = 0.15 # 15% от
# Часть набора данных для проверки
val_data_portion = 0.15
# Количество элементов данных в одном классе
nb_images = 230

def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "MI"))
    os.makedirs(os.path.join(dir_name, "HB"))
    os.makedirs(os.path.join(dir_name, "Normal"))
create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

def copy_images(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(source_dir, "MI(" + str(i) + ").jpg"),
                    os.path.join(dest_dir, "MI"))
        shutil.copy2(os.path.join(source_dir, "HB(" + str(i) + ").jpg"),
                   os.path.join(dest_dir, "HB"))
        shutil.copy2(os.path.join(source_dir, "Normal(" + str(i) + ").jpg"),
                     os.path.join(dest_dir, "Normal"))
start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
print(start_val_data_idx)
print(start_test_data_idx)


copy_images(0, start_val_data_idx, data_dir, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_images(start_test_data_idx, nb_images, data_dir, test_dir)


# данные для обучения модели
train_dir = 'train'
# данные для проверки
val_dir = 'val'
# данные для тестирования
test_dir = 'test'

img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 60
# Размер мини-выборки
batch_size = 16

nb_train_samples = 510 # для обучение (сумма 3-х классов)

nb_validation_samples = 90 # проверка

nb_test_samples = 90 # тестирование

model = Sequential() # создане модели нейронной сети
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))  # 3 выходных нейрона (соответствуют количеству классов)
model.add(Activation('softmax')) # многоклассовая класификация
print(model.summary())
model.compile(loss='categorical_crossentropy', # компиляция модели
              optimizer='sgd',
              metrics=['accuracy'])
datagen = ImageDataGenerator(rescale=1. / 255) # обучение модели с помощью дата генератора

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # много классов
print(train_generator.class_indices)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Accurancy test data: %.2f%%" % (scores[1]*100))

# сохранение (.h5 - весы)
model.save('my_model.h5')
json_string = model.to_json()
model_json = model.to_json()


with open("model_num.json", "w") as json_file:
    json_file.write(model_json)













