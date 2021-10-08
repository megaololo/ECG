import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton, QLabel, QVBoxLayout,QMessageBox)
from PyQt5.QtGui import QPixmap
from tensorflow.python.keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# минималистичный виджет-интерпретатор ЭКГ
class DialogApp(QWidget):
	def __init__(self):
		super().__init__()
		self.resize(600, 400)
		self.setWindowTitle('ЭКГ анализ')

		self.button1 = QPushButton('Загрузите изображение') # создание кнопки
		self.button1.clicked.connect(self.get_image_file) # привязывание функции
		self.button2 = QPushButton('Анализ')
		self.button2.clicked.connect(self.Predictions)


		self.labelImage = QLabel()

		layout = QVBoxLayout()
		layout.addWidget(self.button1)
		layout.addWidget(self.button2)
		layout.addWidget(self.labelImage)
		self.setLayout(layout)


	def get_image_file(self): # функция для импорта файла

		self.file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"<Default dir>", "Image files (*.jpg *.jpeg *.gif *.png)")

		self.labelImage.setPixmap(QPixmap(self.file_name)) # выводит изображение в окно виджета

	def Predictions(self):
		self.result = QMessageBox() # создание всплывающего окна
		self.result.setWindowTitle('Результат интерпретация')

		self.model = load_model('my_model.h5') # весы нашей нейросети
		# препроцессинг изображения
		self.img = image.load_img(self.file_name, target_size=(150, 150))
		self.img = image.img_to_array(self.img)
		self.img = np.expand_dims(self.img, axis=0)

		if self.model.predict(self.img)[0][0] == 1: # вывод педсказаний [1,0,0] - аритмия
			self.result.setInformativeText('НАРУШЕНИЕ РИТМА')
		elif self.model.predict(self.img)[0][1] == 1: # [0,1,0] - инфаркт миокарда
			self.result.setInformativeText('ИНФАРКТ МИОКАРДА')
		else:
			self.result.setInformativeText( 'ПАТОЛОГИИ НЕ ВЫЯВЛЕНО')

		self.result.exec_()


if __name__ == '__main__':
	app = QApplication(sys.argv)

	demo = DialogApp()
	demo.show()

	sys.exit(app.exec_())