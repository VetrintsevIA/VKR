# Ветринцев Илья гр. 148 4 курс.
import time
import sys
import os
import cv2
import easyocr
import glob
from ultralytics import YOLO
import numpy as np
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QFont, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit)


# Класс отвечает за обрабокту изображения поступающего на вход алгоритма
class ImageProcessor(QThread):
    progress_updated = pyqtSignal(str, np.ndarray, list, str)
    processing_finished = pyqtSignal()


    def __init__(self, directory, ocr_model, yolo_model):
        super().__init__()
        self.directory = directory
        self.ocr_model = ocr_model
        self.yolo_model = yolo_model
        self.running = True


    # Сама работа алгоритма 
    def run(self):
        image_files = glob.glob(os.path.join(self.directory, '*.png')) + \
                     glob.glob(os.path.join(self.directory, '*.jpg')) + \
                     glob.glob(os.path.join(self.directory, '*.bmp'))
        for img_path in image_files:
            if not self.running:
                break

            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Работа YOLO
            yolo_results = self.yolo_model(img_rgb)
            boxes = yolo_results[0].boxes

            # Получаем результат распознавания
            detection_info = []
            if len(boxes.data) > 0:
                for box in boxes.data:
                    x1, y1, x2, y2, conf = map(float, box[:5])
                    detection_info.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'conf': float(conf)
                    })
                    
            # Работа OCR
            if detection_info:
                bbox = detection_info[0]['bbox']
                plate_img = img_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            else:
                plate_img = img_rgb
            results = read_text_advanced(self.ocr_model, plate_img)
            recognized_text = results[0][1] if results else "Текст не обнаружен"
            self.progress_updated.emit(img_path, img_rgb, detection_info, recognized_text)
            
            time.sleep(3) # Убрать, чтобы алгоритм не стоял афк после каждой фотки. 

        self.processing_finished.emit()


    # Остановка алгоритма
    def stop(self):
        self.running = False


# Класс главного окна
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ocr_model = None
        self.yolo_model = None
        self.initUI()
        self.setIcon()
        self.loadModels()


    # Установка иконки приложения
    def setIcon(self):
        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, 'imgs', 'icons', 'icon.png')
        else:
            icon_path = os.path.join('imgs', 'icons', 'icon.png')
        
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))


    # Загрузка моделей
    def loadModels(self):
        try:
            # Загрузка OCR
            self.ocr_model = load_improved_ocr()

            # Загрузка YOLO
            if hasattr(sys, '_MEIPASS'):
                yolo_model_path = os.path.join(sys._MEIPASS, 'v12.pt')  # Путь для сборки
            else:
                yolo_model_path = 'v12.pt'  # Локальный путь во время разработки

            if not os.path.exists(yolo_model_path):
                raise FileNotFoundError(f"Файл модели YOLO не найден: {yolo_model_path}")
            
            self.yolo_model = YOLO(yolo_model_path)

            self.log_text.append("Загрузка моделей прошла успешно!\n")
        except Exception as e:
            self.log_text.append(f"Ошибка загрузки модели(ей)! {str(e)}\n")


    def initUI(self):
        self.setWindowTitle('Распознаватель номерных знаков')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
                font-family: 'Times New Roman';
                font-size: 14px;
                padding: 5px;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                font-family: 'Times New Roman';
                font-size: 14px;
                border: none;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QTextEdit {
                color: white;
                font-family: 'Times New Roman';
                font-size: 16px;
                background-color: #3d3d3d;
                border: 1px solid #4d4d4d;
                padding: 10px;
                line-height: 1.5;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        images_widget = QWidget()
        images_layout = QHBoxLayout(images_widget)
        images_widget.setFixedHeight(400)

        original_widget = QWidget()
        original_layout = QVBoxLayout(original_widget)
        self.original_image = QLabel()
        self.original_image.setFixedSize(380, 380)
        self.original_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image.setStyleSheet("border: 2px solid #4d4d4d;")
        original_label = QLabel("Исходное изображение")
        original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_layout.addWidget(self.original_image)
        original_layout.addWidget(original_label)

        yolo_widget = QWidget()
        yolo_layout = QVBoxLayout(yolo_widget)
        self.yolo_image = QLabel()
        self.yolo_image.setFixedSize(380, 380)
        self.yolo_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.yolo_image.setStyleSheet("border: 2px solid #4d4d4d;")
        yolo_label = QLabel("Результат YOLO-детекции")
        yolo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        yolo_layout.addWidget(self.yolo_image)
        yolo_layout.addWidget(yolo_label)

        ocr_widget = QWidget()
        ocr_layout = QVBoxLayout(ocr_widget)
        self.ocr_image = QLabel()
        self.ocr_image.setFixedSize(380, 380)
        self.ocr_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocr_image.setStyleSheet("border: 2px solid #4d4d4d;")
        ocr_label = QLabel("Реузльтат распознавания OCR")
        ocr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ocr_layout.addWidget(self.ocr_image)
        ocr_layout.addWidget(ocr_label)

        images_layout.addWidget(original_widget)
        images_layout.addWidget(yolo_widget)
        images_layout.addWidget(ocr_widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                color: white;
                font-family: 'Times New Roman';
                font-size: 16px;
                background-color: #3d3d3d;
                border: 1px solid #4d4d4d;
                padding: 10px;
                line-height: 1.5;
            }
        """)

        # Кнопки
        buttons_layout = QHBoxLayout()
        self.select_dir_btn = QPushButton('Выбрать папку')
        self.start_btn = QPushButton('Начать обработку')
        self.export_btn = QPushButton('Выгрузить логи')
        buttons_layout.addWidget(self.select_dir_btn)
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.export_btn)

        layout.addWidget(images_widget)
        layout.addWidget(self.log_text)
        layout.addLayout(buttons_layout)

        self.select_dir_btn.clicked.connect(self.selectDirectory)
        self.start_btn.clicked.connect(self.startProcessing)
        self.export_btn.clicked.connect(self.exportStats)

        self.results = []
        self.current_directory = None

    # Рисовалка того, что нашла YOLO
    def drawDetectionWithConf(self, img, bbox, conf):
        img_height, img_width = img.shape[:2]
        x1, y1, x2, y2 = bbox
        
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        
        painter = QPainter(pixmap)
        
        pen = QPen(Qt.GlobalColor.red, 2)
        painter.setPen(pen)
        
        painter.drawRect(x1, y1, x2-x1, y2-y1)
        
        font = QFont('Times New Roman', 12, QFont.Weight.Bold)
        painter.setFont(font)
        conf_text = f"Conf: {conf:.2f}"
        painter.drawText(x1, y1-5, conf_text)
        
        painter.end()
        return pixmap


    # Функция вызов которой загружает предобученую модель OCR    
    def loadOCR(self):
        try:
            self.ocr_model = load_improved_ocr()
            self.info_label.setText("Модель OCR загружена!")
        except Exception as e:
            self.info_label.setText(f"Модель OCR не смогла загрузиться: {str(e)}")

    # Функция выбора директории
    def selectDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выбрать папку")
        if dir_path:
            self.current_directory = dir_path
            self.log_text.append(f"Выбрать папку: {dir_path}\n")

    # Старт обработки
    def startProcessing(self):
        if not self.current_directory or not self.ocr_model or not self.yolo_model:
            self.log_text.append("Пожалуйста, выберите папку и убедитесь, что все модели загружены!\n")
            return

        self.start_btn.setEnabled(False)
        self.select_dir_btn.setEnabled(False)
        
        self.processor = ImageProcessor(self.current_directory, self.ocr_model, self.yolo_model)
        self.processor.progress_updated.connect(self.updateProgress)
        self.processor.processing_finished.connect(self.processingFinished)
        self.processor.start()


    def updateProgress(self, img_path, img_rgb, detection_info, text):
        height, width = img_rgb.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        orig_pixmap = QPixmap.fromImage(q_img)
        self.original_image.setPixmap(orig_pixmap.scaled(
            self.original_image.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        if detection_info:
            detection = detection_info[0]
            yolo_pixmap = self.drawDetectionWithConf(
                img_rgb.copy(),
                detection['bbox'],
                detection['conf']
            )
            self.yolo_image.setPixmap(yolo_pixmap.scaled(
                self.yolo_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            bbox = detection['bbox']
            plate_img = img_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
            plate_height, plate_width = plate_img.shape[:2]
            plate_bytes_per_line = 3 * plate_width
            q_plate = QImage(plate_img.data, plate_width, plate_height, 
                           plate_bytes_per_line, QImage.Format.Format_RGB888)
            plate_pixmap = QPixmap.fromImage(q_plate)
            
            painter = QPainter(plate_pixmap)
            font = QFont('Times New Roman', 12, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawText(10, 30, text)
            painter.end()
            
            self.ocr_image.setPixmap(plate_pixmap.scaled(
                self.ocr_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        
        file_name = os.path.basename(img_path)
        log_entry = f"\nFile: {file_name}\n"
        log_entry += f"Распознанный текст: {text}\n"
        if detection_info:
            log_entry += f"Уверенность: {detection_info[0]['conf']:.2f}\n"
        log_entry += "=" * 50 + "\n"
        self.log_text.append(log_entry)
        
        self.results.append((file_name, text))


    def processingFinished(self):
        self.start_btn.setEnabled(True)
        self.select_dir_btn.setEnabled(True)
        self.log_text.append("\nОбработка завершена!\n" + "="*50 + "\n")

    def exportStats(self):
        if not self.results:
            self.log_text.append("Нечего выгружать!\n")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for file_name, text in self.results:
                        f.write(f"{file_name}: {text}\n")
                self.log_text.append("Логи успешно выгружены!\n")
            except Exception as e:
                self.log_text.append(f"Ошибка выгрузки логов!: {str(e)}\n")

    def closeEvent(self, event):
        if hasattr(self, 'processor'):
            self.processor.stop()
            self.processor.wait()
        event.accept()


def load_improved_ocr():
    if hasattr(sys, '_MEIPASS'):
        models_dir = os.path.join(sys._MEIPASS, 'ocr_models')
    else:
        models_dir = 'ocr_models'
    model_file = 'iter_1.pth'
    yaml_file = 'iter_1.yaml'
    
    for file in [model_file, yaml_file]:
        if not os.path.exists(os.path.join(models_dir, file)):
            raise FileNotFoundError(f"Не удалось найти файл: {file} в {models_dir}")
    reader = easyocr.Reader(
        ['en'],
        model_storage_directory=models_dir,
        user_network_directory=models_dir,
        recog_network=model_file.replace('.pth', ''),
        detector=True
    )
    
    return reader


def read_text_advanced(reader: easyocr.Reader, image):
    result = reader.readtext(image)
    return result

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 1200, 800)
    window.show()
    sys.exit(app.exec())

