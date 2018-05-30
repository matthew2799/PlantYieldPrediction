import sys
import imghdr
import os
import pandas as pd
# pylint: disable=E0611
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QFileDialog, QLabel, QAction, QSizePolicy, QMessageBox, QWidget, QPushButton, QSizePolicy, QMessageBox, QScrollArea,  QLineEdit, QMessageBox
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtCore import pyqtSlot, QDir, Qt
# pylint: enable=E0611

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
 
import random

from CurrentYield.CurrentYieldEstimator import PlantWeightModel
from FutureYield.FutureYieldEstimator   import PlantYieldPredictor
from skimage import io, transform

class YieldViewer(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 960
        self.height = 700

        self.image_path   = None
        self.days_to_harvest = None

        self.valid_day = False
        self.valid_path = False

        self.dataset = pd.read_csv('./data/future_yield_dataset.csv')

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height )
        
        # Image Viewer Stuff
        self.imageLabel = QLabel(self)
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.resize(480, 480)
        self.imageLabel.move(0,0)
        self.imageLabel.setStyleSheet('border: 2px solid black;')

        # Update image path
        self.update_path_button = QPushButton('Update Path', self)
        self.update_path_button.move(285,500)
        self.update_path_button.resize(75,30)
        self.update_path_button.clicked.connect(self.check_valid_path)

        # Image Browser text box and button
        self.browse_button = QPushButton('Browse', self)
        self.browse_button.move(365,500)
        self.browse_button.resize(75,30)
        self.browse_button.clicked.connect(self.open)

        # Image Browser Path Text Box
        self.browse_box = QLineEdit(self)
        self.browse_box.move(30,500)
        self.browse_box.resize(250, 30)
        self.browse_box.textChanged.connect(self.invalid_path)

        # Image Browser text box and button
        self.days_button = QPushButton('Set Days To Harvest', self)
        self.days_button.move(285,535)
        self.days_button.resize(155,30)
        self.days_button.clicked.connect(self.check_valid_day)

        # Day Browser Path Text Box
        self.days_box = QLineEdit(self)
        self.days_box.move(30,535)
        self.days_box.resize(50, 30)
        self.default_box_ss = self.days_box.styleSheet()
        self.days_box.textChanged.connect(self.invalid_day)

        # Graph Plotting Abilities
        self.plot = PlotCanvas(self, width=5, height=5)
        self.plot.move(480,0)
        self.plot.setStyleSheet('border: 2px solid black;')
 
        self.button = QPushButton('PLOT! THAT! YIELD!', self)
        self.button.move(200,600)
        self.button.resize(140,50)
        self.button.clicked.connect(self.predict_yield)

        # Reset Button

        self.button = QPushButton('Reset Everything', self)
        self.button.move(50,600)
        self.button.resize(140,50)
        self.button.clicked.connect(self.reset_everything)

        self.show()

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            self.check_valid_path(fileName)
    
    def load_image(self, path):

        image = QImage(path).scaled(480,480)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot load %s." % path)
            return

        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.browse_box.setText(self.current_filename)
        # self.imageLabel.show()   

    def update_path(self):
        self.image_path = self.browse_box.text()

    def predict_image(self):
        return None

    def reset_everything(self):
        self.imageLabel.clear()
        self.plot.clear_plot()
        self.days_box.clear()
        self.browse_box.clear()
        self.image_path   = None
        self.days_to_harvest = None
        self.valid_day = False
        self.valid_path = False
        self.textbox_clear(self.days_box)
        self.textbox_clear(self.browse_box)
 
    def invalid_path(self):
        self.valid_path = False
        self.browse_box.setStyleSheet("border: 2px solid red;")

    def check_valid_path(self, path=None):
        if path == False:    
            print('here')
            path = self.browse_box.text()

        try:
            if imghdr.what(path) == 'png':
                self.current_filename = path
                self.valid_path = True
                self.browse_box.setStyleSheet("border: 2px solid green;")
                self.load_image(path)
            else:
                self.valid_path = False
                self.browse_box.setStyleSheet("border: 2px solid red;")
        except:
            self.valid_path = False
            self.browse_box.setStyleSheet("border: 2px solid red;")
    
    def invalid_day(self):
        self.valid_day = False
        self.days_box.setStyleSheet("border: 2px solid red;")

    def check_valid_day(self):
        text = self.days_box.text()
        try:
            days = int(text)
            if (days > 0):
                self.valid_day = True
                self.days_to_harvest = days
                self.days_box.setStyleSheet("border: 2px solid green;")
            else:
                self.days_box.clear()
                self.days_box.setStyleSheet("border: 2px solid red;")
        except ValueError:
            self.days_box.clear()
            self.days_box.setStyleSheet("border: 2px solid red;")

    def textbox_clear(self, box):
        box.clear()
        box.setStyleSheet(self.default_box_ss)

    def predict_yield(self):
        if self.valid_day and self.valid_path:
            path = self.current_filename
            days = self.days_to_harvest

            currentYieldPredictor = PlantWeightModel(load_model=True, model_name='current_mlp.pkl')
            futureYieldPredictor  = PlantYieldPredictor(load_model=True, net_name='future_mlp.pkl')

            # Get the yield prediction
            image = io.imread(path)
            current_weight = currentYieldPredictor.Guess(image)

            x_days = list()
            y_prediction = list()
            for i in range(days):
                guess = futureYieldPredictor.Guess(current_weight, i)
                x_days.append(i)
                y_prediction.append(guess)

            try:
                print(self.dataset.head())
                target_data = self.dataset.loc[self.dataset['name'] == os.path.basename(self.current_filename),:]
                print(target_data)
                target = target_data.loc[:,'target_dry']
                self.plot.update_plot(x_days, y_prediction, plot_target=True, target=target, title='Test Plot: ' + os.path.basename(self.current_filename))
            except KeyError:
                self.plot.update_plot(x_days, y_prediction, plot_target=False, title='Yield Prediction: ' + os.path.basename(self.current_filename))

        else:
            if not self.valid_day:
                self.invalid_day()
            if not self.valid_path:
                self.invalid_path()

class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=96):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()
    
    def plot(self):
        self.ax = self.figure.add_subplot(111)
        self.draw()
    
    def dummy_data(self, title):
        data = [random.random() for i in range(25)]
        self.clear_plot()
        self.ax.plot(data, 'r-')
        self.ax.set_title(title)
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self.draw()

    def update_plot(self, days, pred, plot_target=False ,target=None, title=None):
        
        self.clear_plot()
        self.ax.plot(days, pred, 'b-')
        self.ax.plot(days[len(days) - 1], pred[len(days) - 1], color='b', marker='.', markersize=10)
        
        if plot_target:
            self.ax.plot(max(days), target, color='g', marker='.', markersize=10)
            self.ax.legend(['Prediction Curve','Final Prediction','Harvest Truth'])
        else:
            self.ax.legend(['Predition Curve','Final Prediction'])
        
        self.ax.set_xlabel('Days Until Harvest')
        self.ax.set_ylabel('Predicted Yield (mg)')

        self.ax.set_title(title)
        self.draw() 

if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    ex = YieldViewer()
    
    sys.exit(app.exec_())