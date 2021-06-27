from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
from numpy import arange, sin, cos, pi
import pyqtgraph as pg
import sys

# regularizer functions
import regularizers

#GUI Plot class
class Plot2D(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        # set up plot configurations
        super(Plot2D, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setXRange(-1, 1, padding=0)
        self.graphWidget.setYRange(-1, 1, padding=0)
        self.graphWidget.getPlotItem().hideAxis('bottom')
        self.graphWidget.getPlotItem().hideAxis('left')
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.trace)

        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.addItem("No Regularizer")
        self.comboBox.addItem("R1 Regularizer")
        self.comboBox.addItem("WGAN Regularizer")
        self.comboBox.addItem("WGAN-GP Regularizer")


        self.e1 = QLineEdit("100")
        self.e1.setValidator(QtGui.QIntValidator())
        self.e1.setMaxLength(3)
        self.e1.setAlignment(Qt.AlignRight)

        self.e2 = QLineEdit("0.5")
        self.e2.setValidator(QtGui.QDoubleValidator())
        self.e2.setMaxLength(5)
        self.e2.setAlignment(Qt.AlignRight)

        flo = QFormLayout()
        flo.addRow("Number of Samples", self.e1)
        flo.addRow("Learning Rate", self.e2)

        buttonlayout = QHBoxLayout()
        mainlayout = QVBoxLayout()
        mainlayout.addWidget(self.graphWidget)
        mainlayout.addWidget(self.comboBox)
        textbox = QGroupBox()
        textbox.setLayout(flo)
        mainlayout.addWidget(textbox)

        b1 = QPushButton("Start Animation")
        b1.setCheckable(True)
        b1.clicked.connect(self.startAnimation)
        buttonlayout.addWidget(b1)
        b2 = QPushButton("Stop Animation")
        b2.setCheckable(True)
        b2.clicked.connect(self.stopAnimation)
        buttonlayout.addWidget(b2)
        buttonbox = QGroupBox(self)
        buttonbox.setLayout(buttonlayout)
        mainlayout.addWidget(buttonbox)

        mainbox = QGroupBox(self)
        mainbox.setLayout(mainlayout)
        self.setCentralWidget(mainbox)

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)


    def stopAnimation(self):
        self.timer.stop()


    def startAnimation(self):
        self.graphWidget.clear()
        # number of samples for the plot
        self.n = int(self.e1.text())
        # learning rate
        self.h = float(self.e2.text())
        # some smart function to chose the regularizer here
        if self.comboBox.currentText() == "No Regularizer":
            self.regularize = regularizers.No_Reg(self.h)
        elif self.comboBox.currentText() == "R1 Regularizer":
            reg = 0.3
            self.regularize = regularizers.Reg1(self.h, reg)
        elif self.comboBox.currentText() == "WGAN Regularizer":
            n_critic = 5
            c = 1
            self.regularize = regularizers.WGAN_Reg(self.h, n_critic, c)
        elif self.comboBox.currentText() == "WGAN-GP Regularizer":
            n_critic = 5
            gamma = 1.0
            g_0 = 0.3
            self.regularize = regularizers.WGAN_GP_reg(self.h, n_critic, gamma, g_0)

        # initialize theta and phi for each sample
        self.theta = np.random.rand(self.n, 1)*2-1
        self.phi = np.random.rand(self.n, 1)*2-1
        self.history_theta = []
        self.history_phi = []
        self.color = []
        self.points = []
        self.traces = []
        # initialize all traces
        for i in range(self.n):
            self.history_theta.append(self.theta[i])
            self.history_phi.append(self.phi[i])
            self.color.append((np.random.rand()*255, np.random.rand()*255, np.random.rand()*255))
            self.points.append(self.graphWidget.plot(pen=None, symbol='o', symbolSize=5,
                                                 symbolBrush=self.color[i]))
            self.traces.append(self.graphWidget.plot(width=1))
        # set up activation function
        self.timer.start()


    # activation function
    def trace(self):
        #iterates over all samples
        for i in range(self.n):
            self.theta[i], self.phi[i] = self.regularize.AGD_step(self.theta[i], self.phi[i])
            self.history_theta[i] = np.append(self.history_theta[i], self.theta[i])
            self.history_phi[i] = np.append(self.history_phi[i], self.phi[i])
            self.traces[i].setData(self.history_phi[i], self.history_theta[i])
            self.points[i].setData([self.phi[i]], [self.theta[i]])

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Plot2D()
    w.show()
    sys.exit(app.exec_())