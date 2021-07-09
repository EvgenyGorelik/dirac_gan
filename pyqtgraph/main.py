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
        self.comboBox.addItem("Moving Average Regularizer")
        self.comboBox.addItem("NS-GAN with no Regularizer")
        self.comboBox.addItem("Consensus Optimization Regularizer")
        self.comboBox.addItem("Instance Noise Regularizer")

        self.gd_method = QtGui.QComboBox(self)
        self.gd_method.addItem("Alternating Gradient Descent")
        self.gd_method.addItem("Simultaneous Gradient Descent")

        self.e1 = QLineEdit("100")
        self.e1.setValidator(QtGui.QIntValidator())
        self.e1.setMaxLength(3)
        self.e1.setAlignment(Qt.AlignRight)
        self.e2 = QLineEdit("0.5")
        self.e2.setValidator(QtGui.QDoubleValidator())
        self.e2.setMaxLength(5)
        self.e2.setAlignment(Qt.AlignRight)
        flo = QFormLayout()
        flo.addRow(self.comboBox)
        flo.addRow(self.gd_method)
        flo.addRow("Number of Samples", self.e1)
        flo.addRow("Learning Rate", self.e2)
        textbox = QGroupBox()
        textbox.setTitle('General Settings')
        textbox.setLayout(flo)


        self.e3 = QLineEdit("0.3")
        self.e3.setValidator(QtGui.QDoubleValidator())
        self.e3.setMaxLength(5)
        self.e3.setAlignment(Qt.AlignRight)
        r1_flo = QFormLayout()
        r1_flo.addRow("Regularization Parameter", self.e3)
        r1_textbox = QGroupBox()
        r1_textbox.setTitle('R1 Settings')
        r1_textbox.setLayout(r1_flo)

        self.e4 = QLineEdit("5")
        self.e4.setValidator(QtGui.QIntValidator())
        self.e4.setMaxLength(5)
        self.e4.setAlignment(Qt.AlignRight)
        self.e5 = QLineEdit("1")
        self.e5.setValidator(QtGui.QDoubleValidator())
        self.e5.setMaxLength(5)
        self.e5.setAlignment(Qt.AlignRight)
        wgan_flo = QFormLayout()
        wgan_flo.addRow("Descriminator Iterations", self.e4)
        wgan_flo.addRow("c", self.e5)
        wgan_textbox = QGroupBox()
        wgan_textbox.setTitle('WGAN Settings')
        wgan_textbox.setLayout(wgan_flo)

        self.e6 = QLineEdit("5")
        self.e6.setValidator(QtGui.QIntValidator())
        self.e6.setMaxLength(5)
        self.e6.setAlignment(Qt.AlignRight)
        self.e7 = QLineEdit("1")
        self.e7.setValidator(QtGui.QDoubleValidator())
        self.e7.setMaxLength(5)
        self.e7.setAlignment(Qt.AlignRight)
        self.e8 = QLineEdit("0.3")
        self.e8.setValidator(QtGui.QDoubleValidator())
        self.e8.setMaxLength(5)
        self.e8.setAlignment(Qt.AlignRight)
        wgan_gp_flo = QFormLayout()
        wgan_gp_flo.addRow("Descriminator Iterations", self.e6)
        wgan_gp_flo.addRow("gamma", self.e7)
        wgan_gp_flo.addRow("g_0", self.e8)
        wgan_gp_textbox = QGroupBox()
        wgan_gp_textbox.setTitle('WGAN-GP Settings')
        wgan_gp_textbox.setLayout(wgan_gp_flo)

        self.e9 = QLineEdit("0")
        self.e9.setValidator(QtGui.QIntValidator())
        self.e9.setMaxLength(5)
        self.e9.setAlignment(Qt.AlignRight)
        self.e10 = QLineEdit("0")
        self.e10.setValidator(QtGui.QDoubleValidator())
        self.e10.setMaxLength(5)
        self.e10.setAlignment(Qt.AlignRight)
        self.e11 = QLineEdit("0.99")
        self.e11.setValidator(QtGui.QDoubleValidator())
        self.e11.setMaxLength(5)
        self.e11.setAlignment(Qt.AlignRight)
        self.e12 = QLineEdit("0.99")
        self.e12.setValidator(QtGui.QDoubleValidator())
        self.e12.setMaxLength(5)
        self.e12.setAlignment(Qt.AlignRight)
        ma_flo = QFormLayout()
        ma_flo.addRow("alpha_r", self.e9)
        ma_flo.addRow("alpha_f", self.e10)
        ma_flo.addRow("gamma", self.e11)
        ma_flo.addRow("lambda", self.e12)
        ma_textbox = QGroupBox()
        ma_textbox.setTitle('Moving-Average Settings')
        ma_textbox.setLayout(ma_flo)

        param_flo = QHBoxLayout()
        param_flo.addWidget(textbox)
        param_flo.addWidget(r1_textbox)
        param_flo.addWidget(wgan_textbox)
        param_flo.addWidget(wgan_gp_textbox)
        param_flo.addWidget(ma_textbox)
        param_textbox = QGroupBox()
        param_textbox.setTitle('Settings')
        param_textbox.setLayout(param_flo)



        buttonlayout = QHBoxLayout()
        mainlayout = QVBoxLayout()
        mainlayout.addWidget(self.graphWidget)
        mainlayout.addWidget(param_textbox)

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
            reg = float(self.e3.text())
            self.regularize = regularizers.Reg1(self.h, reg)
        elif self.comboBox.currentText() == "WGAN Regularizer":
            n_critic = int(self.e4.text())
            c = float(self.e5.text())
            self.regularize = regularizers.WGAN_Reg(self.h, n_critic, c)
        elif self.comboBox.currentText() == "WGAN-GP Regularizer":
            n_critic = int(self.e6.text())
            gamma = float(self.e7.text())
            g_0 = float(self.e8.text())
            self.regularize = regularizers.WGAN_GP_reg(self.h, n_critic, gamma, g_0)
        elif self.comboBox.currentText() == "Moving Average Regularizer":
            alpha_r = float(self.e9.text())
            alpha_f = float(self.e10.text())
            gamma = float(self.e11.text())
            lambd = float(self.e12.text())
            self.regularize = regularizers.Moving_Average_Reg(self.h, alpha_r, alpha_f, gamma, lambd)
        elif self.comboBox.currentText() == "NS-GAN with no Regularizer":
            self.regularize = regularizers.No_Reg_Non_Sat(self.h)
        elif self.comboBox.currentText() == "Consensus Optimization Regularizer":
            self.regularize = regularizers.Reg_Cons_Opt(self.h)
        elif self.comboBox.currentText() == "Instance Noise Regularizer":
            self.regularize = regularizers.Reg_Inst_Noise(self.h)

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
            if self.gd_method.currentText() == "Alternating Gradient Descent":
                self.theta[i], self.phi[i] = self.regularize.AGD_step(self.theta[i], self.phi[i])
            if self.gd_method.currentText() == "Simultaneous Gradient Descent":
                self.theta[i], self.phi[i] = self.regularize.SGD_step(self.theta[i], self.phi[i])
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
