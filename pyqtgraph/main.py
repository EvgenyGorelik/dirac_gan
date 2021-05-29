from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
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
        self.setCentralWidget(self.graphWidget)

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # number of samples for the plot
        self.n = 100
        # learning rate
        h = 0.5
        # some smart function to chose the regularizer here
        self.regularize = regularizers.Reg1(h)
        #self.regularize = regularizers.No_Reg(h)

        # initialize theta and phi for each sample
        self.theta = np.random.rand(self.n,1)*2-1
        self.phi = np.random.rand(self.n,1)*2-1
        self.history_theta = []
        self.history_phi = []
        self.color = []
        self.points = []
        self.traces = []
        # initialize all traces
        for i in range(self.n):
            self.history_theta.append(self.theta[i])
            self.history_phi.append(self.phi[i])
            self.color.append((np.random.rand()*255,np.random.rand()*255,np.random.rand()*255))
            self.points.append(self.graphWidget.plot(pen=None, symbol='o', symbolSize=5,
                                                 symbolBrush=self.color[i]))
            self.traces.append(self.graphWidget.plot(width=1))

        # set up activation function
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.trace)
        self.timer.start()

    # activation function
    def trace(self):
        #iterates over all samples
        for i in range(self.n):
            self.theta[i], self.phi[i] = self.regularize.AGD_step(self.theta[i], self.phi[i])
            self.history_theta[i] = np.append(self.history_theta[i], self.theta[i])
            self.history_phi[i] = np.append(self.history_phi[i], self.phi[i])
            self.traces[i].setData(self.history_phi[i],self.history_theta[i])
            self.points[i].setData([self.phi[i]], [self.theta[i]])

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Plot2D()
    w.show()
    sys.exit(app.exec_())