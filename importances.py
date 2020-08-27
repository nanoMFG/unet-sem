from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
import os,json,shap,cv2,sys
from PIL import Image
from keras.models import load_model, Model
from keras.layers import Flatten, Reshape
import keras.backend as K
import tensorflow as tf
from deepexplain.tensorflow import DeepExplain
import matplotlib.colors as colors
import matplotlib
from sklearn.metrics import accuracy_score, precision_score, recall_score
from functools import wraps
import datetime

class ImageItem(pg.ImageItem):
    clicked = QtCore.pyqtSignal(object)
    def __init__(self,*args,**kwargs):
        super(ImageItem,self).__init__(*args,**kwargs)
        kern = (np.ones((2,2))*255).astype(np.uint8)
        self.setDrawKernel(kern, mask=None, center=(int(1),int(1)), mode='set')
    
    def drawAt(self,pos,ev=None):
        self.clicked.emit(pos)

class HeaderLabel(QtWidgets.QLabel):
    def __init__(self,*args,**kwargs):
        super(HeaderLabel,self).__init__(*args,**kwargs)
        self.setFont(QtGui.QFont("Helvetica", 20, QtGui.QFont.Bold))

def threader(func, wait=True, run_msg='Running', finished_msg='Finished'):
    @wraps(func)
    def async_func(self, *args, **kwargs):
        if wait==False or (wait==True and (self._runner == None or self._runner.isFinished())):
            runner = Runner(func, self, *args, **kwargs)
            runner.started.connect(lambda: self.status(run_msg))
            runner.done.connect(lambda out: self.status(finished_msg) if out==True else self.status(out))
            
            if wait:
                self._runner = runner
            else:
                async_func.__runner = runner
            runner.start()

    return async_func

class Runner(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    def __init__(self, target, obj ,*args, **kwargs):
        super().__init__()
        self._target = target
        self._obj = obj
        self._args = args
        self._kwargs = kwargs

        self._out = None

        self.finished.connect(lambda: self.done.emit(self._out))

    def run(self):
        try:
            self._target(self._obj,*self._args, **self._kwargs)
            self._out = True
        except Exception as e:
            self._out = str(e)

# def status_check(func):
#     @wraps(func)
#     def wrapper(self,*args,**kwargs):
#         try:
#             func(self)

attribution_methods = [
    'saliency',
    'grad*input',
    'intgrad',
    'elrp',
    'deeplift'
    ]

model_path = os.path.join(os.getcwd(),'N58_1','FOLD_00','model_E099_0.897.hdf5')


class Main(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super(Main,self).__init__(parent=parent)
        self._input = None
        self._mask = None
        self._output = None
        self._attributions = None
        self._runner = None

        self.inputWidget = pg.GraphicsLayoutWidget()
        self.inputViewBox = self.inputWidget.addViewBox(row=1,col=1)
        self.inputItem = ImageItem()
        self.inputViewBox.addItem(self.inputItem)
        self.inputViewBox.setAspectLocked(True)

        self.maskWidget = pg.GraphicsLayoutWidget()
        self.maskViewBox = self.maskWidget.addViewBox(row=1,col=1)
        self.maskItem = pg.ImageItem()
        self.maskViewBox.addItem(self.maskItem)
        self.maskViewBox.setAspectLocked(True)

        self.outputWidget = pg.GraphicsLayoutWidget()
        self.outputViewBox = self.outputWidget.addViewBox(row=1,col=1)
        self.outputItem = pg.ImageItem()
        self.outputViewBox.addItem(self.outputItem)
        self.outputViewBox.setAspectLocked(True)

        self.impWidget = pg.GraphicsLayoutWidget()
        self.impViewBox = self.impWidget.addViewBox(row=1,col=1)
        self.impItem = pg.ImageItem()
        self.impViewBox.addItem(self.impItem)
        self.impViewBox.setAspectLocked(True)

        self.methodSelection = QtWidgets.QComboBox()
        self.methodSelection.addItems(attribution_methods)
        self.methodSelection.setCurrentIndex(0)

        self.modeSelection = QtWidgets.QComboBox()
        self.modeSelection.addItems(['Select Pixel','Black Mask','White Mask','Full Image'])
        self.modeSelection.setCurrentIndex(0)

        self.importInputBtn = QtWidgets.QPushButton("Import Image")
        self.importMaskBtn = QtWidgets.QPushButton("Import Mask")

        self.minPercentile = QtWidgets.QLineEdit('5')
        self.minPercentile.setValidator(QtGui.QDoubleValidator(0,100,3))
        self.maxPercentile = QtWidgets.QLineEdit('95')
        self.maxPercentile.setValidator(QtGui.QDoubleValidator(0,100,3))

        self.attributionHist = pg.PlotWidget()
        self.exportBtn = QtWidgets.QPushButton('Export')

        self.attributionType = QtWidgets.QComboBox()
        self.attributionType.addItems(
            [
                'All Attributions',
                'Positive Attributions',
                'Negative Attributions',
                'Absolute Attributions'
            ])
        self.attributionType.setCurrentIndex(0)
        self.colormapStatus = QtWidgets.QLabel('')
        self.colormapStatus.setWordWrap(True)

        self.statusLabel = QtWidgets.QLabel('')
        self.statusLabel.setWordWrap(True)

        self.precisionLabel = QtWidgets.QLabel('')
        self.accuracyLabel = QtWidgets.QLabel('')
        self.recallLabel = QtWidgets.QLabel('')

        self.layout = QtGui.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addWidget(HeaderLabel("Input Image"),0,0)
        self.layout.addWidget(self.inputWidget,1,0)
        self.layout.addWidget(HeaderLabel("Attributions"),0,1)
        self.layout.addWidget(self.impWidget,1,1)

        self.layout.addWidget(HeaderLabel("Input Mask"),2,0)
        self.layout.addWidget(self.maskWidget,3,0)
        self.layout.addWidget(HeaderLabel("Predicted Mask"),2,1)
        self.layout.addWidget(self.outputWidget,3,1)

        self.statLayout = QtGui.QGridLayout()
        self.statLayout.setAlignment(QtCore.Qt.AlignTop)
        self.statLayout.addWidget(HeaderLabel("Stats"),0,0)
        self.statLayout.addWidget(QtGui.QLabel("Accuracy:"),1,0)
        self.statLayout.addWidget(self.accuracyLabel,1,1)
        self.statLayout.addWidget(QtGui.QLabel("Precision:"),2,0)
        self.statLayout.addWidget(self.precisionLabel,2,1)
        self.statLayout.addWidget(QtGui.QLabel("Recall:"),3,0)
        self.statLayout.addWidget(self.recallLabel,3,1)

        self.controlLayout = QtGui.QGridLayout()
        self.controlLayout.setAlignment(QtCore.Qt.AlignTop)
        self.controlLayout.addWidget(self.importInputBtn,0,0)
        self.controlLayout.addWidget(self.importMaskBtn,0,1)
        self.controlLayout.addWidget(HeaderLabel("Control"),1,0)
        self.controlLayout.addWidget(QtWidgets.QLabel("Method:"),2,0)
        self.controlLayout.addWidget(self.methodSelection,2,1)
        self.controlLayout.addWidget(QtWidgets.QLabel("Mode:"),3,0)
        self.controlLayout.addWidget(self.modeSelection,3,1)
        self.controlLayout.addWidget(QtWidgets.QLabel("Status:"),4,0)
        self.controlLayout.addWidget(self.statusLabel,4,1)
        self.controlLayout.addWidget(self.exportBtn,5,0)

        self.colormapLayout = QtGui.QGridLayout()
        self.colormapLayout.setAlignment(QtCore.Qt.AlignTop)
        self.colormapLayout.addWidget(HeaderLabel("Colormap Bounds"),0,0)
        self.colormapLayout.addWidget(QtWidgets.QLabel("Filter:"),1,0)
        self.colormapLayout.addWidget(self.attributionType,1,1)
        self.colormapLayout.addWidget(QtWidgets.QLabel("Min Percentile:"),2,0)
        self.colormapLayout.addWidget(self.minPercentile,2,1)
        self.colormapLayout.addWidget(QtWidgets.QLabel("Max Percentile:"),3,0)
        self.colormapLayout.addWidget(self.maxPercentile,3,1)
        self.colormapLayout.addWidget(self.attributionHist,4,0,2,2)


        self.panelLayout = QtGui.QGridLayout()
        self.panelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.panelLayout.addLayout(self.controlLayout,0,0)
        self.panelLayout.addLayout(self.colormapLayout,1,0)
        self.panelLayout.addLayout(self.statLayout,2,0)

        self.layout.addLayout(self.panelLayout,0,2,4,1)

        self.minPercentile.textChanged.connect(self.updateAttributionImage)
        self.maxPercentile.textChanged.connect(self.updateAttributionImage)
        self.inputItem.clicked.connect(self.updateAttributions)
        self.importInputBtn.clicked.connect(self.importInput)
        self.importMaskBtn.clicked.connect(self.importMask)
        self.modeSelection.currentIndexChanged[str].connect(lambda s: self.updateAttributions())
        self.attributionType.currentIndexChanged[str].connect(lambda s: self.updateAttributionImage())
        self.exportBtn.clicked.connect(self.export)

    def export(self):
        name = QtWidgets.QFileDialog.getSaveFileName()[0]
        if name != '':
            np.save(name,self._attributions)

    def importInput(self):
        self._input = self.importImage()
        if isinstance(self._input,np.ndarray):
            self.inputItem.setImage(self._input,levels=(0,1))

            inp = self._input[np.newaxis,...,np.newaxis]

            model = load_model(model_path)
            self._output = np.rint(model.predict(inp))[0,...,0]
            self.outputItem.setImage(self._output,levels=(0,1))

            self.updateStats()

            K.clear_session()

    def importMask(self):
        self._mask = self.importImage()
        if isinstance(self._mask,np.ndarray):
            self._mask = np.rint(self._mask)
            self.maskItem.setImage(self._mask,levels=(0,1))
            self.updateStats()

    def updateStats(self):
        if isinstance(self._output,np.ndarray) and isinstance(self._mask,np.ndarray):
            self.accuracyLabel.setText(
                str(round(accuracy_score(
                    self._mask.flatten(),
                    self._output.flatten())
                    ,3))
                )
            self.precisionLabel.setText(
                str(round(precision_score(
                    self._mask.flatten(),
                    self._output.flatten(),
                    pos_label=0)
                    ,3))
                )
            self.recallLabel.setText(
                str(round(recall_score(
                    self._mask.flatten(),
                    self._output.flatten(),
                    pos_label=0)
                    ,3))
                )


    def status(self,msg=None):
        """If msg is None, return status. Otherwise, sets status to msg."""
        if msg:
            self.statusLabel.setText(str(msg))
        else:
            return self.statusLabel.text()

    def updateAttributions(self,pos=None):
        if isinstance(self._input,np.ndarray):
            method = self.methodSelection.currentText()
            mode = self.modeSelection.currentText()

            try:
                self._attributions = self.calculateAttributions(
                    xs = self._input,
                    method = method,
                    mode = mode,
                    model_path = model_path,
                    pos = pos
                    )
                self.status('Finished.')
                self.updateAttributionImage()
            except Exception as e:
                self.status(str(e))

    def calculateAttributions(self,xs,method,mode,model_path,pos=None):
        with DeepExplain(session=K.get_session()) as de:
            model = load_model(model_path)
            flat = Reshape(target_shape=(256*256,))(model.layers[-1].output)
            flat_model = Model(model.layers[0].input,flat)

            input_tensor = flat_model.layers[0].input
            target_tensor = flat_model(input_tensor)
            
            xs = xs[np.newaxis,...,np.newaxis]

            if mode == 'Select Pixel' and pos != None:
                x,y = int(pos.x()),int(pos.y())
                idx = x*256 + y
                ys = np.zeros(256**2)
                ys[idx] = 1
            elif mode == 'Black Mask':
                ys = 1-self._mask.flatten()
            elif mode == 'White Mask':
                ys = self._mask.flatten()
            elif mode == 'Full Image':
                ys = np.ones(256**2)
            ys = ys[np.newaxis,...]

            attributions = de.explain(method,target_tensor,input_tensor,xs,ys)
        K.clear_session()
        return attributions

    def updateAttributionImage(self):
        low = float('0'+self.minPercentile.text())
        high = float('0'+self.maxPercentile.text())

        attributions = self._attributions[0,...,0]
        if self.attributionType.currentText() == 'All Attributions':
            attributions_mask = np.ones_like(attributions,dtype=bool)
            flat_attributions = attributions.flatten()
            mn = np.percentile(flat_attributions,low)
            mx = np.percentile(flat_attributions,high)
            ctr = 0
            temp = [mn,ctr,mx]

            if any(temp[i]>=temp[i+1] for i in range(2)):
                return

            norm = colors.DivergingNorm(vmin=mn, vcenter=ctr, vmax=mx)
            cmap = matplotlib.cm.get_cmap('RdBu_r')

        elif self.attributionType.currentText() == 'Positive Attributions':
            attributions_mask = attributions>=0
            flat_attributions = attributions[attributions_mask].flatten()
            mn = 0
            mx = np.percentile(flat_attributions,high)

            norm = colors.Normalize(vmin=mn,vmax=mx)
            cmap = matplotlib.cm.get_cmap('Reds')

        elif self.attributionType.currentText() == 'Negative Attributions':
            attributions_mask = attributions<=0
            flat_attributions = attributions[attributions_mask].flatten()
            mn = np.percentile(flat_attributions,low)
            mx = 0

            norm = colors.Normalize(vmin=mn,vmax=mx)
            cmap = matplotlib.cm.get_cmap('Blues_r')

        elif self.attributionType.currentText() == 'Absolute Attributions':
            attributions = np.abs(attributions)
            attributions_mask = np.ones_like(attributions,dtype=bool)
            flat_attributions = attributions.flatten()
            mn = 0
            mx = np.percentile(flat_attributions,high)

            norm = colors.Normalize(vmin=mn,vmax=mx)
            cmap = matplotlib.cm.get_cmap('Greens')


        normalized_attribution = norm(attributions)
        mapped_attribution = cmap(normalized_attribution)
        self.impItem.setImage(mapped_attribution,levels=(0,1))

        vals, edges = np.histogram(flat_attributions,bins=100,density=True)

        self.attributionHist.clear()
        self.attributionHist.hideAxis('left')
        self.attributionHist.plot((edges[:-1]+edges[1:])/2,np.log(vals))
        self.attributionHist.setXRange(flat_attributions.min(),flat_attributions.max())
        self.attributionHist.addItem(pg.InfiniteLine(mn))
        self.attributionHist.addItem(pg.InfiniteLine(mx))

    def importImage(self):
        try:
            img_file_path = QtGui.QFileDialog.getOpenFileName()
            if isinstance(img_file_path,tuple):
                img_file_path = img_file_path[0]
            else:
                return None
            img = cv2.imread(img_file_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(img)
            img = img.resize((256,256))
            img = np.array(img)/255
            return img

        except Exception as e:
            print(e)
            return None

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main = Main()
    main.show()
    sys.exit(app.exec_())