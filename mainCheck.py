# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 23:14:26 2019

@author: majaa
"""
import sys
import numpy as np
import nibabel as nib
#import matplotlib.pyplot as plt
import os
import vtk.util.colors as colors
from PyQt5 import QtCore, QtGui
from PyQt5 import Qt
from fury import window,actor
#from dipy.viz import window, actor

from PyQt5.QtWidgets import QLabel,QRadioButton,QComboBox,QPushButton, QAction, QMenu, QApplication, QFileDialog, QWidget, QSplitter, QGroupBox
from PyQt5.QtGui import QIcon
import PyQt5.QtGui

from PyQt5 import QtCore, QtWidgets

from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from dipy.tracking.streamline import transform_streamlines, length

from shutil import copyfile

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FingureCanvas
from matplotlib.figure import Figure
import pickle
from nibabel import trackvis
from dipy.tracking import utils

import time
from dipy.tracking.streamline import set_number_of_points
from sklearn import svm
import nibabel as nib
from joblib import Parallel, delayed
from dipy.tracking.vox2track import streamline_mapping
from pykdtree.kdtree import KDTree

from kd_tree_segmentation import kd2 
from one_class_svm import ocsvm
from statistical_analysis import stat
from preprocessing import preprocessing as pre

import tkinter.font as tkFont
import tkinter as tk

######### time to import matplotlib functions ############
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100, x=np.random.normal(size = 1000)):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure(x)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self,x):
        # t = arange(0.0, 3.0, 0.01)
        # s = sin(2*pi*t)
        # self.axes.plot(t, s)

        # Plot Histogram on x
        #x = np.random.normal(size = 1000)
        self.axes.hist(x, bins=50)
        #self.axes.gca().set(title='Frequency Histogram', ylabel='Frequency')
    def update_figure(self,x):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        self.axes.cla()
        self.axes.hist(x, bins=50)
        self.draw()


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    #from mainCheck import MainWindow
    

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        
        #self.listen = MainWindow.getListener(MainWindow())
        #print(listen)
        # if(listen):
        #     print(listen)
        #self.update_figure
        # timer = QtCore.QTimer(self)
        # timer.timeout.connect(self.update_figure)
        # timer.start(1000)

    def compute_initial_figure(self, x):
        # from mainCheck import MainWindow
        # object1 = MainWindow()
        #self.object2 = None
        #global object2 
        #self.object2 = object1
        self.axes.hist(x, bins=50)

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        #l = [random.randint(0, 10) for i in range(4)]
        from mainCheck import MainWindow
        object1 = MainWindow()
        x = object1.getX()
        # isUpdatable=object1.setListener()
        print("Ha ami ashsi update korte")
        # if(isUpdatable):
        self.axes.cla()
        self.axes.hist(x, bins=50)
        self.draw()



interactive = False  # set to True to show the interactive display window

class MainWindow(Qt.QMainWindow):

    def getListener(self):
        return self.listener
    def setListener(self):
        self.listener = False
    
    def getX(self):
        return self.whole_brain_length_info_all
    
    def load_streamline2(self,whole_brain_tractogram,fileName2):
        from fury import window
        scene = window.Scene()
        self.no_of_points=12    
        self.leafsize=10
        # whole_brain_id = "100307"
        # #model_tracts =[ "124422", "111312",  "100408", "100307", "856766"]
        # self.tract = "_af.left.trk"
        test_data= pre.load(self, whole_brain_tractogram) 
        # print("Ki shomossha! ",whole_brain_tractogram," 2 ",fileName2)
        if self.method == 1:
            print(fileName2)
            segmented_tract_positive, segmented_tract_negative = ocsvm.oneClassSVM(self, whole_brain_tractogram, fileName2)
            color_tract = colors.blue
        else:
            segmented_tract_positive, segmented_tract_negative = kd2.segmentation_with_NN(self, whole_brain_tractogram, fileName2, self.no_of_points, self.leafsize)
            color_tract = colors.green
        
        
        affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
        
        
        ########### Statistical Analysis ################
        self.whole_brain_length_info_all = stat.length_info_all(self, test_data)
        self.listener = True
        xw = self.whole_brain_length_info_all
        whole_brain_length_info= np.array(stat.length_info (self, test_data))
        max_label = "\nMaximum length of tracts: " + str(whole_brain_length_info[0])
        min_label = "\nMinimum length of tracts: "+  str(whole_brain_length_info[1])
        streamline_count_label = "\nNo of streamlines: "+  str(len(test_data))
        voxel_count_label = "\nNo of voxels:: "+  str(stat.voxel_count(self, test_data))
        #dev_streamline_count_label = "\nNo of streamlines: "+  str(len(test_data))
        #dev_voxel_count_label = "\nNo of voxels:: "+  str(self.voxel_count(test_data))
        
        self.max_length_label.setText(max_label)
        self.min_length_label.setText(min_label)
        self.no_streamlines_label.setText(streamline_count_label)
        #self.dev_no_streamlines_label.setText(dev_streamline_count_label)
        self.no_voxels_label.setText(voxel_count_label)
        #self.dev_no_voxels_label.setText(dev_voxel_count_label)
        ####### Adding Matplotlib
        from mainCheck import MyDynamicMplCanvas as mp 
        from mainCheck import MyMplCanvas as mc
        
        #self.main_widget.update_figure


        # self.main_widget = QtWidgets.QWidget(self)
        # print(self.whole_brain_length_info_all)
        # l = QtWidgets.QVBoxLayout(self.main_widget)
        # sc = MyStaticMplCanvas(self.main_widget, width=5, height=4, dpi=100, x=self.whole_brain_length_info_all)
        # l.addWidget(sc)
        # self.main_widget.setFocus()
        # self.setCentralWidget(self.main_widget)
        # self.predBox.addWidget(self.main_widget)

        # self.predBox.addStretch(2)


        # self.statBox.setLayout(self.predBox)
        # main_widget = QtWidgets.QWidget(self)
        # l = QtWidgets.QVBoxLayout(main_widget)
        # sc = MyStaticMplCanvas(main_widget, width=5, height=4, dpi=100, x=self.whole_brain_length_info_all)
        # l.addWidget(sc)
        # main_widget.setFocus()
        # self.setCentralWidget(main_widget)
        # self.predBox.addWidget(main_widget)
        # self.predBox.widget


        
        #tract_transform = transform_streamlines(test_data, np.linalg.inv(affine))
        #stream_actor = actor.line(tract_transform,(1., 0.5, 0))

        tract_transform2 = transform_streamlines(segmented_tract_positive, np.linalg.inv(affine))
        stream_actor2 = actor.line(tract_transform2,
                           colors=color_tract, linewidth=0.1)
        
        bundle_nativeNeg = transform_streamlines(segmented_tract_negative, np.linalg.inv(affine))
        stream_actorNeg = actor.line(bundle_nativeNeg, colors=colors.red,
                           opacity=0.01, linewidth=0.1)
        
        
        #scene.add(stream_actor)
        scene.add(stream_actor2)        
        scene.add(stream_actorNeg)
        
        self.vtkWidget2.GetRenderWindow().AddRenderer(scene)
        self.iren = self.vtkWidget2.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        self.iren.Start()
    def action_reload(self): 
        Qt.QMainWindow.update(self)

    def load_whole_brain(self,fileName):
        from fury import window
        scene = window.Scene()
        affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
        wholeTract= nib.streamlines.load(fileName)
        wholeTract = wholeTract.streamlines
        wholeTract_transform = transform_streamlines(wholeTract, np.linalg.inv(affine))
        stream_actor = actor.line(wholeTract_transform, colors=colors.white,
                           opacity=0.02, linewidth=0.1)

        # scene.set_camera(position=(-176.42, 118.52, 128.20),
        #         focal_point=(113.30, 128.31, 76.56),
        #         view_up=(0.18, 0.00, 0.98))   
        scene.add(stream_actor)
        self.vtkWidget.GetRenderWindow().AddRenderer(scene)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        self.iren.Start()

    def openTractFilesDialog(self):
        # global Tract_fileName 
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileWithPath, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Track Files (*.trk)", options=options)
        # Tract_fileName=os.path.basename(fileWithPath)
        
        # splitFile=Tract_fileName.split(".")[0]
        # print(splitFile)

        
        # dest="AllPickles/"+Tract_fileName
        # #copyfile(fileWithPath, dest)
        self.combo.clear()
        self.tract_name2 = []
        root_dir = 'G:\\Thesis\\ThesisIm plementations\\Tract Segmentation Visualizer\\Tract-Segmentation-Visualizer\\Tracts\\AF\\Left'
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                self.tract_name2.append(os.path.join(root, name))
                print(os.path.join(root, name))
               
        
        self.combo.addItems(self.tract_name2)
        print("done...")
        
    def openFileNameDialog(self):
        # global fileName 
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileWithPath, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Track Files (*.trk)", options=options)
        # fileName=os.path.basename(fileWithPath)
        # print(fileName)
        # splitFile=fileName.split(".")[0]
        # print(splitFile)
        


        #dest="WholeBrains/"+fileName
        #copyfile(fileWithPath, dest)
        self.combo_box1.clear()
        whole_brain = []
        root_dir = 'G:\\Thesis\\ThesisIm plementations\\Tract Segmentation Visualizer\\Tract-Segmentation-Visualizer\\WholeBrains'
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                whole_brain.append(os.path.join(root, name))
                print(os.path.join(root, name))
                    
            #print(dirs,type(dirs))
        # for (dirpath, dirnames, filenames) in os.walk("."):
        #     print("File ki morse", filenames)
        #     self.whole_brain.extend(str("G:\\Thesis\\ThesisIm plementations\\Tract Segmentation Visualizer\\Tract-Segmentation-Visualizer\\WholeBrains" + filenames))
        #     break
        print("Combo te bombo", whole_brain)
        self.combo_box1.addItems(whole_brain)
        print("done...")

    def checkFile(self,fileName):
        whole_brain = []
        for (dirpath, dirnames, filenames) in os.walk("WholeBrains"):
            whole_brain.extend(filenames)
            break
        if fileName in whole_brain:
            return 1
        else:
            return 0

    def clickMethod(self):

        whole_brain = str(self.combo_box1.currentText())
        print(whole_brain)
        self.load_whole_brain(whole_brain)
    def clickMethod2(self):

        #AllPickles = []
        #for (dirpath, dirnames, filenames) in os.walk("AllPickles"):
        #    AllPickles.extend(filenames)
        #    break
        side = ""
        if self.radio1.isChecked():
            side = "Left"
        elif self.radio2.isChecked():
            side = "Right"
        elif self.radio3.isChecked():
            side = "Left"

        choose_pickle = self.combo.currentText() + side + ".pickle"
        Title = self.combo.currentText() + side

        choosed_pickle = self.tract_name2
        whole_brain = str(self.combo_box1.currentText())
        
        self.method = 1
        if self.ocsvm_radio.isChecked():
            self.method = 1
        elif self.kd_radio.isChecked():
            self.method = 2

        self.load_streamline2(whole_brain,choosed_pickle)
        #if choose_pickle in AllPickles:
        #    choosed_pickle = "AllPickles/" + choose_pickle
        #    whole_brain = "WholeBrains/" + str(self.combo_box1.currentText())
        #    print("choosed_pickle",choosed_pickle)
        #    print("whole_brain", whole_brain)


    def HorizontalLayout(self):
        self.groupBox2 = QGroupBox("Method Visuals")
        self.groupBox2.setStyleSheet("background-color: #006868")
            
        self.vbox2 = Qt.QHBoxLayout()
        
        self.wb_box = Qt.QVBoxLayout()
        
        self.whole_brain_label = QLabel()
        self.whole_brain_label.setText("\nWhole Brain Tractogram")
        self.wb_box.addWidget(self.whole_brain_label)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.wb_box.addWidget(self.vtkWidget)
        self.vbox2.addLayout(self.wb_box)
        # self.vl.addWidget(self.frame)

        ############################################
        self.seg_box = Qt.QVBoxLayout()
        self.vtkWidget2 = QVTKRenderWindowInteractor(self.frame)        
        self.segmented_label = QLabel()
        self.segmented_label.setText("\nTract Segmentation")
        self.seg_box.addWidget(self.segmented_label)
        self.seg_box.addWidget(self.vtkWidget2)
        self.vbox2.addLayout(self.seg_box)
        self.groupBox2.setLayout(self.vbox2)
        #verbox.addLayout(self.vbox2)
        
        ### Adding statistical bar
        self.statBox = QGroupBox("Tract Insights")
        self.statBox.setStyleSheet("QGroupBox {font-size: 15pt; font-weight: bold; font-family: Courier; color: white}")
        self.predBox = Qt.QVBoxLayout()
        
        # ## Creating font with QtFont
        # font = QtGui.QFont()
        # font.setFamily("Helvetica")
        # font.setBold(True)

        self.max_length_label = QLabel()
        self.max_length_label.setText("\nMaximum length of tracts: ")
        self.max_length_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        self.min_length_label = QLabel()
        self.min_length_label.setText("\nMinimum length of tracts: ")
        self.min_length_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        self.fa_label = QLabel()
        self.fa_label.setText("\nFA value: ")
        self.fa_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        self.dev_fa_label = QLabel()
        self.dev_fa_label.setText("\nDeviation FA value: ")
        self.dev_fa_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        self.no_streamlines_label = QLabel()
        self.no_streamlines_label.setText("\nNo of streamlines: ")
        self.no_streamlines_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        self.dev_no_streamlines_label = QLabel()
        self.dev_no_streamlines_label.setText("\nDeviation No of streamlines: ")
        self.dev_no_streamlines_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        self.no_voxels_label = QLabel()
        self.no_voxels_label.setText("\nNo of voxels: ")
        self.no_voxels_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        self.dev_no_voxels_label = QLabel()
        self.dev_no_voxels_label.setText("\nDeviation No of voxels: ")
        self.dev_no_voxels_label.setStyleSheet("QLabel {font-size: 10pt; font-family: Courier; color: white}")
        
        self.predBox.addWidget(self.max_length_label)
        self.predBox.addWidget(self.min_length_label)
        self.predBox.addWidget(self.fa_label)
        self.predBox.addWidget(self.dev_fa_label)
        self.predBox.addWidget(self.no_streamlines_label)
        self.predBox.addWidget(self.dev_no_streamlines_label)
        self.predBox.addWidget(self.no_voxels_label)
        self.predBox.addWidget(self.dev_no_voxels_label)

        ####### Adding Matplotlib
        self.main_widget = QtWidgets.QWidget(self)
        print(self.whole_brain_length_info_all)
        self.l = QtWidgets.QVBoxLayout(self.main_widget)
        self.sc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100, x=self.whole_brain_length_info_all)
        self.l.addWidget(self.sc)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.predBox.addWidget(self.main_widget)

        self.predBox.addStretch(2)


        self.statBox.setLayout(self.predBox)

    def VerticalLayout(self):

        # root =  tk.Tk()
        # helv36 = tkFont.Font(root, family = "Helvetica",size = 36,weight = "bold")
        # tkFont.families()

        
        self.vMain = Qt.QHBoxLayout() ## Added horizontal box layout

        self.groupBox1 = QGroupBox("Select Whole Brain") 
        whole_brain = []
        for (dirpath, dirnames, filenames) in os.walk("WholeBrains"):
            whole_brain.extend(filenames)
            break

        self.combo_box1 = QComboBox()
        self.combo_box1.addItems(whole_brain)
        button_box1 = QPushButton('Choose Directory')
        ## Creating font with QtFont
        # font = QtGui.QFont()
        # font.setFamily("Helvetica")
        # font.setBold(True)
        # button_box1.setFont(font)
        button_box2 = QPushButton("Load The Whole Brain")

        button_box2.clicked.connect(self.clickMethod)
        button_box1.clicked.connect(self.openFileNameDialog)
        
        ##### Adjusting Color and font 
        self.groupBox1.setStyleSheet("background-color: #006868; font-size: 13pt; font-weight: bold; font-family: Arial; color: white")
        self.combo_box1.setStyleSheet("QComboBox {background-color: #005151; font-size: 12pt; font-family: Courier; color: white}")
        button_box1.setStyleSheet("background-color: #00A0A0; font-size: 13pt; font-family: Arial; color: white")
        button_box2.setStyleSheet("background-color: #72d874; font-size: 13pt; font-family: Arial; color: white")
        
        vbox = Qt.QVBoxLayout()
        vbox.addWidget(self.combo_box1)
        vbox.addWidget(button_box1)
        vbox.addWidget(button_box2)
        vbox.addStretch(1)

        self.groupBox1.setLayout(vbox)

        #### Trying to add another GroupBox
        self.createTrainingBox = QGroupBox("Make Your Own Training Set") 
        tract_files = []
        for (dirpath, dirnames, filenames) in os.walk("Tracts/AF/Left"):
            tract_files.extend(filenames)
            break

        self.training_tract_combo = QComboBox()
        self.training_tract_combo.addItems(tract_files)
        training_tract_button = QPushButton('Choose Your Tract')
        training_button = QPushButton('Train Me!')

        self.createTrainingBox.setStyleSheet("background-color: #006868; font-size: 13pt; font-weight: bold; font-family: Arial; color: white")
        self.training_tract_combo.setStyleSheet("QComboBox {background-color: #005151; font-size: 12pt; font-family: Courier; color: white}")
        training_tract_button.setStyleSheet("background-color: #00A0A0; font-size: 13pt; font-family: Arial; color: white")
        training_button.setStyleSheet("background-color: #00A0AA; font-size: 13pt; font-family: Arial; color: white")


        #### Method Radio Button 
        method_label = QLabel()
        method_label.setText("\nSelect a method: ")
        self.ocsvm_radio = QRadioButton("One Class SVM")
        self.ocsvm_radio.setChecked(True)
        self.kd_radio = QRadioButton("KD Tree")
        button = QPushButton("Finish!")
        button.setStyleSheet("background-color: #20a949; font-size: 13pt; font-family: Arial; color: white")

        button.clicked.connect(self.clickMethod2)


        training_tract_box = Qt.QVBoxLayout()
        training_tract_box.addWidget(self.training_tract_combo)
        training_tract_box.addWidget(training_tract_button)
        training_tract_box.addWidget(training_button)
        
        training_tract_box.addWidget(method_label)
        training_tract_box.addWidget(self.ocsvm_radio)
        training_tract_box.addWidget(self.kd_radio)
        training_tract_box.addWidget(button)
        training_tract_box.addStretch(1)

        ##################################
        self.createTrainingBox.setLayout(training_tract_box)

        ####################################################
        groupBox3 = QGroupBox("Use Pre-Trained Tracts")
        groupBox3.setStyleSheet("background-color: #006868")
            
        self.tractsName = ["AF", "CG", "UF"]
        self.tract_box = QPushButton("Done!")
        self.tract_box.clicked.connect(self.openTractFilesDialog)
        
        self.combo = QComboBox()
        self.combo.addItems(self.tractsName)
        self.radio1 = QRadioButton("Left tract")
        self.radio1.setChecked(True)
        self.radio2 = QRadioButton("Right tract")
        self.radio3 = QRadioButton("None")
        
        groupBox3.setStyleSheet("background-color: #006868; font-size: 13pt; font-weight:bold; font-family: Arial; color: white")
        self.tract_box.setStyleSheet("background-color: #00A82d; font-size: 13pt; font-family: Arial; color: white")
        self.combo.setStyleSheet("background-color: #005151; font-size: 13pt; font-family: Arial; color: white")
        self.radio1.setStyleSheet("background-color: #006868; font-size: 12pt; font-family: Arial; color: white")
        self.radio2.setStyleSheet("background-color: #006868; font-size: 12pt; font-family: Arial; color: white")
        self.radio3.setStyleSheet("background-color: #006868; font-size: 12pt; font-family: Arial; color: white")

        

        vbox3 = Qt.QVBoxLayout()
        vbox3.addWidget(self.combo)
        vbox3.addWidget(self.radio1)
        vbox3.addWidget(self.radio2)
        vbox3.addWidget(self.radio3)
        vbox3.addWidget(self.tract_box)
        
        vbox3.addStretch(1)
        groupBox3.setLayout(vbox3)

        self.vMain.addWidget(self.groupBox1)
        self.vMain.addWidget(groupBox3)
        self.vMain.addWidget(self.createTrainingBox)
        #verbox.addLayout(self.vMain)
    
    def __init__(self, parent = None):
        
        Qt.QMainWindow.__init__(self, parent)

        ####################### menubar###############
        title = "FastTractSeg"
        ########## Took screen width and height to adjust in any screen ##########
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        #print(width, height)
        top = width // 6
        left = height // 6
        width = (width // 6) * 4
        height = (height // 6) * 4
        ######## Initialized hist plot for whole brain
        self.whole_brain_length_info_all = np.random.normal(size = 1000)
        xw = self.whole_brain_length_info_all
        self.listener = False
        # menubar = self.menuBar()
        # fileMenu = menubar.addMenu('File ')
        # newAct = QAction('New', self)                
        # fileMenu.addAction(newAct)
        ##### Set your icon            
        icon_path = './Tract-Segmentation-Visualizer/icon.png'
        if (QtCore.QFile.exists( icon_path )):
            self.setWindowIcon(QIcon(icon_path))
            #self.setIconSize(QtCore.QSize(0,0))
        self.setWindowTitle(title)
        self.setGeometry(top,left, width, height)
        
        # Set window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        print(QtCore.Qt.black)
        p.setColor(self.backgroundRole(), QtCore.Qt.darkCyan) ## Set the background color
        self.setPalette(p)     
        ####################### Pyqt############### 


        ########## frame##################


        self.frame = Qt.QFrame()
        
        
        self.VerticalLayout()
        self.HorizontalLayout()
        

        self.vl = Qt.QVBoxLayout()

        self.vl.addLayout(self.vMain)
        self.vl.addWidget(self.groupBox2)
        
        
        ###### Adding Horizontal Box
        self.vbox = Qt.QHBoxLayout()
        self.vbox.addLayout(self.vl)
        self.vbox.addWidget(self.statBox)

        self.frame.setLayout(self.vbox)
        self.setCentralWidget(self.frame)
        
        self.setLayout(self.vl)
        self.show()


if __name__ == "__main__":
    
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.exec()
    #sys.exit(app.exec_())