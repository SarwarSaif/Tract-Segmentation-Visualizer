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
import kd_tree_segmentation as kdts
from kd_tree_segmentation import kd2 
import tkinter.font as tkFont
import tkinter as tk
#from kd_tree_segmentation.kd2 import 
interactive = False  # set to True to show the interactive display window

class MainWindow(Qt.QMainWindow):

    def fa (self,filename):
        FA_img = nib.load(filename)
        FA_data = FA_img.get_data()    
        FA_data=np.array(FA_data)    
        return FA_data
    def length_info(self,tract):
    
        lengths = np.array(list(length(tract)))
        return lengths.max(),lengths.min(),np.average(lengths)        
    def voxel_count(self, tract ):
        #affine=utils.affine_for_trackvis(voxel_size=np.array([2,2,2]))
        affine=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
        return len(streamline_mapping(tract, affine=affine).keys())
    
    
    

        
    
    def load_streamline2(self,fileName,fileName2):
        from fury import window
        scene = window.Scene()
        self.no_of_points=12    
        self.leafsize=10
        whole_brain_id = "100307"
        #model_tracts =[ "124422", "111312",  "100408", "100307", "856766"]
        self.tract = "_af.left.trk"
        test_data= self.load(fileName) 
        filename_tractogram = fileName
        print("Ki shomossha! ",fileName," 2 ",fileName2)
        if self.method == 1:
            print(fileName2)
            segmented_tract_positive, segmented_tract_negative = kd2.oneClassSVM(self, filename_tractogram, fileName2, self.tract)
            color_tract = colors.blue
        else:
            segmented_tract_positive, segmented_tract_negative = kd2.segmentation_with_NN(self, filename_tractogram, fileName2, self.no_of_points, self.leafsize, self.tract)
            color_tract = colors.green
        
        
        affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
        
        
        ########### Statistical Analysis ################
        whole_brain_length_info= np.array(self.length_info (test_data))
        max_label = "\nMaximum length of tracts: " + str(whole_brain_length_info[0])
        min_label = "\nMinimum length of tracts: "+  str(whole_brain_length_info[1])
        streamline_count_label = "\nNo of streamlines: "+  str(len(test_data))
        voxel_count_label = "\nNo of voxels:: "+  str(self.voxel_count(test_data))
        #dev_streamline_count_label = "\nNo of streamlines: "+  str(len(test_data))
        #dev_voxel_count_label = "\nNo of voxels:: "+  str(self.voxel_count(test_data))
        
        self.max_length_label.setText(max_label)
        self.min_length_label.setText(min_label)
        self.no_streamlines_label.setText(streamline_count_label)
        #self.dev_no_streamlines_label.setText(dev_streamline_count_label)
        self.no_voxels_label.setText(voxel_count_label)
        #self.dev_no_voxels_label.setText(dev_voxel_count_label)
        
        
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
    
    def load_streamline(self,fileName):
        
        print(fileName)
        global wholeTract
        from fury import window
        scene = window.Scene()
        
        affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))

        wholeTract= nib.streamlines.load(fileName)
        wholeTract = wholeTract.streamlines
        print(wholeTract)        
        wholeTract_transform = transform_streamlines(wholeTract, np.linalg.inv(affine))
        stream_actor = actor.line(wholeTract_transform, colors=colors.white,
                           opacity=0.02, linewidth=0.1)

        #scene.set_camera(position=(-176.42, 118.52, 128.20),
        #         focal_point=(113.30, 128.31, 76.56),
        #         view_up=(0.18, 0.00, 0.98))   
        scene.add(stream_actor)
        
        self.vtkWidget.GetRenderWindow().AddRenderer(scene)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        self.iren.Start()
    
    
    def load(self, filename):
        """Load tractogram from TRK file 
        """
        wholeTract= nib.streamlines.load(filename)  
        wholeTract = wholeTract.streamlines
        return  wholeTract    

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
        self.load_streamline(whole_brain)
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
        self.groupBox2.setStyleSheet("background-color: #EAEAEA")
            
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
        self.predBox = Qt.QVBoxLayout()
        
        self.max_length_label = QLabel()
        self.max_length_label.setText("\nMaximum length of tracts: ")
        self.min_length_label = QLabel()
        self.min_length_label.setText("\nMinimum length of tracts: ")
        self.fa_label = QLabel()
        self.fa_label.setText("\nFA value: ")
        self.dev_fa_label = QLabel()
        self.dev_fa_label.setText("\nDeviation FA value: ")
        self.no_streamlines_label = QLabel()
        self.no_streamlines_label.setText("\nNo of streamlines: ")
        self.dev_no_streamlines_label = QLabel()
        self.dev_no_streamlines_label.setText("\nDeviation No of streamlines: ")
        self.no_voxels_label = QLabel()
        self.no_voxels_label.setText("\nNo of voxels: ")
        self.dev_no_voxels_label = QLabel()
        self.dev_no_voxels_label.setText("\nDeviation No of voxels: ")
        
        self.predBox.addWidget(self.max_length_label)
        self.predBox.addWidget(self.min_length_label)
        self.predBox.addWidget(self.fa_label)
        self.predBox.addWidget(self.dev_fa_label)
        self.predBox.addWidget(self.no_streamlines_label)
        self.predBox.addWidget(self.dev_no_streamlines_label)
        self.predBox.addWidget(self.no_voxels_label)
        self.predBox.addWidget(self.dev_no_voxels_label)
        self.predBox.addStretch(2)
        
        self.statBox.setLayout(self.predBox)

    def VerticalLayout(self):

        # root =  tk.Tk()
        # helv36 = tkFont.Font(root, family = "Helvetica",size = 36,weight = "bold")
        # tkFont.families()

        
        self.vMain = Qt.QHBoxLayout() ## Added horizontal box layout
        groupBox1 = QGroupBox("Select Whole Brain") 
        whole_brain = []
        for (dirpath, dirnames, filenames) in os.walk("WholeBrains"):
            whole_brain.extend(filenames)
            break

        self.combo_box1 = QComboBox()
        self.combo_box1.addItems(whole_brain)
        button_box1 = QPushButton('Add')
        ## Creating font with QtFont
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setBold(True)
        button_box1.setFont(font)
        button_box2 = QPushButton("Okay")

        button_box2.clicked.connect(self.clickMethod)
        button_box1.clicked.connect(self.openFileNameDialog)
        
        ##### Adding Color 
        groupBox1.setStyleSheet("background-color: #EAEAEA")
        self.combo_box1.setStyleSheet("background-color: #99FFEF")
        button_box1.setStyleSheet("background-color: #00A0A0")
        button_box2.setStyleSheet("background-color: #72d874")
        
        vbox = Qt.QVBoxLayout()
        vbox.addWidget(self.combo_box1)
        vbox.addWidget(button_box1)
        vbox.addWidget(button_box2)
        vbox.addStretch(1)


        groupBox1.setLayout(vbox)
        ####################################################
        groupBox3 = QGroupBox("Select Tract")
        groupBox3.setStyleSheet("background-color: #EAEAEA")
            
        self.tractsName = ["AF", "CG", "UF"]
        self.tract_box = QPushButton("Add Tracts")
        self.tract_box.clicked.connect(self.openTractFilesDialog)
        
        self.combo = QComboBox()
        self.combo.addItems(self.tractsName)
        self.radio1 = QRadioButton("Left tract")
        self.radio1.setChecked(True)
        self.radio2 = QRadioButton("Right tract")
        self.radio3 = QRadioButton("None")
        
        #### Method Radio Button 
        self.ocsvm_radio = QRadioButton("One Class SVM")
        self.ocsvm_radio.setChecked(True)
        self.kd_radio = QRadioButton("Nearest Neighbour (KD Tree)")
        method_label = QLabel()
        method_label.setText("\nSelect a method: ")
        
        #label = QLabel()
        #label.setText("\nKD tree K value :")

        #k = ['1', '2', '3', '4']
        #comboForK = QComboBox()
        #comboForK.addItems(k)

        button = QPushButton("Okay")

        button.clicked.connect(self.clickMethod2)

        vbox3 = Qt.QVBoxLayout()
        vbox3.addWidget(self.combo)
        vbox3.addWidget(self.radio1)
        vbox3.addWidget(self.radio2)
        vbox3.addWidget(self.radio3)
        vbox3.addWidget(self.tract_box)
        
        vbox3.addWidget(method_label)
        vbox3.addWidget(self.ocsvm_radio)
        vbox3.addWidget(self.kd_radio)
        #vbox3.addWidget(label)
        #vbox3.addWidget(comboForK)
        vbox3.addWidget(button)
        vbox3.addStretch(1)
        groupBox3.setLayout(vbox3)

        self.vMain.addWidget(groupBox1)
        self.vMain.addWidget(groupBox3)
        #verbox.addLayout(self.vMain)

    def __init__(self, parent = None):
        
        Qt.QMainWindow.__init__(self, parent)

        ####################### menubar###############
        title = "FastTractSeg"
#        top = 400
#        left = 400
#        width = 900
#        height = 500
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File ')
        newAct = QAction('New', self)                
        fileMenu.addAction(newAct)
        ##newAct.triggered.connect(self.openFileNameDialog)
        self.setWindowIcon(QIcon("download.jpg"))
        self.setWindowTitle(title)
        #self.setGeometry(top,left, width, height)
        
        # Set window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
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
    app.exec_()
    #sys.exit(app.exec_())