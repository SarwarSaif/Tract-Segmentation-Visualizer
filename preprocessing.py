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

# from PyQt5.QtWidgets import QLabel,QRadioButton,QComboBox,QPushButton, QAction, QMenu, QApplication, QFileDialog, QWidget, QSplitter, QGroupBox
# from PyQt5.QtGui import QIcon
# import PyQt5.QtGui

# from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
# from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# from dipy.tracking.streamline import transform_streamlines, length

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

class preprocessing:
    
    def load(self, filename):
        """Load tractogram from TRK file 
        """
        print("Print hos nai naki! ",filename)
        wholeTract= nib.streamlines.load(filename)  
        wholeTract = wholeTract.streamlines
        return  wholeTract    
    def resample(self, streamlines, no_of_points):
        """Resample streamlines using 12 points and also flatten the streamlines
        """
        return np.array([set_number_of_points(s, no_of_points).ravel() for s in streamlines])     
    def CreateModelTracts_for_SVM(self, filepath):
        
        train_data=[]
        #tract_files = brain_id.split(',')
        print("Ekhn subject print korbo! ",len(filepath), filepath)
        for tract_files in filepath: 
            
        #     print (sub)        
        #     T_filename=sub #+tract_name
        #     print("Morar filename: ",T_filename)
            print("Brain id ",tract_files)
            wholeTract = preprocessing.load (self, tract_files)       
            train_data=np.concatenate((train_data, wholeTract),axis=0) 
    
        print ("train data Shape") 
        resample_tract= preprocessing.resample(self, train_data,no_of_points=self.no_of_points)
   
        return resample_tract, train_data
    def create_test_data_set(self, testTarget_brain):    
    
        print ("Preparing Test Data")    
        t_filename=testTarget_brain #"124422_af.left.trk"    
        test_data= preprocessing.load(self, t_filename)  
        resample_tractogram= preprocessing.resample(self, test_data,no_of_points= self.no_of_points)

        return resample_tractogram, test_data 