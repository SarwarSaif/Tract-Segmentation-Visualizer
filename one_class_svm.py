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
from preprocessing import preprocessing as pre

class ocsvm:
        
    def oneClassSVM(self, whole_brain_id, model_tracts):
        self.no_of_points=12    
        self.leafsize=10
        
        ################################ Train Data ######################################
        print ("Preparing Train Data")
        resample_tract_train, train_data= pre.CreateModelTracts_for_SVM(self, model_tracts)    
    
        ###################### Test Data################################
        testTarget = "161731"
        testTarget_brain = "full1M_"+testTarget+".trk"
        
        #whole_brain_id = "161731"
        testTarget_brain = "full1M_"+whole_brain_id+".trk"
        
        t0=time.time()
        resample_tract_test, test_data= pre.create_test_data_set(self, whole_brain_id)
        #trueTract= self.load(whole_brain_id + tract)  
        t1=t0-time.time()
        
        """###########################one class SVM######################"""
        t2=time.time()
        gamma_value = 0.001
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma_value)
        clf.fit(resample_tract_train)
        """ linear poly rbf """
        """#########################################"""
    
        x_pred_train = clf.predict(resample_tract_train.tolist())
        n_error_test = x_pred_train[x_pred_train==-1].size
        print('number of error for training =', n_error_test)    
    
    
        x_pred_test=clf.predict(resample_tract_test.tolist())    
        n_error_test = x_pred_test[x_pred_test==-1].size
        print('number of error for testing=',n_error_test)
    
    
        ########################### visualize tract ######################
        test_data=np.array(test_data)
        segmented_tract_positive= test_data[np.where(x_pred_test==1)]
        segmented_tract_negative= test_data[np.where(x_pred_test==-1)]
    
        return segmented_tract_positive, segmented_tract_negative
    