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

class kd2:
    
    
    def build_kdtree(self, points, leafsize):
        """Build kdtree with resample streamlines 
        """
        return KDTree(points,leafsize =leafsize)        
    def kdtree_query(self, tract,kd_tree):
        """compute 1 NN using kdtree query and return the id of NN
        """
         
        dist_kdtree, ind_kdtree = kd_tree.query(tract, k=1)
        return np.hstack(ind_kdtree) 
    def segmentation_with_NN(self, filename_tractogram, filename_example_tract, no_of_points, leafsize):

        """Nearest Neighbour applied for bundle segmentation 
        """   
        #load tractogram
        print("Loading tractogram: %s" %filename_tractogram)
        tractogram= pre.load(self, filename_tractogram) 
        
        #load tract
        print("Loading example tract: %s" %filename_example_tract)
        resample_tract, train_data= pre.CreateModelTracts_for_SVM(self, filename_example_tract)    
        #tract=self.load(filename_example_tract) 
    
        t0=time.time()
        #resample whole tractogram
        print("Resampling tractogram........" )
        resample_tractogram = pre.resample(self, tractogram,no_of_points=no_of_points)
        
        #resample example tract
        #print("Resampling example tract.......")
        #resample_tract= self.resample(tract,no_of_points=no_of_points)
    
        #build kdtree
        print("Buildng kdtree")
        kd_tree= kd2.build_kdtree (self, resample_tractogram, leafsize=leafsize)
    
        #kdtree query to retrive the NN id
        query_idx= kd2.kdtree_query(self, resample_tract, kd_tree)
    
        #extract the streamline from tractogram
        estimated_tract=tractogram[query_idx] 
        estimated_tract_neg= np.delete(tractogram, query_idx)
        print(estimated_tract)
        print("Total amount of time to segment the bundle is %f seconds" % (time.time()-t0))   
        return  estimated_tract, estimated_tract_neg

        