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

class kd2:
    
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
    def CreateModelTracts_for_SVM(self, filepath, tract_name):
        
        train_data=[]
        #tract_files = brain_id.split(',')
        print("Ekhn subject print korbo! ",len(filepath), filepath)
        for tract_files in filepath: 
            
        #     print (sub)        
        #     T_filename=sub #+tract_name
        #     print("Morar filename: ",T_filename)
            print("Brain id ",tract_files)
            wholeTract = kd2.load (self, tract_files)       
            train_data=np.concatenate((train_data, wholeTract),axis=0) 
    
        print ("train data Shape") 
        resample_tract= kd2.resample(self, train_data,no_of_points=self.no_of_points)
   
        return resample_tract, train_data
    def create_test_data_set(self, testTarget_brain):    
    
        print ("Preparing Test Data")    
        t_filename=testTarget_brain #"124422_af.left.trk"    
        test_data= kd2.load(self, t_filename)  
        resample_tractogram= kd2.resample(self, test_data,no_of_points= self.no_of_points)

        return resample_tractogram, test_data 
        
    def oneClassSVM(self, whole_brain_id, model_tracts, tract):
        self.no_of_points=12    
        self.leafsize=10
        
        ################################ Train Data ######################################
        print ("Preparing Train Data")
        resample_tract_train, train_data= kd2.CreateModelTracts_for_SVM(self, model_tracts, tract)    
    
        ###################### Test Data################################
        testTarget = "161731"
        testTarget_brain = "full1M_"+testTarget+".trk"
        
        #whole_brain_id = "161731"
        testTarget_brain = "full1M_"+whole_brain_id+".trk"
        
        t0=time.time()
        resample_tract_test, test_data= kd2.create_test_data_set(self, whole_brain_id)
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
    
    
    def build_kdtree(self, points, leafsize):
        """Build kdtree with resample streamlines 
        """
        return KDTree(points,leafsize =leafsize)        
    def kdtree_query(self, tract,kd_tree):
        """compute 1 NN using kdtree query and return the id of NN
        """
         
        dist_kdtree, ind_kdtree = kd_tree.query(tract, k=1)
        return np.hstack(ind_kdtree) 
    def segmentation_with_NN(self, filename_tractogram, filename_example_tract, no_of_points, leafsize, tract):

        """Nearest Neighbour applied for bundle segmentation 
        """   
        #load tractogram
        print("Loading tractogram: %s" %filename_tractogram)
        tractogram= kd2.load(self, filename_tractogram) 
        
        #load tract
        print("Loading example tract: %s" %filename_example_tract)
        resample_tract, train_data= kd2.CreateModelTracts_for_SVM(self, filename_example_tract, tract)    
        #tract=self.load(filename_example_tract) 
    
        t0=time.time()
        #resample whole tractogram
        print("Resampling tractogram........" )
        resample_tractogram = kd2.resample(self, tractogram,no_of_points=no_of_points)
        
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

        