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
from one_class_svm import ocsvm
import tkinter.font as tkFont
import tkinter as tk

class stat:

    def fa (self,filename):
        FA_img = nib.load(filename)
        FA_data = FA_img.get_data()    
        FA_data=np.array(FA_data)    
        return FA_data
    def length_info(self,tract):
    
        lengths = np.array(list(length(tract)))
        return lengths.max(),lengths.min(),np.average(lengths)        
    def length_info_all(self,tract):
    
        lengths = np.array(list(length(tract)))
        return lengths       

    def voxel_count(self, tract ):
        #affine=utils.affine_for_trackvis(voxel_size=np.array([2,2,2]))
        affine=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
        return len(streamline_mapping(tract, affine=affine).keys())
    
    