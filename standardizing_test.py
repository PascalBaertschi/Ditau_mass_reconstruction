import numpy
numpy.random.seed(1337)
import pandas
import ROOT
import random
import time
import math
import sys
import csv
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout,Activation
#from keras.layers.advanced_activations import LeakyReLU
from sklearn.ensemble import RandomForestRegressor
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from rep.estimators import xgboost
from sklearn import preprocessing
#ROOT.gSystem.Load("libSVfitStandaloneAlgorithm")


###inclnu dataset only for proof ###################
dataframe_mass_inclnu= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_inclnu = dataframe_mass_inclnu.values
#dataset_mass_inclnu[:,0:6] = preprocessing.scale(dataset_mass_inclnu[:,0:6])

mass_input_inclnu = dataset_mass_inclnu[:,0:8]
mass_output_inclnu = dataset_mass_inclnu[:,8]
inclnu_length = len(mass_output_inclnu)
inclnu_train_length = int(round(inclnu_length*0.7))
inclnu_test_length = int(round(inclnu_length*0.3))
print "inclnu length:",inclnu_length

mass_input_inclnu_para = []
ditaumass_inclnu = []
test = []
for j in range(0,len(mass_output_inclnu)):
    px_vis = dataset_mass_inclnu[j,0]
    py_vis = dataset_mass_inclnu[j,1]
    pz_vis = dataset_mass_inclnu[j,2]
    energy_vis = dataset_mass_inclnu[j,3]
    px_nu = dataset_mass_inclnu[j,4]
    py_nu = dataset_mass_inclnu[j,5]
    pz_nu = dataset_mass_inclnu[j,6]
    energy_nu = dataset_mass_inclnu[j,7]
    ditaumass = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).M()
    vismass = ROOT.TLorentzVector(px_vis,py_vis,pz_vis,energy_vis).M()
    p_vis = ROOT.TLorentzVector(px_vis,py_vis,pz_vis,energy_vis).P()
    mt = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).Mt2()
    pt = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).Pt()
    px = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).Px()
    py = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).Py()
    pt_cal = numpy.sqrt(px**2+py**2)
    phi = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).Phi()
    px_cal = pt*numpy.cos(phi)
    py_cal = pt*numpy.sin(phi)
    test.append([px,py,px_cal,py_cal])
    mass_no_pz = energy_vis**2-pz_vis**2-pt**2 
    #mass_input_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,pz_nu,energy_nu,p_vis,vismass])
    #mass_input_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,vismass,pt,p_vis,mass_no_pz])
    mass_input_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,p_vis,vismass])
    ditaumass_inclnu.append(ditaumass)
mass_input_inclnu_para = numpy.array(mass_input_inclnu_para,numpy.float64)
mass_input_inclnu_para_test1 = mass_input_inclnu_para[0:10,:]
mass_input_inclnu_para_test2 = mass_input_inclnu_para[:,:]
mass_input_inclnu_para_test3 = mass_input_inclnu_para[:,:]



"""

print "STANDARDIZING NEW METHOD [:,:]"
mass_input_inclnu_para_test1[:,:] = preprocessing.scale(mass_input_inclnu_para_test1[:,:])

print numpy.mean(mass_input_inclnu_para_test1[:,0])
print numpy.mean(mass_input_inclnu_para_test1[:,1])
print numpy.mean(mass_input_inclnu_para_test1[:,2])
print numpy.mean(mass_input_inclnu_para_test1[:,3])
print numpy.mean(mass_input_inclnu_para_test1[:,4])
print numpy.mean(mass_input_inclnu_para_test1[:,5])
print numpy.mean(mass_input_inclnu_para_test1[:,6])
print numpy.mean(mass_input_inclnu_para_test1[:,7])

print "STANDARDIZING NEW METHOD"
mass_input_inclnu_para_test2 = preprocessing.scale(mass_input_inclnu_para_test2)

print numpy.mean(mass_input_inclnu_para_test2[:,0])
print numpy.mean(mass_input_inclnu_para_test2[:,1])
print numpy.mean(mass_input_inclnu_para_test2[:,2])
print numpy.mean(mass_input_inclnu_para_test2[:,3])
print numpy.mean(mass_input_inclnu_para_test2[:,4])
print numpy.mean(mass_input_inclnu_para_test2[:,5])
print numpy.mean(mass_input_inclnu_para_test2[:,6])
print numpy.mean(mass_input_inclnu_para_test2[:,7])

print "STANDARDIZING OLD METHOD [:,0:8]"
mass_input_inclnu_para_test3[:,0:8] = preprocessing.scale(mass_input_inclnu_para_test3[:,0:8])


print numpy.mean(mass_input_inclnu_para_test3[:,0])
print numpy.mean(mass_input_inclnu_para_test3[:,1])
print numpy.mean(mass_input_inclnu_para_test3[:,2])
print numpy.mean(mass_input_inclnu_para_test3[:,3])
print numpy.mean(mass_input_inclnu_para_test3[:,4])
print numpy.mean(mass_input_inclnu_para_test3[:,5])
print numpy.mean(mass_input_inclnu_para_test3[:,6])
print numpy.mean(mass_input_inclnu_para_test3[:,7])
"""
mass_input_inclnu_para[:,0:8] = preprocessing.scale(mass_input_inclnu_para[:,0:8])
dataset_mass_inclnu[:,0:8] = preprocessing.scale(dataset_mass_inclnu[:,0:8])
train_input_inclnu_para = mass_input_inclnu_para[0:inclnu_train_length,0:8]
test_input_inclnu_para = mass_input_inclnu_para[inclnu_train_length:inclnu_length,0:8]
train_output_inclnu = dataset_mass_inclnu[0:inclnu_train_length,8]
test_output_inclnu = dataset_mass_inclnu[inclnu_train_length:inclnu_length,8]
test_ditaumass_inclnu = ditaumass_inclnu[inclnu_train_length:inclnu_length]
overfit_inclnu_para = train_input_inclnu_para[0:len(test_output_inclnu),:]

mass_input_inclnu_para_test = mass_input_inclnu_para[0:10,:]
mass_input_inclnu_test = mass_input_inclnu[0:10,:]
"""
print "no Prescaling"
print mass_input_inclnu_test
#print mass_input_inclnu_para_test
"""
