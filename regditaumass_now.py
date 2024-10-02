import numpy
numpy.random.seed(1337)
import pandas
import ROOT
import random
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
from ctypes import cdll
#load dataset housing
dataframe_housing = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/housing.csv", delim_whitespace=True, header=None)
dataset_housing = dataframe_housing.values

#load dataset hadronic decays
dataframe_mass_train = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train = dataframe_mass_train.values
dataframe_mass_test= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test = dataframe_mass_test.values
dataframe_mass= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass = dataframe_mass.values

dataframe_mass_train_total_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_total_had = dataframe_mass_train_total_had.values
dataframe_mass_test_total_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_total_had = dataframe_mass_test_total_had.values
dataframe_mass_total_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_total_had = dataframe_mass_total_had.values

dataframe_mass_train_inclnu_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_inclnu_had = dataframe_mass_train_inclnu_had.values
dataframe_mass_test_inclnu_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_inclnu_had = dataframe_mass_test_inclnu_had.values
dataframe_mass_inclnu_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_inclnu_had = dataframe_mass_inclnu_had.values


dataframe_vismass_train_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_vismass_had_1e6_new.csv",delim_whitespace=False,header=None)
dataset_vismass_train_had = dataframe_vismass_train_had.values
dataframe_vismass_test_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_vismass_had_1e6_new.csv",delim_whitespace=False,header=None)
dataset_vismass_test_had = dataframe_vismass_test_had.values


#load dataset all decays
dataframe_mass_train_all = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_all = dataframe_mass_train_all.values
dataframe_mass_test_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_all = dataframe_mass_test_all.values
dataframe_mass_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_all = dataframe_mass_all.values

dataframe_mass_train_inclnu = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_inclnu = dataframe_mass_train_inclnu.values
dataframe_mass_test_inclnu= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_inclnu = dataframe_mass_test_inclnu.values
dataframe_mass_inclnu= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_inclnu = dataframe_mass_inclnu.values

dataframe_mass_train_total = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
#dataframe_mass_train_total = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_total_3e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_total = dataframe_mass_train_total.values
dataframe_mass_test_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
#dataframe_mass_test_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_total_3e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_total = dataframe_mass_test_total.values
dataframe_mass_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
#dataframe_mass_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_total_3e6.csv",delim_whitespace=False,header=None)
dataset_mass_total = dataframe_mass_total.values

dataframe_train_theta = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_train_theta = dataframe_train_theta.values
dataframe_test_theta= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_test_theta = dataframe_test_theta.values
dataframe_theta= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_theta = dataframe_theta.values

dataframe_vismass_train_all = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_vismass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_vismass_train_all = dataframe_vismass_train_all.values
dataframe_vismass_test_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_vismass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_vismass_test_all = dataframe_vismass_test_all.values




# split into input and output variables
#hadronic decays
train_input = dataset_mass_train[:,0:7]
train_output = dataset_mass_train[:,7]
test_input = dataset_mass_test[:,0:7]
test_output = dataset_mass_test[:,7]
mass_input = dataset_mass[:,0:7]
mass_output = dataset_mass[:,7]

train_input_inclnu_had = dataset_mass_train_inclnu_had[:,0:8]
train_output_inclnu_had = dataset_mass_train_inclnu_had[:,8]
test_input_inclnu_had = dataset_mass_test_inclnu_had[:,0:8]
test_output_inclnu_had = dataset_mass_test_inclnu_had[:,8]
mass_input_inclnu_had = dataset_mass_inclnu_had[:,0:8]
mass_output_inclnu_had = dataset_mass_inclnu_had[:,8]

train_input_total_had = dataset_mass_train_total_had[:,0:4]
train_output_total_had = dataset_mass_train_total_had[:,4]
test_input_total_had = dataset_mass_test_total_had[:,0:4]
test_output_total_had = dataset_mass_test_total_had[:,4]
mass_input_total_had = dataset_mass_total_had[:,0:4]
mass_output_total_had = dataset_mass_total_had[:,4]

train_input_vismass_had = dataset_vismass_train_had[:,0:4]
train_output_vismass_had = dataset_vismass_train_had[:,4]
test_input_vismass_had = dataset_vismass_test_had[:,0:4]
test_output_vismass_had = dataset_vismass_test_had[:,4]


#all decays
mass_input_all = dataset_mass_inclnu[:,0:6]
mass_output_all=dataset_mass_inclnu[:,8]
all_length = len(mass_output_all)
all_fixed_train_length = 300000
all_fixed_test_length = 100000
all_fixed_length = all_fixed_test_length+all_fixed_train_length
all_train_length = int(round(all_length*0.7))
all_test_length = int(round(all_length*0.3))

#dataset_mass_inclnu[:,0:6] = preprocessing.scale(dataset_mass_inclnu[:,0:6])
#train_input_all = dataset_mass_inclnu[0:all_fixed_train_length,0:6]
#train_output_all = dataset_mass_inclnu[0:all_fixed_train_length,8]
#test_input_all = dataset_mass_inclnu[all_fixed_train_length:all_fixed_length,0:6]
#test_output_all = dataset_mass_inclnu[all_fixed_train_length:all_fixed_length,8]

#train_input_all = dataset_mass_inclnu[0:all_train_length,0:6]
#train_output_all = dataset_mass_inclnu[0:all_train_length,8]
#test_input_all = dataset_mass_inclnu[all_train_length:all_length,0:6]
#test_output_all = dataset_mass_inclnu[all_train_length:all_length,8]

train_input_all = dataset_mass_train_inclnu[:,0:6]
train_output_all = dataset_mass_train_inclnu[:,8]
test_input_all = dataset_mass_test_inclnu[:,0:6]
test_output_all = dataset_mass_test_inclnu[:,8]
#overfit_all = train_input_all[0:len(test_output_all)]




###########################               data for inclnu                 #########################################

mass_input_inclnu = dataset_mass_inclnu[:,0:8]
mass_output_inclnu = dataset_mass_inclnu[:,8]
inclnu_length = len(mass_output_inclnu)
inclnu_train_length = int(round(inclnu_length*0.7))
inclnu_test_length = int(round(inclnu_length*0.3))
print "inlcnu length:",inclnu_length

mass_input_inclnu_para = []
for j in range(0,len(mass_output_inclnu)):
    px_vis = dataset_mass_inclnu[j,0]
    py_vis = dataset_mass_inclnu[j,1]
    pz_vis = dataset_mass_inclnu[j,2]
    energy_vis = dataset_mass_inclnu[j,3]
    px_nu = dataset_mass_inclnu[j,4]
    py_nu = dataset_mass_inclnu[j,5]
    pz_nu = dataset_mass_inclnu[j,6]
    energy_nu = dataset_mass_inclnu[j,7]
    vismass = ROOT.TLorentzVector(px_vis,py_vis,pz_vis,energy_vis).M()
    p_vis = ROOT.TLorentzVector(px_vis,py_vis,pz_vis,energy_vis).P()
    mt = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).Mt2()
    pt = ROOT.TLorentzVector(px_vis+px_nu,py_vis+py_nu,pz_vis+pz_nu,energy_vis+energy_nu).Pt()
    mass_no_pz = energy_vis**2-pz_vis**2-pt**2 
    #mass_input_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,pz_nu,energy_nu,p_vis,vismass])
    #mass_input_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,vismass,pt,p_vis,mass_no_pz])
    #mass_input_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,p_vis,vismass])
    mass_input_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu])
mass_input_inclnu_para = numpy.array(mass_input_inclnu_para,numpy.float64)

#mass_input_inclnu_para[:,0:8] = preprocessing.scale(mass_input_inclnu_para[:,0:8])
#dataset_mass_inclnu[:,0:8] = preprocessing.scale(dataset_mass_inclnu[:,0:8])

train_input_inclnu = dataset_mass_inclnu[0:inclnu_train_length,0:6]
test_input_inclnu = dataset_mass_inclnu[inclnu_train_length:inclnu_length,0:6]


train_input_inclnu_para = mass_input_inclnu_para[0:inclnu_train_length,0:8]
test_input_inclnu_para = mass_input_inclnu_para[inclnu_train_length:inclnu_length,0:8]


train_output_inclnu = dataset_mass_inclnu[0:inclnu_train_length,8]
test_output_inclnu = dataset_mass_inclnu[inclnu_train_length:inclnu_length,8]
overfit_inclnu_para = train_input_inclnu_para[0:len(test_output_inclnu),:]
test_inclnu_para = []

"""
for i in range(0,len(test_output_inclnu)):
    px_vis = test_input_inclnu_para[i,0]
    py_vis = test_input_inclnu_para[i,1]
    pz_vis = test_input_inclnu_para[i,2]
    energy_vis = test_input_inclnu_para[i,3]
    px_nu = test_input_inclnu_para[i,4]
    py_nu = test_input_inclnu_para[i,5]
    first_para = test_input_inclnu_para[i,6]
    second_para = test_input_inclnu_para[i,7]
    third_para = test_input_inclnu_para[i,8]
    mass = test_output_inclnu[i]
    test_inclnu_para.append([px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,first_para,second_para,third_para,mass])
test_inclnu_para = numpy.array(test_inclnu_para,numpy.float64)
"""
###########################                 data for total                   #################################


mass_input_total = dataset_mass_total[:,0:4]
mass_output_total = dataset_mass_total[:,4]
total_length = len(mass_output_total)
#total_train_length = 300000
#total_test_length = 100000
#total_length = total_train_length+total_test_length
total_train_length = int(round(total_length*0.7))
total_test_length = int(round(total_length*0.3))
total_diff_train_test_length = total_train_length-total_test_length


mass_input_total_para = []
for j in range(0,len(mass_output_total)):
    px = dataset_mass_total[j,0]
    py = dataset_mass_total[j,1]
    pz = dataset_mass_total[j,2]
    energy = dataset_mass_total[j,3]
    eta = ROOT.TLorentzVector(px,py,pz,energy).Eta()
    phi = ROOT.TLorentzVector(px,py,pz,energy).Phi()
    pt =  ROOT.TLorentzVector(px,py,pz,energy).Pt()
    p =   ROOT.TLorentzVector(px,py,pz,energy).P()
    mt = ROOT.TLorentzVector(px,py,pz,energy).Mt2()
    #mass_input_total_para.append([px,py,pz,energy,mt])
    #mass_input_total_para.append([px,py,pz,energy,px*px,py*py,pz*pz,energy*energy])
    mass_input_total_para.append([px,py,pz,energy])

mass_input_total_para = numpy.array(mass_input_total_para,numpy.float64)
#dataset_mass_total[:,0:4] = preprocessing.scale(dataset_mass_total[:,0:4])
#mass_input_total_para[:,0:5] = preprocessing.scale(mass_input_total_para[:,0:5]) 
"""
print "px mean:",numpy.mean(mass_input_total_para[:,0]), "std:",numpy.std(mass_input_total_para[:,0])
print "py mean:",numpy.mean(mass_input_total_para[:,1]), "std:",numpy.std(mass_input_total_para[:,1])
print "pz mean:",numpy.mean(mass_input_total_para[:,2]), "std:",numpy.std(mass_input_total_para[:,2])
print "energy mean:",numpy.mean(mass_input_total_para[:,3]), "std:",numpy.std(mass_input_total_para[:,3])
print "eta mean:",numpy.mean(mass_input_total_para[:,4]), "std:",numpy.std(mass_input_total_para[:,4])
print "phi mean:",numpy.mean(mass_input_total_para[:,5]), "std:",numpy.std(mass_input_total_para[:,5])
print "pt mean:",numpy.mean(mass_input_total_para[:,6]), "std:",numpy.std(mass_input_total_para[:,6])
print "p mean:",numpy.mean(mass_input_total_para[:,7]), "std:",numpy.std(mass_input_total_para[:,7])
"""
train_input_total = dataset_mass_train_total[:,0:4]
train_output_total = dataset_mass_train_total[:,4]
test_input_total = dataset_mass_test_total[:,0:4]
test_output_total = dataset_mass_test_total[:,4]
#train_input_total = dataset_mass_total[0:total_train_length,0:4]
#train_output_total = dataset_mass_total[0:total_train_length,4]
#test_input_total = dataset_mass_total[total_train_length:total_length,0:4]
#test_output_total = dataset_mass_total[total_train_length:total_length,4]
#train_input_total = dataset_mass_total[0:total_train_length,0:4]
train_input_total_para = mass_input_total_para[0:total_train_length,0:5]
#train_output_total = dataset_mass_total[0:total_train_length,4]
#test_input_total = dataset_mass_total[total_train_length:total_length,0:4]
test_input_total_para = mass_input_total_para[total_train_length:total_length,0:5]
#test_output_total = dataset_mass_total[total_train_length:total_length,4]
overfit_total = train_input_total[0:len(test_output_total),:]
overfit_total_para = train_input_total_para[0:len(test_output_total),:]
#overfit_total = train_input_total[total_diff_train_test_length:,:]

"""
print "TRAIN SAMPLE"
print "px mean:",numpy.mean(train_input_total_para[:,0]), "std:",numpy.std(train_input_total_para[:,0])
print "py mean:",numpy.mean(train_input_total_para[:,1]), "std:",numpy.std(train_input_total_para[:,1])
print "pz mean:",numpy.mean(train_input_total_para[:,2]), "std:",numpy.std(train_input_total_para[:,2])
print "energy mean:",numpy.mean(train_input_total_para[:,3]), "std:",numpy.std(train_input_total_para[:,3])
print "eta mean:",numpy.mean(train_input_total_para[:,4]), "std:",numpy.std(train_input_total_para[:,4])
print "phi mean:",numpy.mean(train_input_total_para[:,5]), "std:",numpy.std(train_input_total_para[:,5])
print "pt mean:",numpy.mean(train_input_total_para[:,6]), "std:",numpy.std(train_input_total_para[:,6])
print "p mean:",numpy.mean(train_input_total_para[:,7]), "std:",numpy.std(train_input_total_para[:,7])

print "TEST SAMPLE"
print "px mean:",numpy.mean(test_input_total_para[:,0]), "std:",numpy.std(test_input_total_para[:,0])
print "py mean:",numpy.mean(test_input_total_para[:,1]), "std:",numpy.std(test_input_total_para[:,1])
print "pz mean:",numpy.mean(test_input_total_para[:,2]), "std:",numpy.std(test_input_total_para[:,2])
print "energy mean:",numpy.mean(test_input_total_para[:,3]), "std:",numpy.std(test_input_total_para[:,3])
print "eta mean:",numpy.mean(test_input_total_para[:,4]), "std:",numpy.std(test_input_total_para[:,4])
print "phi mean:",numpy.mean(test_input_total_para[:,5]), "std:",numpy.std(test_input_total_para[:,5])
print "pt mean:",numpy.mean(test_input_total_para[:,6]), "std:",numpy.std(test_input_total_para[:,6])
print "p mean:",numpy.mean(test_input_total_para[:,7]), "std:",numpy.std(test_input_total_para[:,7])
"""

train_input_total1 = dataset_mass_total[0:total_train_length,0:4]
train_output_total1 = dataset_mass_total[0:total_train_length,4]
test_input_total1 = dataset_mass_total[total_train_length:total_length,0:4]
test_output_total1 = dataset_mass_total[total_train_length:total_length,4]
train_input_total2 = dataset_mass_total[total_test_length:total_length,0:4]
train_output_total2 = dataset_mass_total[total_test_length:total_length,4]
test_input_total2 = dataset_mass_total[0:total_test_length,0:4]
test_output_total2 = dataset_mass_total[0:total_test_length,4]

"""
for i in range(0,len(mass_output_total)):
    dataset_mass_total[i,0] = dataset_mass_total[i,0]/scaling_factor
    dataset_mass_total[i,1] = dataset_mass_total[i,1]/scaling_factor
    dataset_mass_total[i,2] = dataset_mass_total[i,2]/scaling_factor
    dataset_mass_total[i,3] = dataset_mass_total[i,3]/scaling_factor
    dataset_mass_total[i,4] = dataset_mass_total[i,4]/scaling_factor

#train_input_total, test_input_total, train_output_total, test_output_total = train_test_split(mass_input_total,mass_output_total,test_size = 0.33, random_state = 42)
print "mean values:",numpy.mean(train_input_total,axis=0)
print "std values:",numpy.std(train_input_total,axis=0)

print "train px",max(train_input_total[:,0])
print "train py",max(train_input_total[:,1])
print "train pz",max(train_input_total[:,2])
print "train E",max(train_input_total[:,3])
print "test px",max(test_input_total[:,0])
print "test py",max(test_input_total[:,1])
print "test pz",max(test_input_total[:,2])
print "test E",max(test_input_total[:,3])

total_px = dataset_mass_total[:,0]
total_py = dataset_mass_total[:,1]
total_pz = dataset_mass_total[:,2]
total_E = dataset_mass_total[:,3]

train_length_total = len(train_output_total)
test_length_total = len(test_output_total)
overfit_total = train_input_total[0:len(test_output_total),:]
"""
####################################################################################################################

train_input_theta = dataset_train_theta[:,0:4]
train_output_theta = dataset_train_theta[:,4]
test_input_theta = dataset_test_theta[:,0:4]
test_output_theta = dataset_test_theta[:,4]
theta_input = dataset_theta[:,0:4]
theta_output = dataset_theta[:,4]

train_input_vismass_all = dataset_vismass_train_all[:,0:4]
train_output_vismass_all = dataset_vismass_train_all[:,4]
test_input_vismass_all = dataset_vismass_test_all[:,0:4]
test_output_vismass_all = dataset_vismass_test_all[:,4]

"""
train_px = dataset_train_theta[:,0]
train_py = dataset_train_theta[:,1]
train_pz = dataset_train_theta[:,2]
train_E = dataset_train_theta[:,3]
train_theta = dataset_train_theta[:,4]
test_px = dataset_test_theta[:,0]
test_py = dataset_test_theta[:,1]
test_pz = dataset_test_theta[:,2]
test_E = dataset_test_theta[:,3]
test_theta = dataset_test_theta[:,4]
"""

train_length = len(dataset_mass_train_total[:,0])
test_length = len(dataset_mass_test_total[:,0])
housing_length = len(dataset_housing[:,0])
train_length_housing = int(round(housing_length*0.7))


#using the housing data for the play model
"""
train_input_play = dataset_housing[0:train_length_housing,0]
test_input_play = dataset_housing[train_length_housing:housing_length,0]
train_input_play = train_input_play.tolist()
test_input_play = test_input_play.tolist()
train_output_play = []
for i in range(0,len(train_input_play)):
    train_output_play.append(train_input_play[i]*train_input_play[i])
test_output_play = []
for j in range(0,len(test_input_play)):
    test_output_play.append(test_input_play[j]*test_input_play[j])
#overfit_play = train_input_play[0:len(test_input_play)]
"""
#using the mass_total data for the play model
"""
train_length_play = len(dataset_mass_train_total[:,3])
test_length_play = len(dataset_mass_test_total[:,3])

for i in range(0,train_length_play):
    dataset_mass_train_total[i,0] = dataset_mass_train_total[i,0]*dataset_mass_train_total[i,0]
    dataset_mass_train_total[i,1] = dataset_mass_train_total[i,1]*dataset_mass_train_total[i,1]
    dataset_mass_train_total[i,2] = dataset_mass_train_total[i,2]*dataset_mass_train_total[i,2]
    dataset_mass_train_total[i,3] = dataset_mass_train_total[i,3]*dataset_mass_train_total[i,3]
    dataset_mass_train_total[i,4] = dataset_mass_train_total[i,4]*dataset_mass_train_total[i,4]

for i in range(0,test_length_play):
    dataset_mass_test_total[i,0] = dataset_mass_test_total[i,0]*dataset_mass_test_total[i,0]
    dataset_mass_test_total[i,1] = dataset_mass_test_total[i,1]*dataset_mass_test_total[i,1]
    dataset_mass_test_total[i,2] = dataset_mass_test_total[i,2]*dataset_mass_test_total[i,2]
    dataset_mass_test_total[i,3] = dataset_mass_test_total[i,3]*dataset_mass_test_total[i,3]
    dataset_mass_test_total[i,4] = dataset_mass_test_total[i,4]*dataset_mass_test_total[i,4]

train_input_play = dataset_mass_train_total[:,0:4]
train_output_play = dataset_mass_train_total[:,4]
test_input_play = dataset_mass_test_total[:,0:4]
test_output_play = dataset_mass_test_total[:,4]
overfit_play = train_input_play[0:len(test_output_play)]
"""
#using random numbers for the play model
"""
numpy.random.seed(10)
train_input_play = []
for i in range(0,100000):
    px = random.uniform(-0.3,0.3)
    py = random.uniform(-0.3,0.3)
    pz = random.uniform(-0.3,0.3)
    E = random.uniform(0.55,1.0)
    train_input_play.append([px,py,pz,E])
    
numpy.random.seed(7)
test_input_play = []
for k in range(0,30000):
    px = random.uniform(-0.3,0.3)
    py = random.uniform(-0.3,0.3)
    pz = random.uniform(-0.3,0.3)
    E = random.uniform(0.55,1.0)
    test_input_play.append([px,py,pz,E])

numpy.random.seed(10)
train_input_play = []
for i in range(0,100000):
    px = random.uniform(-200,200)
    py = random.uniform(-200,200)
    pz = random.uniform(-200,200)
    E = random.uniform(350,6000)
    train_input_play.append([px,py,pz,E])
    
#print "mean px:",numpy.mean(train_input_play[:][0]), "std px:",numpy.std(train_input_play[:][0])
#print "mean py:",numpy.mean(train_input_play[:][1]), "std py:",numpy.std(train_input_play[:][1])
#print "mean pz:",numpy.mean(train_input_play[:][2]),"std pz:",numpy.std(train_input_play[:][2])
#print "mean E:",numpy.mean(train_input_play[:][3]),"std E:",numpy.std(train_input_play[:][3])

numpy.random.seed(7)
test_input_play = []
for k in range(0,30000):
    px = random.uniform(-200,200)
    py = random.uniform(-200,200)
    pz = random.uniform(-200,200)
    E = random.uniform(350,6000)
    test_input_play.append([px,py,pz,E])



train_length_play = len(train_input_play)
test_length_play = len(test_input_play)

train_output_play = []
test_output_play = []
for i in range(0,train_length_play):
    #train_output_play.append(train_input_play[i][0]+train_input_play[i][1]+train_input_play[i][2]+train_input_play[i][3])
    train_output_play.append(numpy.sqrt(train_input_play[i][3]**2-train_input_play[i][0]**2-train_input_play[i][1]**2-train_input_play[i][2]**2))
for j in range(0,test_length_play):
    #test_output_play.append(test_input_play[j][0]+test_input_play[j][1]+test_input_play[j][2]+test_input_play[j][3])
    test_output_play.append(numpy.sqrt(test_input_play[j][3]**2-test_input_play[j][0]**2-test_input_play[j][1]**2-test_input_play[j][2]**2))
overfit_play = train_input_play[0:len(test_input_play)][:]

for k in range(0,len(train_output_play)):
    if math.isnan(train_output_play[k]):
        print "NAN"
        print "px:",train_input_play[k][0]
        print "py:",train_input_play[k][1]
        print "pz:",train_input_play[k][2]
        print "E:",train_input_play[k][3]
        print "E squared:",train_input_play[k][3]**2
        print "p squared:",train_input_play[k][0]**2+train_input_play[k][1]**2+train_input_play[k][2]**2
train_input_play = numpy.array(train_input_play)
train_output_play = numpy.array(train_output_play)
test_input_play = numpy.array(test_input_play)
test_output_play = numpy.array(test_output_play)
"""


#histogram of ditau mass with regression only hadronic decays
histditaumassreg = ROOT.TH1D("ditaumassreg","di-#tau mass using regression hadronic decays",100,0,100)
histditaumassreg.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreg.GetYaxis().SetTitle("number of occurence")
histditaumassreg.SetLineColor(4)
histditaumassreg.SetStats(0)
histditaumassregout = ROOT.TH1D("ditaumassregout","di-#tau mass using regression hadronic decays",100,0,100)
histditaumassregout.SetLineColor(2)
histditaumassregout.SetStats(0)

#histogram of ditau mass with regression all particles
histditaumassregall = ROOT.TH1D("ditaumassregall","reconstructed di-#tau mass using neural network",100,0,100)
#histditaumassregall.GetXaxis().SetTitle("")
histditaumassregall.GetXaxis().SetTitle("di-#tau mass [GeV]")
#histditaumassregall.GetXaxis().SetLabelSize(0)
histditaumassregall.GetYaxis().SetTitle("number of occurence")
histditaumassregall.GetYaxis().SetTitleOffset(1.4)
histditaumassregall.SetLineColor(4)
histditaumassregall.SetStats(0)
histditaumassregallout = ROOT.TH1D("ditaumassregallout","di-#tau mass using regression all decays out",100,0,100)
histditaumassregallout.SetLineColor(2)
histditaumassregallout.SetStats(0)
histditaumassregalloverfit = ROOT.TH1D("ditaumassregalloverfit","disdsdsd",100,0,100)
histditaumassregalloverfit.SetLineColor(3)
histditaumassregalloverfit.SetLineStyle(2)
histditaumassregalloverfit.SetStats(0)
histditaumassregallratio = ROOT.TH1D("ditaumassregallratio","ratio between reconstruced and actual mass",100,0,100)
histditaumassregallratio.SetTitle("")
histditaumassregallratio.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassregallratio.GetXaxis().SetLabelSize(0.08)
histditaumassregallratio.GetXaxis().SetTitleSize(0.08)
histditaumassregallratio.GetYaxis().SetTitle("ratio")
histditaumassregallratio.GetYaxis().SetLabelSize(0.08)
histditaumassregallratio.GetYaxis().SetTitleSize(0.08)
histditaumassregallratio.GetYaxis().SetTitleOffset(0.5)
histditaumassregallratio.GetYaxis().SetNdivisions(504)
histditaumassregallratio.GetYaxis().CenterTitle()
histditaumassregallratio.GetYaxis().SetRangeUser(0.0,5.0)
histditaumassregallratio.SetMarkerStyle(7)
histditaumassregallratio.SetStats(0)
histditaumassregalloverfitratio = ROOT.TH1D("ditaumassregalloverfitratio","ratio between reconstruced and overfit mass",100,0,100)
histditaumassregalloverfitratio.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassregalloverfitratio.GetYaxis().SetTitle("ratio")
histditaumassregalloverfitratio.SetMarkerStyle(7)
histditaumassregalloverfitratio.SetStats(0)



#histogram of ditau mass with regression all particles with all parameter (including pz of neutrino)
histditaumassreginclnu = ROOT.TH1D("ditaumassreginclnu","machine learning example using visible/neutrino 4Vectors",100,0,100)
histditaumassreginclnu.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreginclnu.GetYaxis().SetTitle("number of occurence")
histditaumassreginclnu.GetYaxis().SetTitleOffset(1.5)
histditaumassreginclnu.SetLineColor(4)
histditaumassreginclnu.SetStats(0)
histditaumassreginclnuout = ROOT.TH1D("ditaumassreginclnuout","ssss",100,0,100)
histditaumassreginclnuout.SetLineColor(2)
histditaumassreginclnuout.SetStats(0)
histditaumassreginclnuoverfit = ROOT.TH1D("ditaumassreginclnuoverfit","disdsdsd",100,0,100)
histditaumassreginclnuoverfit.SetLineColor(3)
histditaumassreginclnuoverfit.SetLineStyle(2)
histditaumassreginclnuoverfit.SetStats(0)


#histogram of ditau mass with regression all particles with all parameter (including pz of neutrino) only hadronic decays
histditaumassreginclnuhad = ROOT.TH1D("ditaumassreginclnuhad","di-#tau mass using regression all decays all parameters hadronic decays",100,0,100)
histditaumassreginclnuhad.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreginclnuhad.GetYaxis().SetTitle("number of occurence")
histditaumassreginclnuhad.GetYaxis().SetTitleOffset(1.5)
histditaumassreginclnuhad.SetLineColor(4)
histditaumassreginclnuhad.SetStats(0)
histditaumassreginclnuhadout = ROOT.TH1D("ditaumassreginclnuhadout","zzzz",100,0,100)
histditaumassreginclnuhadout.SetLineColor(2)
histditaumassreginclnuhadout.SetStats(0)



#histogram of ditau mass with regression all particles with 4Vector (all decay-products)
histditaumassregtotal = ROOT.TH1D("ditaumassregtotal","reconstructed di-#tau mass using neural network with di-#tau 4Vector",100,0,100)
histditaumassregtotal.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassregtotal.GetYaxis().SetTitle("number of occurence")
histditaumassregtotal.GetYaxis().SetTitleOffset(1.5)
histditaumassregtotal.SetLineColor(4)
histditaumassregtotal.SetStats(0)
histditaumassregtotalout = ROOT.TH1D("ditaumassregtotalout","ssss",100,0,100)
histditaumassregtotalout.SetLineColor(2)
histditaumassregtotalout.SetStats(0)
histditaumassregtotalcalc = ROOT.TH1D("ditaumassregtotalcalc","ssss",100,0,100)
histditaumassregtotalcalc.SetLineColor(3)
histditaumassregtotalcalc.SetLineStyle(2)
histditaumassregtotalcalc.SetStats(0)
histditaumassregtotaltraintarget = ROOT.TH1D("ditaumassregtotaltraintarget","dddd",100,0,100)
histditaumassregtotaltraintarget.SetLineColor(5)
histditaumassregtotaltraintarget.SetLineStyle(2)
histditaumassregtotaltraintarget.SetStats(0)

#histogram of ditau mass with regression all particles with 4Vector (all decay-products) only hadronic decays
histditaumassregtotalhad = ROOT.TH1D("ditaumassregtotalhad","di-#tau mass using regression all decays 4Vector all decay products hadronic decays",100,0,100)
histditaumassregtotalhad.GetXaxis().SetTitle("di-#tau mass")
histditaumassregtotalhad.GetYaxis().SetTitle("number of occurence")
histditaumassregtotalhad.SetLineColor(4)
histditaumassregtotalhad.SetStats(0)
histditaumassregtotalouthad = ROOT.TH1D("ditaumassregtotalouthad","ssss",100,0,100)
histditaumassregtotalouthad.SetLineColor(2)
histditaumassregtotalouthad.SetStats(0)
histditaumassregtotalcalchad = ROOT.TH1D("ditaumassregtotalcalchad","ssss",100,0,100)
histditaumassregtotalcalchad.SetLineColor(3)
histditaumassregtotalcalchad.SetLineStyle(2)
histditaumassregtotalcalchad.SetStats(0)


#histogram of ditau theta with regression all particles with 4Vector (all decay-products)
histditauregtheta = ROOT.TH1D("ditauregtheta","reconstructed di-#tau #theta using neural network with di-#tau 4Vector",100,0,3.5)
histditauregtheta.GetXaxis().SetTitle("di-#tau #theta")
histditauregtheta.GetYaxis().SetTitle("number of occurence")
histditauregtheta.GetYaxis().SetTitleOffset(1.2)
histditauregtheta.SetLineColor(4)
histditauregtheta.SetStats(0)
histditauregthetaout = ROOT.TH1D("ditauregthetaout","ssss",100,0,3.5)
histditauregthetaout.SetLineColor(2)
histditauregthetaout.SetStats(0)


#histogram of ditau vismass with regression all particles with 4Vector (all decay-products)
histditauvismassregall = ROOT.TH1D("ditauvismassregall","reconstructed di-#tau mass_{vis} using neural network with di-#tau 4Vector",100,0,100)
histditauvismassregall.GetXaxis().SetTitle("di-#tau mass_{vis} [GeV]")
histditauvismassregall.GetYaxis().SetTitle("number of occurence")
histditauvismassregall.GetYaxis().SetTitleOffset(1.5)
histditauvismassregall.SetLineColor(2)
histditauvismassregall.SetStats(0)
histditauvismassregallout = ROOT.TH1D("ditauvismassregallout","ssss",100,0,100)
histditauvismassregallout.SetLineColor(4)
histditauvismassregallout.SetStats(0)
histditauvismassregallcalc = ROOT.TH1D("ditauvismassregallcalc","tttt",100,0,100)
histditauvismassregallcalc.SetLineColor(3)
histditauvismassregallcalc.SetLineStyle(2)
histditauvismassregallcalc.SetStats(0)


#histogram of ditau vismass with regression only hadronic decays
histditauvismassreghad = ROOT.TH1D("ditauvismassreghad","di-#tau vis-mass using regression only hadronic decays",100,0,100)
histditauvismassreghad.GetXaxis().SetTitle("di-#tau vis-mass")
histditauvismassreghad.GetYaxis().SetTitle("number of occurence")
histditauvismassreghad.SetLineColor(2)
histditauvismassreghad.SetStats(0)
histditauvismassreghadout = ROOT.TH1D("ditauvismassreghadout","ssss",100,0,100)
histditauvismassreghadout.SetLineColor(4)
histditauvismassreghadout.SetStats(0)
histditauvismassreghadcalc = ROOT.TH1D("ditauvismassreghadcalc","tttt",100,0,100)
histditauvismassreghadcalc.SetLineColor(3)
histditauvismassreghadcalc.SetLineStyle(2)
histditauvismassreghadcalc.SetStats(0)

hist_p_range = 100.
#histogram of px of ditau comparison train and test
histpxditautrain = ROOT.TH1D("pxditautrain","px di-#tau",100,-hist_p_range,hist_p_range)
histpxditautrain.GetXaxis().SetTitle("di-#tau px [GeV]")
histpxditautrain.GetYaxis().SetTitle("number of occurence")
histpxditautrain.SetLineColor(4)
histpxditautrain.SetStats(0)
histpxditautest = ROOT.TH1D("pxditautest","px sdsd",100,-hist_p_range,hist_p_range)
histpxditautest.SetLineColor(2)
histpxditautest.SetStats(0)

#histogram of py of ditau comparison train and test
histpyditautrain = ROOT.TH1D("pyditautrain","py di-#tau",100,-hist_p_range,hist_p_range)
histpyditautrain.GetXaxis().SetTitle("di-#tau py [GeV]")
histpyditautrain.GetYaxis().SetTitle("number of occurence")
histpyditautrain.SetLineColor(4)
histpyditautrain.SetStats(0)
histpyditautest = ROOT.TH1D("pyditautest","py sdsd",100,-hist_p_range,hist_p_range)
histpyditautest.SetLineColor(2)
histpyditautest.SetStats(0)

#histogram of pz of ditau comparison train and test
histpzditautrain = ROOT.TH1D("pzditautrain","pz di-#tau",100,-hist_p_range,hist_p_range)
histpzditautrain.GetXaxis().SetTitle("di-#tau pz[GeV]")
histpzditautrain.GetYaxis().SetTitle("number of occurence")
histpzditautrain.GetYaxis().SetTitleOffset(1.4)
histpzditautrain.SetLineColor(4)
histpzditautrain.SetStats(0)
histpzditautest = ROOT.TH1D("pzditautest","pz sdsd",100,-hist_p_range,hist_p_range)
histpzditautest.SetLineColor(2)
histpzditautest.SetStats(0)

#histogram of E of ditau comparison train and test
histeditautrain = ROOT.TH1D("editautrain","energy di-#tau",100,0,200)
histeditautrain.GetXaxis().SetTitle("di-#tau E [GeV]")
histeditautrain.GetYaxis().SetTitle("number of occurence")
histeditautrain.GetYaxis().SetTitleOffset(1.4)
histeditautrain.SetLineColor(4)
histeditautrain.SetStats(0)
histeditautest = ROOT.TH1D("editautest","E sdsd",100,0,200)
histeditautest.SetLineColor(2)
histeditautest.SetStats(0)

#histogram of theta of ditau comparison train and test
histthetaditautrain = ROOT.TH1D("thetaditautrain","#theta di-#tau",100,0,3.5)
histthetaditautrain.GetXaxis().SetTitle("di-#tau #theta")
histthetaditautrain.GetYaxis().SetTitle("number of occurence")
histthetaditautrain.SetLineColor(4)
histthetaditautrain.SetStats(0)
histthetaditautest = ROOT.TH1D("thetaditautest","#theta sdsd",100,0,3.5)
histthetaditautest.SetLineColor(2)
histthetaditautest.SetStats(0)


#histogram of ditau playing with different parameters
#histditauplay = ROOT.TH1D("ditauplay","machine learning example mass random numbers",100,0.3,1.1)
histditauplay = ROOT.TH1D("ditauplay","machine learning example mass random numbers",100,0.0,6050.0)
histditauplay.GetXaxis().SetTitle("pseudo mass")
histditauplay.GetYaxis().SetTitle("number of occurence")
histditauplay.GetYaxis().SetTitleOffset(1.2)
histditauplay.SetLineColor(4)
histditauplay.SetStats(0)
#histditauplayout = ROOT.TH1D("ditauplayout","ssss",100,0.3,1.1)
histditauplayout = ROOT.TH1D("ditauplayout","ssss",100,0.0,6050.0)
histditauplayout.SetLineColor(2)
histditauplayout.SetLineStyle(2)
histditauplayout.SetStats(0)
#histditauplaycalc = ROOT.TH1D("ditauplaycalc","ssss",100,0.3,1.1)
histditauplaycalc = ROOT.TH1D("ditauplaycalc","ssss",100,0.0,6050.0)
histditauplaycalc.SetLineColor(3)
histditauplaycalc.SetLineStyle(2)
histditauplaycalc.SetStats(0)

#histogram of ditau different train/test datasets
histditaumassregtotal1 = ROOT.TH1D("ditauregtotal1","machine learning example using di-#tau 4Vector different train/test datasets",100,0.0,100.0)
histditaumassregtotal1.GetXaxis().SetTitle("mass [GeV]")
histditaumassregtotal1.GetYaxis().SetTitle("number of occurence")
histditaumassregtotal1.GetYaxis().SetTitleOffset(1.4)
histditaumassregtotal1.SetLineColor(4)
#histditaumassregtotal1.SetLineStyle(2)
histditaumassregtotal1.SetStats(0)
histditaumassregtotal1out = ROOT.TH1D("ditauregtotal1out","ssss",100,0.0,100.0)
histditaumassregtotal1out.SetLineColor(2)
#histditaumassregtotal1out.SetLineStyle(2)
histditaumassregtotal1out.SetStats(0)
histditaumassregtotal2 = ROOT.TH1D("ditauregtotal2","zzz",100,0.0,100.0)
histditaumassregtotal2.SetLineColor(3)
histditaumassregtotal2.SetLineStyle(2)
histditaumassregtotal2.SetStats(0)
histditaumassregtotal2out = ROOT.TH1D("ditauregtotal2out","ssss",100,0.0,100.0)
histditaumassregtotal2out.SetLineColor(5)
histditaumassregtotal2out.SetLineStyle(2)
histditaumassregtotal2out.SetStats(0)

#histogram of ratio  between reconstructed and overfit mass
histditaumassregtotaloverfitratio = ROOT.TH1D("overfitratio","ratio between reconstructed and overfit mass",50,0.8,1.2)
histditaumassregtotaloverfitratio.GetXaxis().SetTitle("ratio")
histditaumassregtotaloverfitratio.GetYaxis().SetTitle("number of occurence")
histditaumassregtotaloverfitratio.SetLineColor(4)
histditaumassregtotaloverfitratio.SetStats(0)

"""

for i in total_px:
    histpxditautrain.Fill(i)
for j in total_py:
    histpyditautrain.Fill(j)
for k in total_pz:
    histpzditautrain.Fill(k)
for m in total_E:
    histeditautrain.Fill(m)


for i in train_px:
    histpxditautrain.Fill(i)
for j in train_py:
    histpyditautrain.Fill(j)
for k in train_pz:
    histpzditautrain.Fill(k)
for m in train_E:
    histeditautrain.Fill(m)
for s in train_theta:
    histthetaditautrain.Fill(s)
for n in test_px:
    histpxditautest.Fill(n)
for q in test_py:
    histpyditautest.Fill(q)
for c in test_pz:
    histpzditautest.Fill(c)
for v in test_E:
    histeditautest.Fill(v)
for w in test_theta:
    histthetaditautest.Fill(w)
"""

#n=len(train_output_all)


"""
# evaluate model
estimator = KerasRegressor(build_fn=ditaumass_model, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n,n_folds=10,random_state=seed)
results = cross_val_score(estimator, train_input_all, train_output_all, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=ditaumass_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n,n_folds=10, random_state=seed)
results = cross_val_score(pipeline, train_input_all, train_output_all, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
"""

def mass_model_had(batch_size,epochs,output_name):
    mass_model_had = Sequential()
    mass_model_had.add(Dense(40,input_dim=7,kernel_initializer='random_uniform',activation='relu'))
    mass_model_had.add(Dense(20,kernel_initializer='random_uniform',activation = 'relu'))
    mass_model_had.add(Dense(10,kernel_initializer='random_uniform',activation = 'relu'))
    mass_model_had.add(Dense(1,kernel_initializer='random_uniform',activation = 'relu'))
    mass_model_had.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_had.fit(train_input,train_output,batch_size,epochs,verbose=1)
    mass_score_had = mass_model_had.evaluate(test_input,test_output,batch_size,verbose=0)
    ditaumass_had = mass_model_had.predict(test_input,batch_size,verbose=0)
    print "mass_model_had(",batch_size,epochs,")"
    print "score for hadronic decays:",mass_score_had
    #preparing the histograms
    for i in ditaumass_had:
        histditaumassreg.Fill(i)
    for k in test_output:
        histditaumassregout.Fill(k)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression hadronic")
    max_bin = max(histditaumassreg.GetMaximum(),histditaumassregout.GetMaximum())
    histditaumassreg.SetMaximum(max_bin*1.08)
    histditaumassreg.Draw()
    histditaumassregout.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassreg,"reconstructed mass","PL")
    leg.AddEntry(histditaumassregout,"actual mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_hadronic.png")


def mass_model_inclnu_had(batch_size,epochs):
    mass_model_inclnu_had = Sequential()
    mass_model_inclnu_had.add(Dense(40,input_dim=8,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.add(Dense(20,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.add(Dense(5,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.add(Dense(1,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_inclnu_had.fit(train_input_inclnu,train_output_inclnu,batch_size,epochs,verbose=1)
    mass_score_inclnu_had = mass_model_inclnu_had.evaluate(test_input_inclnu,test_output_inclnu,batch_size,verbose=0)
    ditaumass_inclnu = mass_model_inclnu_had.predict(test_input_inclnu,batch_size,verbose=0)
    print "mass_model_inclnu_had(",batch_size,epochs,")"
    print "score:",mass_score_inclnu_had
    #preparing the histograms
    for k in ditaumass_inclnu_had:
        histditaumassreginclnuhad.Fill(k)
    for p in test_output_inclnu:
        histditaumassreginclnuhadout.Fill(p)
    #histogram of di-tau mass using regression all decays all parameters
    canv = ROOT.TCanvas("di-tau mass using regression all decays all parameters hadronic decays")
    max_bin = max(histditaumassreginclnuhad.GetMaximum(),histditaumassreginclnuhadout.GetMaximum())
    histditaumassreginclnuhad.SetMaximum(max_bin*1.08)
    histditaumassreginclnuhad.Draw()
    histditaumassreginclnuhadout.Draw("SAME")
    leg = ROOT.TImage.Create()
    leg.AddEntry(histditaumassreginclnuhad,"reconstructed mass","PL")
    leg.AddEntry(histditaumassreginclnuhadout,"actual mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_inclnu_had.png")


def mass_model_total(batch_size,epochs,output_name):
    mass_model_total = Sequential()
    mass_model_total.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total.add(Dropout(0.2))
    #mass_model_total.add(BatchNormalization())
    mass_model_total.add(Dense(30,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total.add(Dropout(0.2))
    #mass_model_total.add(BatchNormalization())
    mass_model_total.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total.add(LeakyReLU(alpha=0.001))
    #mass_model_total.add(BatchNormalization())
    mass_model_total.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total.add(Dropout(0.2))
    #mass_model_total.add(BatchNormalization()))
    mass_model_total.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    #mass_model_total.add(BatchNormalization())
    #mass_model_total.load_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/mass_model_total_weights_try10.h5")
    mass_model_total.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_total.fit(train_input_total,train_output_total,batch_size,epochs,verbose=2)
    mass_score_total = mass_model_total.evaluate(test_input_total,test_output_total,batch_size,verbose=0)
    ditaumass_total = mass_model_total.predict(test_input_total,batch_size,verbose=0)
    #ditaumass_overfit_total = mass_model_total.predict(overfit_total,batch_size,verbose=0)
    #mass_model_total.fit(train_input_total_para,train_output_total,batch_size,epochs,verbose=2)
    #mass_score_total = mass_model_total.evaluate(test_input_total_para,test_output_total,batch_size,verbose=0)
    #ditaumass_total = mass_model_total.predict(test_input_total_para,batch_size,verbose=0)
    #ditaumass_overfit_total = mass_model_total.predict(overfit_total_para,batch_size,verbose=0)
    mass_model_total.summary()
    print "mass_model_total(",batch_size,epochs,")"
    print "score:",mass_score_total
    print description_of_training
    #mass_model_total.save_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/mass_model_total_weights_try10.h5")
    #preparing the histograms
    number_overfit_fail = 0
    for u in ditaumass_total:
        histditaumassregtotal.Fill(u)
    for g in test_output_total:
        histditaumassregtotalout.Fill(g)
    #for j in ditaumass_overfit_total:
    #    histditaumassregtotalcalc.Fill(j)
    for k in range(0,100):
        if histditaumassregtotalcalc.GetBinContent(k) != 0:
            histditaumassregtotaloverfitratio.Fill(histditaumassregtotal.GetBinContent(k)/histditaumassregtotalcalc.GetBinContent(k))
        elif histditaumassregtotalcalc.GetBinContent(k) == 0 and histditaumassregtotal.GetBinContent(k) == 0:
            histditaumassregtotaloverfitratio.Fill(1.0)
        else:
            number_overfit_fail += 1
    """
    for d in train_output_total[0:len(test_output_total)]:
        histditaumassregtotaltraintarget.Fill(d)

    for i in range(0,test_length_total):
        mass_calc = ROOT.TLorentzVector(test_input_total[i,0],test_input_total[i,1],test_input_total[i,2],test_input_total[i,3]).M()
        histditaumassregtotalcalc.Fill(mass_calc)
    """
    
    #histograms
    canv1 = ROOT.TCanvas("di-tau mass using regression all decays 4Vector all decay products")
    max_bin1 = max(histditaumassregtotal.GetMaximum(),histditaumassregtotalout.GetMaximum(),histditaumassregtotalcalc.GetMaximum())
    histditaumassregtotal.SetMaximum(max_bin1*1.08)
    histditaumassregtotal.Draw()
    histditaumassregtotalout.Draw("SAME")
    histditaumassregtotalcalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassregtotalout,"di-#tau_{gen} mass","PL")
    leg.AddEntry(histditaumassregtotal,"di-#tau_{NN} mass","PL")
    leg.AddEntry(histditaumassregtotalcalc,"di-#tau_{calc} mass","PL")
    leg.Draw()
    output_hist_name = "%s.png" %(output_name)
    img1 = ROOT.TImage.Create()
    img1.FromPad(canv1)
    img1.WriteImage(output_hist_name)
    """
    #histogram of ratio between reconstructed and overfit mass
    canv2 = ROOT.TCanvas("ratio between reconstructed and overfit mass")
    histditaumassregtotaloverfitratio.Draw()
    img2 = ROOT.TImage.Create()
    img2.FromPad(canv2)
    img2.WriteImage("reg_ditau_mass_total_overfit_ratio_try1.png")
    """


def mass_model_total_had(batch_size,epochs,output):
    mass_model_total_had = Sequential()
    mass_model_total_had.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_total_had.fit(train_input_total,train_output_total,batch_size,epochs,verbose=1)
    mass_score_total_had = mass_model_total_had.evaluate(test_input_total,test_output_total,verbose=0)
    ditaumass_total_had = mass_model_total_had.predict(test_input_total,batch_size,verbose=0)
    print "mass_model_total_had(",batch_size,epochs,")"
    print "score:",mass_score_total_had
    #preparing the histograms
    for u in ditaumass_total_had:
        histditaumassregtotalhad.Fill(u)
    for g in test_output_total_had:
        histditaumassregtotalout_had.Fill(g)
    for i in range(0,len(test_output_total_had)):
        mass_calc = ROOT.TLorentzVector(test_input_total[i,0],test_input_total[i,1],test_input_total[i,2],test_input_total[i,3]).M()
        histditaumassregtotalcalc.Fill(mass_calc)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression all decays 4Vector all decay products hadronic decays")
    max_bin = max(histditaumassregtotalhad.GetMaximum(),histditaumassregtotalouthad.GetMaximum(),histditaumassregtotalcalchad.GetMaximum())
    histditaumassregtotalhad.SetMaximum(max_bin*1.08)
    histditaumassregtotalhad.Draw()
    histditaumassregtotalouthad.Draw("SAME")
    histditaumassregtotalcalchad.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassregtotalhad,"reconstructed mass","PL")
    leg.AddEntry(histditaumassregtotalouthad,"actual mass","PL")
    leg.AddEntry(histditaumassregtotalcalchad,"calculated mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_total_had.png")


def theta_model(batch_size,epochs,output):
    theta_model = Sequential()
    theta_model.add(Dense(20,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    theta_model.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    theta_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    theta_model.compile(loss='mean_squared_error',optimizer='adam')
    theta_model.fit(train_input_theta,train_output_theta,batch_size,epochs,verbose=0)
    theta_score = theta_model.evaluate(test_input_theta,test_output_theta,batch_size,verbose=0)
    ditautheta = theta_model.predict(test_input_theta,batch_size,verbose=0)
    print "theta_model(",batch_size,epochs,")"
    print "score:",theta_score
    #preparing the histograms
    for t in ditautheta:
        histditauregtheta.Fill(t)
    for d in test_output_theta:
        histditauregthetaout.Fill(d)
    #histograms
    canv = ROOT.TCanvas("di-tau theta using regression all decays 4Vector all decay products")
    max_bin = max(histditauregtheta.GetMaximum(),histditauregthetaout.GetMaximum())
    histditauregtheta.SetMaximum(max_bin*1.3)
    histditauregtheta.Draw()
    histditauregthetaout.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditauregthetaout,"di-#tau #theta_{gen}","PL")
    leg.AddEntry(histditauregtheta,"di-#tau #theta_{NN}","PL")
    leg.Draw()
    output_name = "%s.png" %(output)
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage(output_name)

def vismass_model_all(batch_size,epochs,output):
    vismass_model_all = Sequential()
    vismass_model_all.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_all.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_all.add(Dense(5,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_all.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    vismass_model_all.compile(loss='mean_squared_error',optimizer='adam')
    vismass_model_all.fit(train_input_vismass_all,train_output_vismass_all,batch_size,epochs,verbose=1)
    vismass_score_all = vismass_model_all.evaluate(test_input_vismass_all,test_output_vismass_all,batch_size,verbose=0)
    ditauvismass_all = vismass_model_all.predict(test_input_vismass_all,batch_size,verbose=0)
    print "vismass_model_all(",batch_size,epochs,")"
    print "score:",vismass_score_all
    #preparing the histograms
    for p in ditauvismass_all:
        histditauvismassregall.Fill(p)
    for s in test_output_vismass_all:
        histditauvismassregallout.Fill(s)
    for i in range(0,len(dataset_vismass_test_all)):
        vismass_calc = ROOT.TLorentzVector(dataset_vismass_test_all[i,0],dataset_vismass_test_all[i,1],dataset_vismass_test_all[i,2],dataset_vismass_test_all[i,3]).M()
        histditauvismassregallcalc.Fill(vismass_calc)
    #histograms
    canv = ROOT.TCanvas("di-tau vismass regression all")
    max_bin = max(histditauvismassregall.GetMaximum(),histditauvismassregallout.GetMaximum(),histditauvismassregallcalc.GetMaximum())
    histditauvismassregall.SetMaximum(max_bin*1.2)
    histditauvismassregall.Draw()
    histditauvismassregallout.Draw("SAME")
    histditauvismassregallcalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditauvismassregallout,"di-#tau visible mass_{gen}","PL")
    leg.AddEntry(histditauvismassregall,"di-#tau visible mass_{NN}","PL")
    leg.AddEntry(histditauvismassregallcalc,"di-#tau visible mass_{calc}","PL")
    leg.Draw()
    output_name = "%s.png" %(output)
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage(output_name)

def vismass_model_had(batch_size,epochs,output):
    vismass_model_had = Sequential()
    vismass_model_had.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.add(Dense(5,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.compile(loss='mean_squared_error',optimizer='adam')
    vismass_model_had.fit(train_input_vismass_had,train_output_vismass_had,batch_size,epochs,verbose=1)
    vismass_score_had = vismass_model_had.evaluate(test_input_vismass_had,test_output_vismass_had,batch_size,verbose=0)
    ditauvismass_had = vismass_model_had.predict(test_input_vismass_had,batch_size,verbose=0)
    print "vismass_model_had(",batch_size,epochs,")"
    print "score:",vismass_score_had
    for a in ditauvismass_had:
        histditauvismassreghad.Fill(a)
    for w in test_output_vismass_had:
        histditauvismassreghadout.Fill(w)
    for i in range(0,len(dataset_vismass_test_had)):
        vismass_calc_had = ROOT.TLorentzVector(dataset_vismass_test_had[i,0],dataset_vismass_test_had[i,1],dataset_vismass_test_had[i,2],dataset_vismass_test_had[i,3]).M()
        histditauvismassreghadcalc.Fill(vismass_calc_had)
    #histogram of di-tau vismass using regression only hadronic decays
    canv = ROOT.TCanvas("di-tau vismass regression had")
    max_bin = max(histditauvismassreghad.GetMaximum(),histditauvismassreghadout.GetMaximum(),histditauvismassreghadcalc.GetMaximum())
    histditauvismassreghad.SetMaximum(max_bin*1.2)
    histditauvismassreghad.Draw()
    histditauvismassreghadout.Draw("SAME")
    histditauvismassreghadcalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.8,0.9,0.9)
    leg.AddEntry(histditauvismassreghad,"reconstructed vis-mass","PL")
    leg.AddEntry(histditauvismassreghadout,"actual vis-mass","PL")
    leg.AddEntry(histditauvismassreghadcalc,"calculated vis-mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("vismass_ditau_had.png")

def play_model(batch_size,epochs,output):
    #clf_play = RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_split=2,random_state=0)
    #clf_play.fit(train_input_play,train_output_play)
    #scores_play = clf_play.score(test_input_play,test_output_play)
    #ditau_play = clf_play.predict(test_input_play)
    #print "score RandomForest total:",scores_play
    play_model = Sequential()
    #play_model.add(Dropout(0.1,input_shape=(4,)))
    play_model.add(Dense(150,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dropout(0.1))
    #play_model.add(Dense(300,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dense(40,kernel_initializer='random_uniform',activation='relu'))
    play_model.add(Dense(130,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dropout(0.1))
    #play_model.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dense(110,kernel_initializer='random_uniform',activation='relu'))
    play_model.add(Dense(100,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dense(80,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dropout(0.1))
    play_model.add(Dense(50,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dropout(0.1))
    #play_model.add(Dense(300,kernel_initializer='random_uniform',activation='relu'))
    play_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    #play_model.load_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/play_model_weights.h5")
    play_model.compile(loss='mean_squared_error',optimizer='adam')
    play_model.fit(train_input_play,train_output_play,batch_size,epochs,verbose=1)
    play_score = play_model.evaluate(test_input_play,test_output_play,batch_size,verbose=0)
    ditau_play = play_model.predict(test_input_play,batch_size,verbose=0)
    ditau_overfit_play = play_model.predict(overfit_play,batch_size,verbose=0)
    play_model.summary()
    play_model.save_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/play_model_weights.h5")
    print "large random number mass try 6"
    print "play_model(",batch_size,epochs,")"
    print "score:",play_score
    #preparing the histograms
    for u in ditau_play:
        histditauplay.Fill(u)
    for g in test_output_play:
        histditauplayout.Fill(g)
    for j in ditau_overfit_play:
        histditauplaycalc.Fill(j)
    """
    for i in range(0,test_length_play):
        play_calc = test_input_play[i,3]-test_input_play[i,0]-test_input_play[i,1]-test_input_play[i,2]
        #play_calc = test_input_play[i]*test_input_play[i]
        histditauplaycalc.Fill(play_calc)
    """
    #histograms
    canv = ROOT.TCanvas("regression random numbers squared")
    max_bin = max(histditauplay.GetMaximum(),histditauplayout.GetMaximum(),histditauplaycalc.GetMaximum())
    histditauplay.SetMaximum(max_bin*1.32)
    histditauplay.Draw()
    histditauplayout.Draw("SAME")
    histditauplaycalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditauplay,"reconstructed mass","PL")
    leg.AddEntry(histditauplayout,"actual mass","PL")
    leg.AddEntry(histditauplaycalc,"overfit mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_random_numbers_mass_try6_largenumbers.png")
    canv.WaitPrimitive()

def mass_model_total_diff(batch_size,epochs,output):
    mass_model_total1 = Sequential()
    mass_model_total1.add(Dense(200,input_dim=4,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total1.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total1.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total1.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total1.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    mass_model_total1.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_total1.fit(train_input_total1,train_output_total1,batch_size,epochs,verbose=1)
    mass_score_total1 = mass_model_total1.evaluate(test_input_total1,test_output_total1,batch_size,verbose=0)
    ditaumass_total1 = mass_model_total1.predict(test_input_total1,batch_size,verbose=0)
    mass_model_total1.summary()
    mass_model_total2 = Sequential()
    mass_model_total2.add(Dense(200,input_dim=4,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total2.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total2.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total2.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_total2.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    mass_model_total2.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_total2.fit(train_input_total2,train_output_total2,batch_size,epochs,verbose=1)
    mass_score_total2 = mass_model_total2.evaluate(test_input_total2,test_output_total2,batch_size,verbose=0)
    ditaumass_total2 = mass_model_total2.predict(test_input_total2,batch_size,verbose=0)
    mass_model_total2.summary()
    ditaumass_total = []
    #for i in range(0,len(ditaumass_total1)):
    #    ditaumass_total.append((ditaumass_total1[i]+ditaumass_total2[i])/2.0)
    print "mass_model_total(",batch_size,epochs,")"
    print "score1:",mass_score_total1
    print "score2:",mass_score_total2
    print "2models 20 EPOCHS"
    #preparing the histograms
    for u in ditaumass_total:
        histditaumassregtotal1.Fill(u)
    for g in test_output_total1:
        histditaumassregtotal1out.Fill(g)
    #for t in ditaumass_total2:
    #    histditaumassregtotal2.Fill(t)
    #for f in test_output_total2:
    #    histditaumassregtotal2out.Fill(f)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression all decays 4Vector all decay products")
    max_bin = max(histditaumassregtotal1.GetMaximum(),histditaumassregtotal1out.GetMaximum())
    histditaumassregtotal1.SetMaximum(max_bin*1.08)
    histditaumassregtotal1.Draw()
    histditaumassregtotal1out.Draw("SAME")
    #histditaumassregtotal2.Draw("SAME")
    #histditaumassregtotal2out.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassregtotal1,"reconstructed mass with 2 NN","PL")
    leg.AddEntry(histditaumassregtotal1out,"actual mass","PL")
    #leg.AddEntry(histditaumassregtotal2,"reconstructed mass dataset2","PL")
    #leg.AddEntry(histditaumassregtotal2out,"actual mass dataset2","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_total_2models_20.png")

def mass_model_inclnu(batch_size,epochs,output_name):
    mass_model_inclnu = Sequential()
    mass_model_inclnu.add(Dense(200,input_dim=9,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_inclnu.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_inclnu.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model_inclnu.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model_inclnu.add(Dense(200,kernel_initializer='random_uniform',activation='softsign')) 
    mass_model_inclnu.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    mass_model_inclnu.compile(loss='mean_squared_error',optimizer='adam')
    #Input data: px_vis,py_vis,pz_vis,energy_vis,px_nu,py_nu,pz_nu,energy_nu
    #mass_model_inclnu.fit(train_input_inclnu,train_output_inclnu,batch_size,epochs,verbose=2)
    #mass_score_inclnu = mass_model_inclnu.evaluate(test_input_inclnu,test_output_inclnu,batch_size,verbose=0)
    #ditaumass_inclnu = mass_model_inclnu.predict(test_input_inclnu,batch_size,verbose=0)

    mass_model_inclnu.fit(train_input_inclnu_para,train_output_inclnu,batch_size,epochs,verbose=2)
    mass_score_inclnu = mass_model_inclnu.evaluate(test_input_inclnu_para,test_output_inclnu,batch_size,verbose=0)
    ditaumass_inclnu = mass_model_inclnu.predict(test_input_inclnu_para,batch_size,verbose=0)
    ditaumass_inclnu_overfit = mass_model_inclnu.predict(overfit_inclnu_para,batch_size,verbose=0)
    mass_model_inclnu.summary()
    print "mass_model_inclnu(",batch_size,epochs,")"
    print "score:",mass_score_inclnu
    print description_of_training
    #preparing the histograms
    for k in ditaumass_inclnu:
        histditaumassreginclnu.Fill(k)
    for j in test_output_inclnu:
        histditaumassreginclnuout.Fill(j)
    for w in ditaumass_inclnu_overfit:
        histditaumassreginclnuoverfit.Fill(w)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression all decays all parameters")
    max_bin = max(histditaumassreginclnu.GetMaximum(),histditaumassreginclnuout.GetMaximum(),histditaumassreginclnuoverfit.GetMaximum())
    histditaumassreginclnu.SetMaximum(max_bin*1.08)
    histditaumassreginclnu.Draw()
    histditaumassreginclnuout.Draw("SAME")
    histditaumassreginclnuoverfit.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassreginclnu,"reconstructed mass","PL")
    leg.AddEntry(histditaumassreginclnuout,"actual mass","PL")
    leg.AddEntry(histditaumassreginclnuoverfit,"overfit mass","PL")
    leg.Draw()
    output_hist_name = "%s.png" %(output_name)
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage(output_hist_name)

def mass_model_all(batch_size,epochs,learning_rate,decay_rate,output_name):
    mass_model_all = Sequential()
    mass_model_all.add(Dense(40,input_dim=6,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(30,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))

    def exp_decay(epoch):
        initial_lrate = 0.01
        lrate = initial_lrate *numpy.exp(-initial_lrate*epoch)
        return lrate
    lrate = [LearningRateScheduler(exp_decay)]

    def step_decay(epoch):
        initial_lrate = 0.005
        drop = 0.7
        epochs_drop = 10.0
        lrate = initial_lrate*math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate

    #lrate = [LearningRateScheduler(exp_decay)]
    #lrate = [LearningRateScheduler(step_decay)]
    #adam = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = decay_rate)
    #adam = Adam(lr = 0.0, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = 0.0)
    mass_model_all.compile(loss='mean_squared_error',optimizer='adam')
    #history = mass_model_all.fit(mass_input_inclnu_para,mass_output_inclnu,batch_size,epochs, validation_split = 0.3,verbose = 2)
    history = mass_model_all.fit(train_input_all,train_output_all,batch_size,epochs, validation_data = (test_input_all,test_output_all),verbose = 2)
    #history = mass_model_all.fit(mass_input_inclnu_para,mass_output_inclnu,batch_size,epochs, validation_split = 0.3,callbacks = lrate,verbose = 2)
    #mass_score_all = mass_model_all.evaluate(test_input_inclnu_para,test_output_inclnu,batch_size,verbose=0)
    mass_score_all = mass_model_all.evaluate(test_input_all,test_output_all,batch_size,verbose=0)
    ditaumass_all = mass_model_all.predict(test_input_all,batch_size,verbose=0)
    #ditaumass_all = mass_model_all.predict(test_input_inclnu_para,batch_size,verbose=0)
    #ditaumass_all_overfit = mass_model_all.predict(overfit_inclnu_para,batch_size,verbose=0)
   
    mass_model_all.summary()
    print "mass_model_all(",batch_size,epochs,")"
    print "loss (MSE):",mass_score_all
    print description_of_training
    failed_division = 0
    #preparing the histograms
    for j in ditaumass_all:
        histditaumassregall.Fill(j)
    for h in test_output_all:
        histditaumassregallout.Fill(h)
    #for d in ditaumass_all_overfit:
    #    histditaumassregalloverfit.Fill(d)
    for k in range(0,100):
        if histditaumassregallout.GetBinContent(k) != 0:
            histditaumassregallratio.SetBinContent(k,histditaumassregall.GetBinContent(k)/histditaumassregallout.GetBinContent(k))
            print "bin",k,"ratio:", histditaumassregall.GetBinContent(k)/histditaumassregallout.GetBinContent(k)
        elif histditaumassregall.GetBinContent(k) == 0 and histditaumassregallout.GetBinContent(k) == 0:
            histditaumassregallratio.SetBinContent(k,1.0)
            print "bin",k,"ratio: 1"
        else:
            failed_division +=1
    epochs_range = numpy.array([float(i) for i in range(1,epochs+1)])
    loss_graph = ROOT.TGraph(epochs,epochs_range,numpy.array(history.history['loss']))
    loss_graph.SetTitle("model loss")
    loss_graph.GetXaxis().SetTitle("epochs")
    loss_graph.GetYaxis().SetTitle("loss")
    loss_graph.SetMarkerColor(4)
    loss_graph.SetMarkerSize(0.8)
    loss_graph.SetMarkerStyle(21)
    val_loss_graph = ROOT.TGraph(epochs,epochs_range,numpy.array(history.history['val_loss']))
    val_loss_graph.SetMarkerColor(2)
    val_loss_graph.SetMarkerSize(0.8)
    val_loss_graph.SetMarkerStyle(21)
    print "failed divisions:",failed_division
    #plots of the loss of train and test sample
    canv1 = ROOT.TCanvas("loss di-tau mass")
    loss_graph.Draw("AP")
    val_loss_graph.Draw("P")
    leg1 = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg1.AddEntry(loss_graph,"loss on train sample","P")
    leg1.AddEntry(val_loss_graph,"loss on test sample","P")
    leg1.Draw()
    output_plot_name = "%s_loss.png" %(output_name)
    img1 = ROOT.TImage.Create()
    img1.FromPad(canv1)
    img1.WriteImage(output_plot_name)
    
    #histogram of di-tau mass using regression all decays
    canv2 = ROOT.TCanvas("di-tau mass using regression all decays")
    max_bin = max(histditaumassregall.GetMaximum(),histditaumassregallout.GetMaximum())
    histditaumassregall.SetMaximum(max_bin*1.08)
    histditaumassregall.Draw()
    histditaumassregallout.Draw("SAME")
    leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg2.AddEntry(histditaumassregallout,"di-#tau_{gen} mass","PL")
    leg2.AddEntry(histditaumassregall,"di-#tau_{NN} mass","PL")
    leg2.Draw()
    """
    pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
    pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
    pad1.SetMargin(0.09,0.02,0.02,0.1)
    pad2.SetMargin(0.09,0.02,0.3,0.02)
    pad1.Draw()
    pad2.Draw()
    pad1.cd()
    max_bin = max(histditaumassregall.GetMaximum(),histditaumassregallout.GetMaximum())
    histditaumassregall.SetMaximum(max_bin*1.08)
    histditaumassregall.Draw()
    histditaumassregallout.Draw("SAME")
    #histditaumassregalloverfit.Draw("SAME")
    leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg2.AddEntry(histditaumassregall,"di-#tau_{gen} mass","PL")
    leg2.AddEntry(histditaumassregallout,"di-#tau_{NN} mass","PL")
    #leg2.AddEntry(histditaumassregalloverfit,"overfit-test","PL")
    leg2.Draw()
    pad2.cd()
    histditaumassregallratio.Draw("P")
    unit_line = ROOT.TLine(0.0,1.0,100.0,1.0)
    unit_line.Draw("SAME")
    """
    output_hist_name = "%s.png" %(output_name)
    img2 = ROOT.TImage.Create()
    img2.FromPad(canv2)
    img2.WriteImage(output_hist_name)

####put wanted model here ####
#SET VERBOSE=2 AND CHECK IF SAVING IS ENABLED BEFORE SUBMITTING TO BATCH
#output_name = "ditau_mass_4Vector_1"
#output_name = "ditau_mass_first_try_without_nupz_7"
output_name = "ditau_mass_4Vector_theta1"
description_of_training = "standardized 30 EPOCHS LR=0.001 INPUT:vis di-tau 4vec+nu px+nu py"
output_file_name = "%s.txt" % (output_name)
output_file = open(output_file_name,'w')
sys.stdout = output_file

batch_size = 30
epochs = 50
learning_rate = 0.001
decay_rate = 0.0


theta_model(20,20,output_name)
#vismass_model_all(20,20,output_name)
#mass_model_all(batch_size,epochs,learning_rate,decay_rate,output_name)
#mass_model_total(batch_size,epochs,output_name)
#mass_model_inclnu(60,20,output_name)
#play_model(15,30,output)
#mass_model_had(10,20,output_name)
#theta_model(20,20,output_name)
#vismass_model_all(20,20,output_name)
#mass_model_total_diff(35,20,output_name)

output_file.close()

"""
#histogram of di-tau px train and test
canv6 = ROOT.TCanvas("di-tau px comparison train and test")
#max_bin6 = max(histpxditautrain.GetMaximum(),histpxditautest.GetMaximum())
#histpxditautrain.SetMaximum(max_bin6*1.08)
histpxditautrain.Draw()
#histpxditautest.Draw("SAME")

#leg6 = ROOT.TLegend(0.6,0.7,0.9,0.9)
#leg6.AddEntry(histpxditautrain,"px train sample","PL")
#leg6.AddEntry(histpxditautest,"px test sample","PL")
#leg6.Draw()

img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage("px_ditau_total.png")

#histogram of di-tau py train and test
canv7 = ROOT.TCanvas("di-tau py comparison train and test")
#max_bin7 = max(histpyditautrain.GetMaximum(),histpyditautest.GetMaximum())
#histpyditautrain.SetMaximum(max_bin7*1.08)
histpyditautrain.Draw()
#histpyditautest.Draw("SAME")

#leg7 = ROOT.TLegend(0.6,0.8,0.9,0.9)
#leg7.AddEntry(histpyditautrain,"py train sample","PL")
#leg7.AddEntry(histpyditautest,"py test sample","PL")
#leg7.Draw()

img7 = ROOT.TImage.Create()
img7.FromPad(canv7)
img7.WriteImage("py_ditau_total.png")

#histogram of di-tau pz train and test
canv8 = ROOT.TCanvas("di-tau pz comparison train and test")
#max_bin8 = max(histpzditautrain.GetMaximum(),histpzditautest.GetMaximum())
#histpzditautrain.SetMaximum(max_bin8*1.08)
histpzditautrain.Draw()
#histpzditautest.Draw("SAME")

#leg8 = ROOT.TLegend(0.6,0.8,0.9,0.9)
#leg8.AddEntry(histpzditautrain,"pz train sample","PL")
#leg8.AddEntry(histpzditautest,"pz test sample","PL")
#leg8.Draw()

img8 = ROOT.TImage.Create()
img8.FromPad(canv8)
img8.WriteImage("pz_ditau_total.png")

#histogram of di-tau energy train and test
canv9 = ROOT.TCanvas("di-tau energy comparison train and test")
#max_bin9 = max(histeditautrain.GetMaximum(),histeditautest.GetMaximum())
#histeditautrain.SetMaximum(max_bin9*1.08)
histeditautrain.Draw()
#histeditautest.Draw("SAME")

#leg9 = ROOT.TLegend(0.6,0.8,0.9,0.9)
#leg9.AddEntry(histeditautrain,"energy train sample","PL")
#leg9.AddEntry(histeditautest,"energy test sample","PL")
#leg9.Draw()

img9 = ROOT.TImage.Create()
img9.FromPad(canv9)
img9.WriteImage("e_ditau_total.png")

canv9.WaitPrimitive()

#histogram of di-tau theta train and test
canv10 = ROOT.TCanvas("di-tau theta comparison train and test")
max_bin10 = max(histthetaditautrain.GetMaximum(),histthetaditautest.GetMaximum())
histthetaditautrain.SetMaximum(max_bin10*1.2)
histthetaditautrain.Draw()
histthetaditautest.Draw("SAME")

leg10 = ROOT.TLegend(0.6,0.8,0.9,0.9)
leg10.AddEntry(histthetaditautrain,"theta train sample","PL")
leg10.AddEntry(histthetaditautest,"theta test sample","PL")
leg10.Draw()

img10 = ROOT.TImage.Create()
img10.FromPad(canv10)
img10.WriteImage("theta_ditau_comp.png")
"""
