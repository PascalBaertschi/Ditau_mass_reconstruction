#!/bin/env python
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
import multiprocessing
from multiprocessing.pool import Pool
"""
dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfit_1e6.csv",delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values
#dataset_length = len(dataset_ditaumass[:,0])
dataset_length = 30000
train_length = int(round(dataset_length*0.7))
test_length = int(round(dataset_length*0.3))

inputNN = []
inputSVfit = []
ditaumass = []
for i in range(0,dataset_length):
    vistau1_pt = dataset_ditaumass[i,0]
    vistau1_eta = dataset_ditaumass[i,1]
    vistau1_phi = dataset_ditaumass[i,2]
    vistau1_mass = dataset_ditaumass[i,3]
    vistau1_att = dataset_ditaumass[i,4]
    vistau1_prongs = dataset_ditaumass[i,5]
    vistau1_pi0 = dataset_ditaumass[i,6] 
    vistau2_pt = dataset_ditaumass[i,7]
    vistau2_eta = dataset_ditaumass[i,8]
    vistau2_phi = dataset_ditaumass[i,9]
    vistau2_mass = dataset_ditaumass[i,10]
    vistau2_att = dataset_ditaumass[i,11]
    vistau2_prongs = dataset_ditaumass[i,12]
    vistau2_pi0 = dataset_ditaumass[i,13]
    nu_pt = dataset_ditaumass[i,14]
    nu_eta = dataset_ditaumass[i,15]
    nu_phi = dataset_ditaumass[i,16]
    nu_mass = dataset_ditaumass[i,17]

    v_vistau1 = ROOT.TLorentzVector()
    v_vistau1.SetPtEtaPhiM(vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass)
    v_vistau2 = ROOT.TLorentzVector()
    v_vistau2.SetPtEtaPhiM(vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass)
    v_nu = ROOT.TLorentzVector()
    v_nu.SetPtEtaPhiM(nu_pt,nu_eta,nu_phi,nu_mass)
    v_mot = v_vistau1+v_vistau2+v_nu
    v_vismot = v_vistau1+v_vistau2
    ditaumass_value = v_mot.M()
    vismass = v_vismot.M()
    p_vis = v_vismot.P()
    pt_vis = v_vismot.Pt()
    pt_nu = v_nu.Pt()
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    #inputNN.append([v_vismot.Px(),v_vismot.Py(),v_vismot.Pz(),v_vismot.E(),v_nu.Px(),v_nu.Py(),p_vis,vismass])
    #inputNN.append([v_vistau1.Px(),v_vistau1.Py(),v_vistau1.Pz(),v_vistau1.E(),v_vistau2.Px(),v_vistau2.Py(),v_vistau2.Pz(),v_vistau2.E(),v_nu.Px(),v_nu.Py(),pt_nu,p_vis,vismass])
    #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
    inputNN.append([v_vistau1.Px(),v_vistau1.Py(),v_vistau1.Pz(),v_vistau1.E(),vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,v_vistau2.Px(),v_vistau2.Py(),v_vistau2.Pz(),v_vistau2.E(),vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),pt_nu,p_vis,vismass,pt_vis])
    ditaumass.append(ditaumass_value)
    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,v_nu.Px(),v_nu.Py()])
inputNN = numpy.array(inputNN,numpy.float64)
ditaumass = numpy.array(ditaumass,numpy.float64)
inputSVfit = numpy.array(inputSVfit,numpy.float64)
inputNN[:,:] = preprocessing.scale(inputNN[:,:])

train_inputNN = inputNN[0:train_length,:]
train_ditaumass = ditaumass[0:train_length]
test_inputNN = inputNN[train_length:dataset_length,:]
test_ditaumass = ditaumass[train_length:dataset_length]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
inputSVfit = inputSVfit[train_length:dataset_length,:]
"""
#print "without split"
#print inputSVfit[0:10,:]


#def easy_function(input1,input2,input3):
#    return input1+input2+input3

def easy_function(input1):
    return input1[0]+input1[1]+input1[2]

def process_easy_function(process_number,split_length):
    sum_split = []
    for g in range(split_length):
        sum_value = easy_function(input_list_split[process_number][g][0],input_list_split[process_number][g][1],input_list_split[process_number][g][2])
        sum_split.append(sum_value)
    q.put(sum_split)

result_queue = multiprocessing.Queue()

nProcesses = 5
#inputSVfit_split = numpy.array_split(inputSVfit[0:100,:], nProcesses)
jobs = []
input_list = []
for i in range(100):
    input_list.append([random.random(),random.random(),random.random()])

#input_list = numpy.array(input_list)
#print "before spliting"
#print input_list
input_list_split = numpy.array_split(input_list,nProcesses)

#print "after splitting"
#print input_list_split
q = multiprocessing.Queue()

args = [[3,5,2],[7,3,3],[2,4,8],[9,1,1]]

pool = multiprocessing.Pool(processes=20)
result_list = pool.map(easy_function,input_list)
print result_list

"""
results = []
for j in range(len(input_list_split)):
    p = multiprocessing.Process(target=process_easy_function, args=(j,len(input_list_split[j])))
    p.start()
    p.join()
    results_loop = q.get()
    for i in range(len(results_loop)):
        results.append(results_loop[i])
print results

"""

"""
nProcesses = 2
inputSVfit_split = numpy.array_split(inputSVfit[0:10,:], nProcesses)
print inputSVfit_split
print inputSVfit_split[0][0][0]
print "length of one split:", len(inputSVfit_split[0])
print "length of split:",len(inputSVfit_split)

def f(name):
    info('function f')
    print 'hello', name

p = Process(target=f, args=('bob',))
p.start()
list = p.join()
"""
