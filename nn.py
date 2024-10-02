#!/bin/env python
import numpy
numpy.random.seed(1337)
#numpy.random.seed(246)
import pandas
import random
import time
import math
import sys
import csv
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout,Activation
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
import multiprocessing


def neural_network(batch_size,epochs,output_name,nninput,nntarget,nninput_test,nntarget_test,nninput_test_100GeV,nntarget_test_100GeV,nninput_test_110GeV,nntarget_test_110GeV,nninput_test_125GeV,nntarget_test_125GeV,nninput_test_140GeV,nntarget_test_140GeV,nninput_test_180GeV,nntarget_test_180GeV,nninput_test_250GeV,nntarget_test_250GeV,nninput_test_dy,nntarget_test_dy):
    print ("NEURAL NETWORK")
    mass_model = Sequential()
    mass_model.add(Dense(200,input_dim=17,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    mass_model.compile(loss='mean_squared_error',optimizer='adam')
    history = mass_model.fit(nninput,nntarget,batch_size,epochs,validation_data = (nninput_test,nntarget_test),verbose = 2)
    mass_score = mass_model.evaluate(nninput_test,nntarget_test,batch_size,verbose=0)
    mass_score100GeV = mass_model.evaluate(nninput_test_100GeV,nntarget_test_100GeV,batch_size,verbose=0)
    mass_score110GeV = mass_model.evaluate(nninput_test_110GeV,nntarget_test_110GeV,batch_size,verbose=0)
    mass_score125GeV = mass_model.evaluate(nninput_test_125GeV,nntarget_test_125GeV,batch_size,verbose=0)
    mass_score140GeV = mass_model.evaluate(nninput_test_140GeV,nntarget_test_140GeV,batch_size,verbose=0)
    mass_score180GeV = mass_model.evaluate(nninput_test_180GeV,nntarget_test_180GeV,batch_size,verbose=0)
    mass_score250GeV = mass_model.evaluate(nninput_test_250GeV,nntarget_test_250GeV,batch_size,verbose=0)
    mass_scoredy = mass_model.evaluate(nninput_test_dy,nntarget_test_dy,batch_size,verbose=0)
    start_predictions = time.time()
    ditaumass_nn = mass_model.predict(nninput_test,batch_size,verbose=0)
    end_predictions = time.time()
    ditaumass_nn_100GeV = mass_model.predict(nninput_test_100GeV,batch_size,verbose=0)
    ditaumass_nn_110GeV = mass_model.predict(nninput_test_110GeV,batch_size,verbose=0)
    ditaumass_nn_125GeV = mass_model.predict(nninput_test_125GeV,batch_size,verbose=0)
    ditaumass_nn_140GeV = mass_model.predict(nninput_test_140GeV,batch_size,verbose=0)
    ditaumass_nn_180GeV = mass_model.predict(nninput_test_180GeV,batch_size,verbose=0)
    ditaumass_nn_250GeV = mass_model.predict(nninput_test_250GeV,batch_size,verbose=0)
    ditaumass_nn_dy = mass_model.predict(nninput_test_dy,batch_size,verbose=0)
    mass_model.summary()
    loss_values = numpy.array(history.history['loss'])
    val_loss_values = numpy.array(history.history['val_loss'])
    print ("mass_model(",batch_size,epochs,")")
    print ("loss (MSE):",mass_score)
    print ("loss (MSE) 100GeV:",mass_score100GeV)
    print ("loss (MSE) 110GeV:",mass_score110GeV)
    print ("loss (MSE) 125GeV:",mass_score125GeV)
    print ("loss (MSE) 140GeV:",mass_score140GeV)
    print ("loss (MSE) 180GeV:",mass_score180GeV)
    print ("loss (MSE) 250GeV:",mass_score250GeV)
    print ("loss (MSE) DY:",mass_scoredy)
    print ("time to predict 100000 events:",end_predictions - start_predictions,"s")
    print (description_of_training)
    nn_output_name = "nnoutput_%s.csv" %(output_name)
    nn_output_100GeV_name = "nnoutput_100GeV_%s.csv" % (output_name)
    nn_output_110GeV_name = "nnoutput_110GeV_%s.csv" % (output_name)
    nn_output_125GeV_name = "nnoutput_125GeV_%s.csv" % (output_name)
    nn_output_140GeV_name = "nnoutput_140GeV_%s.csv" % (output_name)
    nn_output_180GeV_name = "nnoutput_180GeV_%s.csv" % (output_name)
    nn_output_250GeV_name = "nnoutput_250GeV_%s.csv" % (output_name)
    nn_output_dy_name = "nnoutput_dy_%s.csv" % (output_name)
    nn_output_loss_name = "nnoutput_loss_%s.csv" % (output_name)
    nn_output_val_loss_name = "nnoutput_val_loss_%s.csv" % (output_name)
    numpy.savetxt(nn_output_name, ditaumass_nn, delimiter=",")
    numpy.savetxt(nn_output_100GeV_name, ditaumass_nn_100GeV, delimiter=",")
    numpy.savetxt(nn_output_110GeV_name, ditaumass_nn_110GeV, delimiter=",")
    numpy.savetxt(nn_output_125GeV_name, ditaumass_nn_125GeV, delimiter=",")
    numpy.savetxt(nn_output_140GeV_name, ditaumass_nn_140GeV, delimiter=",")
    numpy.savetxt(nn_output_180GeV_name, ditaumass_nn_180GeV, delimiter=",")
    numpy.savetxt(nn_output_250GeV_name, ditaumass_nn_250GeV, delimiter=",")
    numpy.savetxt(nn_output_dy_name,ditaumass_nn_dy, delimiter=",")
    numpy.savetxt(nn_output_loss_name, loss_values, delimiter=",")
    numpy.savetxt(nn_output_val_loss_name, val_loss_values, delimiter=",")

##### get neural network inputs ###############

nninput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_train_nostand_small.csv"
nntarget_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_train_nostand_small.csv"
nninput_test_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_nostand_small.csv"
nntarget_test_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small.csv"
nninput_test_100GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_100GeV_nostand_small.csv"
nninput_test_110GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_110GeV_nostand_small.csv"
nninput_test_125GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_125GeV_nostand_small.csv"
nninput_test_140GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_140GeV_nostand_small.csv"
nninput_test_180GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_180GeV_nostand_small.csv"
nninput_test_250GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_250GeV_nostand_small.csv"
nninput_test_dy_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nninput_test_dy_nostand_small.csv"
nntarget_test_100GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_100GeV_nostand_small.csv"
nntarget_test_110GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_110GeV_nostand_small.csv"
nntarget_test_125GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_125GeV_nostand_small.csv"
nntarget_test_140GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_140GeV_nostand_small.csv"
nntarget_test_180GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_180GeV_nostand_small.csv"
nntarget_test_250GeV_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_250GeV_nostand_small.csv"
nntarget_test_dy_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_dy_nostand_small.csv"

nninput = numpy.array(pandas.read_csv(nninput_name, delim_whitespace=False,header=None))
nntarget = numpy.array(pandas.read_csv(nntarget_name, delim_whitespace=False,header=None))
nninput_test = numpy.array(pandas.read_csv(nninput_test_name, delim_whitespace=False,header=None))
nntarget_test = numpy.array(pandas.read_csv(nntarget_test_name, delim_whitespace=False,header=None))
nninput_test_100GeV = numpy.array(pandas.read_csv(nninput_test_100GeV_name, delim_whitespace=False,header=None))
nninput_test_110GeV = numpy.array(pandas.read_csv(nninput_test_110GeV_name, delim_whitespace=False,header=None))
nninput_test_125GeV = numpy.array(pandas.read_csv(nninput_test_125GeV_name, delim_whitespace=False,header=None))
nninput_test_140GeV = numpy.array(pandas.read_csv(nninput_test_140GeV_name, delim_whitespace=False,header=None))
nninput_test_180GeV = numpy.array(pandas.read_csv(nninput_test_180GeV_name, delim_whitespace=False,header=None))
nninput_test_250GeV = numpy.array(pandas.read_csv(nninput_test_250GeV_name, delim_whitespace=False,header=None))
nninput_test_dy = numpy.array(pandas.read_csv(nninput_test_dy_name, delim_whitespace=False,header=None))
nntarget_test_100GeV = numpy.array(pandas.read_csv(nntarget_test_100GeV_name, delim_whitespace=False,header=None))
nntarget_test_110GeV = numpy.array(pandas.read_csv(nntarget_test_110GeV_name, delim_whitespace=False,header=None))
nntarget_test_125GeV = numpy.array(pandas.read_csv(nntarget_test_125GeV_name, delim_whitespace=False,header=None))
nntarget_test_140GeV = numpy.array(pandas.read_csv(nntarget_test_140GeV_name, delim_whitespace=False,header=None))
nntarget_test_180GeV = numpy.array(pandas.read_csv(nntarget_test_180GeV_name, delim_whitespace=False,header=None))
nntarget_test_250GeV = numpy.array(pandas.read_csv(nntarget_test_250GeV_name, delim_whitespace=False,header=None))
nntarget_test_dy = numpy.array(pandas.read_csv(nntarget_test_dy_name, delim_whitespace=False,header=None))


######################### run neural network  ######################
output_name = "higgs_dy_nostand_29"
description_of_training = "standardized input data  INPUT: 1000 (classification of decaychannel) vis tau1 (pt,eta,phi,E,mass)+vis tau2 (pt,eta,phi,E,mass)+MET_ET+MET_Phi+collinear ditaumass"
output_file_name = "%s.txt" % (output_name)
output_file = open(output_file_name,'w')
sys.stdout = output_file

batch_size = 128
epochs = 400

start_nn = time.time()
neural_network(batch_size,epochs,output_name,nninput,nntarget,nninput_test,nntarget_test,nninput_test_100GeV,nntarget_test_100GeV,nninput_test_110GeV,nntarget_test_110GeV,nninput_test_125GeV,nntarget_test_125GeV,nninput_test_140GeV,nntarget_test_140GeV,nninput_test_180GeV,nntarget_test_180GeV,nninput_test_250GeV,nntarget_test_250GeV, nninput_test_dy, nntarget_test_dy)
end_nn = time.time()

print ("NN executon time:",(end_nn-start_nn)/3600,"h") 

output_file.close()
