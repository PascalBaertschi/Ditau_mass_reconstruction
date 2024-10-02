#!/bin/env python
import numpy
import os, sys
os.environ['OMP_NUM_THREADS'] = "20"
numpy.random.seed(1337)
#numpy.random.seed(246)
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
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
import multiprocessing


#################  choose size of used dataset ######################
#fixed_dataset_length = 2640000
fixed_dataset_length = 1910000
fixed_train_length = 560000
fixed_test_length = 100000
fulllep_fixed_length = int(round(0.1226*fixed_dataset_length))
semilep_fixed_length = int(round(0.4553*fixed_dataset_length))
fullhad_fixed_length = int(round(0.422* fixed_dataset_length))
###################################################################################
list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_skim_correct40.csv"
dataframe_ditaumass = pandas.read_csv(list_name,delim_whitespace=False,header=None)
dataframe_ditaumass_shuffled = dataframe_ditaumass.sample(frac=1,random_state =1337)
dataset_ditaumass = dataframe_ditaumass_shuffled.values
dataset_total_length = len(dataset_ditaumass[:,0])

histwholedatacheck = ROOT.TH1D("wholedatacheck","datacheck",300,0,300)

inputNN = []
ditaumass = []
ditauvismass = []
ditaucollinearmass = []
decaymode_count = 0
decaymode_count_fulllep = 0
decaymode_count_semilep = 0
decaymode_count_fullhad = 0

for i in range(0,dataset_total_length):
    tau1_pt = dataset_ditaumass[i,0]
    tau1_eta = dataset_ditaumass[i,1] 
    tau1_phi = dataset_ditaumass[i,2]
    tau1_mass = dataset_ditaumass[i,3]
    vistau1_pt = dataset_ditaumass[i,4]
    vistau1_eta = dataset_ditaumass[i,5]
    vistau1_phi = dataset_ditaumass[i,6]
    vistau1_mass = dataset_ditaumass[i,7]
    vistau1_att = dataset_ditaumass[i,8]
    vistau1_prongs = dataset_ditaumass[i,9]
    vistau1_pi0 = dataset_ditaumass[i,10]
    tau2_pt = dataset_ditaumass[i,11]
    tau2_eta = dataset_ditaumass[i,12] 
    tau2_phi = dataset_ditaumass[i,13]
    tau2_mass = dataset_ditaumass[i,14]
    vistau2_pt = dataset_ditaumass[i,15]
    vistau2_eta = dataset_ditaumass[i,16]
    vistau2_phi = dataset_ditaumass[i,17]
    vistau2_mass = dataset_ditaumass[i,18]
    vistau2_att = dataset_ditaumass[i,19]
    vistau2_prongs = dataset_ditaumass[i,20]
    vistau2_pi0 = dataset_ditaumass[i,21]
    nu_pt = dataset_ditaumass[i,22]
    nu_eta = dataset_ditaumass[i,23]
    nu_phi = dataset_ditaumass[i,24]
    nu_mass = dataset_ditaumass[i,25]
    genMissingET_MET = dataset_ditaumass[i,26]
    genMissingET_Phi = dataset_ditaumass[i,27]
    MissingET_MET = dataset_ditaumass[i,28]
    MissingET_Phi = dataset_ditaumass[i,29]
    v_tau1 = ROOT.TLorentzVector()
    v_tau1.SetPtEtaPhiM(tau1_pt,tau1_eta,tau1_phi,tau1_mass)
    v_tau2 = ROOT.TLorentzVector()
    v_tau2.SetPtEtaPhiM(tau2_pt,tau2_eta,tau2_phi,tau2_mass)
    v_vistau1 = ROOT.TLorentzVector()
    v_vistau1.SetPtEtaPhiM(vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass)
    v_vistau2 = ROOT.TLorentzVector()
    v_vistau2.SetPtEtaPhiM(vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass)
    v_nu = ROOT.TLorentzVector()
    v_nu.SetPtEtaPhiM(nu_pt,nu_eta,nu_phi,nu_mass)
    v_mot = v_vistau1+v_vistau2+v_nu
    v_vismot = v_vistau1+v_vistau2
    ditaumass_value = v_mot.M()
    ditauvismass_value = v_vismot.M()
    vistau1_E = v_vistau1.E()
    vistau2_E = v_vistau2.E()
    vismass = v_vismot.M()
    p_vis = v_vismot.P()
    pt_vis = v_vismot.Pt()
    pt_nu = v_nu.Pt()
    pt = v_mot.Pt()
    #ditaumass_collinear = vismass/numpy.sqrt(vistau1_pt/(vistau1_pt+pt_nu)*vistau2_pt/(vistau2_pt+pt_nu))
    ditaumass_collinear = vismass/numpy.sqrt(vistau1_pt/(vistau1_pt+genMissingET_MET)*vistau2_pt/(vistau2_pt+genMissingET_MET))
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    histwholedatacheck.Fill(ditaumass_value)
    if decaymode_count < fixed_dataset_length and ditaumass_value > 80.0:
        if vistau1_att in (1,2) and vistau2_att in (1,2) and decaymode_count_fulllep < fulllep_fixed_length:
            #inputNN.append([1.0,0.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            inputNN.append([1.0,0.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            ditaucollinearmass.append(ditaumass_collinear)
            decaymode_count_fulllep += 1
            decaymode_count += 1
        elif vistau1_att in (1,2) and vistau2_att == 3 and decaymode_count_semilep < semilep_fixed_length:
            #inputNN.append([0.0,1.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            inputNN.append([0.0,1.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            ditaucollinearmass.append(ditaumass_collinear)
            decaymode_count_semilep += 1
            decaymode_count += 1
        elif vistau1_att == 3 and vistau2_att in (1,2) and decaymode_count_semilep < semilep_fixed_length:
            #inputNN.append([0.0,0.0,1.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            inputNN.append([0.0,0.0,1.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            ditaucollinearmass.append(ditaumass_collinear)
            decaymode_count_semilep += 1
            decaymode_count += 1
        elif vistau1_att == 3 and vistau2_att == 3 and decaymode_count_fullhad < fullhad_fixed_length:
            #inputNN.append([0.0,0.0,0.0,1.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            inputNN.append([0.0,0.0,0.0,1.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            ditaucollinearmass.append(ditaumass_collinear)
            decaymode_count_fullhad += 1 
            decaymode_count += 1



inputNN = numpy.array(inputNN,numpy.float64)
dataset_length = len(inputNN[:,0])

######## standardizing over the whole mass range #########
#inputNN_stand = preprocessing.scale(inputNN[:,4:])
#for i in range(len(inputNN_stand[0,:])):
#    for j in range(len(inputNN_stand[:,0])):
#        inputNN[j,i+4] = inputNN_stand[j,i]

train_inputNN = inputNN[fixed_test_length:,:]
train_ditaumass = ditaumass[fixed_test_length:]
train_ditauvismass = ditauvismass[fixed_test_length:]
test_inputNN = inputNN[0:fixed_test_length,:]
test_ditaumass = ditaumass[0:fixed_test_length]
test_ditauvismass = ditauvismass[0:fixed_test_length]
test_ditaucollinearmass = ditaucollinearmass[0:fixed_test_length]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
overfit_ditaumass = train_ditaumass[0:len(test_ditaumass)]
overfit_ditauvismass = train_ditauvismass[0:len(test_ditauvismass)]

train_inputNN_forselection = inputNN[fixed_test_length:,:]
train_ditaumass_forselection = ditaumass[fixed_test_length:]
test_inputNN_selected = inputNN[0:fixed_test_length,:]
test_ditaumass_selected = ditaumass[0:fixed_test_length]
test_ditauvismass_selected = ditauvismass[0:fixed_test_length]
test_ditaucollinearmass_selected = ditaucollinearmass[0:fixed_test_length]
######### make flat mass distribution with less events ######################
min_bincontent = 4900
histtrainditaumasscalc = ROOT.TH1D("trainditaumasscalc","trainditaumasscalc",300,0,300)


train_inputNN_selected = []
train_ditaumass_selected = []
train_inputNN_notused = []
train_ditaumass_notused = []

for k,ditaumass_loopvalue in enumerate(train_ditaumass_forselection):
    bin_index = histtrainditaumasscalc.GetXaxis().FindBin(ditaumass_loopvalue)
    bin_content = histtrainditaumasscalc.GetBinContent(bin_index)
    if bin_content < min_bincontent:
        histtrainditaumasscalc.SetBinContent(bin_index,bin_content+1)
        train_inputNN_selected.append(train_inputNN_forselection[k,:])
        train_ditaumass_selected.append(ditaumass_loopvalue)
    else:
        train_inputNN_notused.append(train_inputNN_forselection[k,:])
        train_ditaumass_notused.append(ditaumass_loopvalue)

train_inputNN_selected = numpy.array(train_inputNN_selected,numpy.float64)
train_inputNN_notused = numpy.array(train_inputNN_notused,numpy.float64)
test_inputNN_125GeV = []
test_ditaumass_125GeV = []
test_inputNN_180GeV = []
test_ditaumass_180GeV = []
test_inputNN_250GeV = []
test_ditaumass_250GeV = []
count_125GeV = 0
count_180GeV = 0
count_250GeV = 0
fixed_length_125GeV = 10000
fixed_length_180GeV = 22000
fixed_length_250GeV = 35000


for j,ditaumass_loopvalue in enumerate(train_ditaumass_notused):
    if 124.0 < ditaumass_loopvalue and 126.0 > ditaumass_loopvalue and count_125GeV < fixed_length_125GeV:
        test_inputNN_125GeV.append(train_inputNN_notused[j,:])
        test_ditaumass_125GeV.append(ditaumass_loopvalue)
        count_125GeV += 1
    if 179.0 < ditaumass_loopvalue and 181.0 > ditaumass_loopvalue and count_180GeV < fixed_length_180GeV:
        test_inputNN_180GeV.append(train_inputNN_notused[j,:])
        test_ditaumass_180GeV.append(ditaumass_loopvalue)
        count_180GeV += 1
    if 249.0 < ditaumass_loopvalue and 251.0 > ditaumass_loopvalue and count_250GeV < fixed_length_250GeV:
        test_inputNN_250GeV.append(train_inputNN_notused[j,:])
        test_ditaumass_250GeV.append(ditaumass_loopvalue)
        count_250GeV += 1


for g,ditaumass_loopvalue in enumerate(test_ditaumass_selected):
    if 124.0 < ditaumass_loopvalue and 126.0 > ditaumass_loopvalue and count_125GeV < fixed_length_125GeV:
        test_inputNN_125GeV.append(test_inputNN_selected[g,:])
        test_ditaumass_125GeV.append(ditaumass_loopvalue)
        count_125GeV += 1
    if 179.0 < ditaumass_loopvalue and 181.0 > ditaumass_loopvalue and count_180GeV < fixed_length_180GeV:
        test_inputNN_180GeV.append(test_inputNN_selected[g,:])
        test_ditaumass_180GeV.append(ditaumass_loopvalue)
        count_180GeV += 1
    if 249.0 < ditaumass_loopvalue and 251.0 > ditaumass_loopvalue and count_250GeV < fixed_length_250GeV:
        test_inputNN_250GeV.append(test_inputNN_selected[g,:])
        test_ditaumass_250GeV.append(ditaumass_loopvalue)
        count_250GeV += 1



test_inputNN_125GeV = numpy.array(test_inputNN_125GeV,numpy.float64)
test_inputNN_180GeV = numpy.array(test_inputNN_180GeV,numpy.float64)
test_inputNN_250GeV = numpy.array(test_inputNN_250GeV,numpy.float64)

histtrainditaumasscheck = ROOT.TH1D("trainditaumasscheck","train sample of neural network",350,0,350)
histtrainditaumasscheck.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histtrainditaumasscheck.GetYaxis().SetTitle("number of occurence")
histtrainditaumasscheck.SetStats(0)

for j in train_ditaumass_selected:
    histtrainditaumasscheck.Fill(j)

overfit_inputNN_selected = train_inputNN_selected[0:len(test_ditaumass_selected),:]
overfit_ditaumass_selected = train_ditaumass_selected[0:len(test_ditaumass_selected)]
test_inputNN_125GeV_stand = test_inputNN_125GeV
test_inputNN_180GeV_stand = test_inputNN_180GeV
test_inputNN_250GeV_stand = test_inputNN_250GeV
test_inputNN_selected_stand = test_inputNN_selected


###########     standardization of input data     ##################

for j in range(4,len(train_inputNN_selected[0,:])):
    mean = numpy.mean(train_inputNN_selected[:,j])
    std = numpy.std(train_inputNN_selected[:,j])
    for i in range(0,len(test_ditaumass_selected)):
        value = test_inputNN_selected[i,j]
        new_value = (value-mean)/std
        test_inputNN_selected_stand[i,j] = new_value
    for k in range(0,len(test_ditaumass_125GeV)):
        value = test_inputNN_125GeV[k,j]
        new_value = (value-mean)/std
        test_inputNN_125GeV_stand[k,j] = new_value
    for m in range(0,len(test_ditaumass_180GeV)):
        value = test_inputNN_180GeV[m,j]
        new_value = (value-mean)/std
        test_inputNN_180GeV_stand[m,j] = new_value
    for p in range(0,len(test_ditaumass_250GeV)):
        value = test_inputNN_250GeV[p,j]
        new_value = (value-mean)/std
        test_inputNN_250GeV_stand[p,j] = new_value

test_inputNN_selected = numpy.array(test_inputNN_selected_stand,numpy.float64)        
test_inputNN_125GeV_stand = numpy.array(test_inputNN_125GeV_stand,numpy.float64)
test_inputNN_180GeV_stand = numpy.array(test_inputNN_180GeV_stand,numpy.float64)
test_inputNN_250GeV_stand = numpy.array(test_inputNN_250GeV_stand,numpy.float64)

train_stand = preprocessing.scale(train_inputNN_selected[:,4:])
train_inputNN_set = []
test_inputNN_set = []
for j in range(0,len(train_ditaumass_selected)):
    train_inputNN_set.append([train_inputNN_selected[j,0],train_inputNN_selected[j,1],train_inputNN_selected[j,2],train_inputNN_selected[j,3],train_stand[j,0],train_stand[j,1],train_stand[j,2],train_stand[j,3],train_stand[j,4],train_stand[j,5],train_stand[j,6],train_stand[j,7],train_stand[j,8],train_stand[j,9],train_stand[j,10],train_stand[j,11],train_stand[j,12]])

train_inputNN_selected = numpy.array(train_inputNN_set,numpy.float64)

"""
print "para 0 train mean:",numpy.mean(train_inputNN_selected[:,0]),"std:", numpy.std(train_inputNN_selected[:,0])
print "para 1 train mean:",numpy.mean(train_inputNN_selected[:,1]),"std:", numpy.std(train_inputNN_selected[:,1])
print "para 2 train mean:",numpy.mean(train_inputNN_selected[:,2]),"std:", numpy.std(train_inputNN_selected[:,2])
print "para 3 train mean:",numpy.mean(train_inputNN_selected[:,3]),"std:", numpy.std(train_inputNN_selected[:,3])
print "para 4 train mean:",numpy.mean(train_inputNN_selected[:,4]),"std:", numpy.std(train_inputNN_selected[:,4])
print "para 5 train mean:",numpy.mean(train_inputNN_selected[:,5]),"std:", numpy.std(train_inputNN_selected[:,5])
print "para 6 train mean:",numpy.mean(train_inputNN_selected[:,6]),"std:", numpy.std(train_inputNN_selected[:,6])
print "para 7 train mean:",numpy.mean(train_inputNN_selected[:,7]),"std:", numpy.std(train_inputNN_selected[:,7])
print "para 8 train mean:",numpy.mean(train_inputNN_selected[:,8]),"std:", numpy.std(train_inputNN_selected[:,8])
print "para 9 train mean:",numpy.mean(train_inputNN_selected[:,9]),"std:", numpy.std(train_inputNN_selected[:,9])
print "para 10 train mean:",numpy.mean(train_inputNN_selected[:,10]),"std:", numpy.std(train_inputNN_selected[:,10])
print "para 11 train mean:",numpy.mean(train_inputNN_selected[:,11]),"std:", numpy.std(train_inputNN_selected[:,11])
print "para 12 train mean:",numpy.mean(train_inputNN_selected[:,12]),"std:", numpy.std(train_inputNN_selected[:,12])
print "para 13 train mean:",numpy.mean(train_inputNN_selected[:,13]),"std:", numpy.std(train_inputNN_selected[:,13])
print "para 14 train mean:",numpy.mean(train_inputNN_selected[:,14]),"std:", numpy.std(train_inputNN_selected[:,14])
print "para 15 train mean:",numpy.mean(train_inputNN_selected[:,15]),"std:", numpy.std(train_inputNN_selected[:,15])
print "para 16 train mean:",numpy.mean(train_inputNN_selected[:,16]),"std:", numpy.std(train_inputNN_selected[:,16])
print "para 0 test mean:",numpy.mean(test_inputNN_selected[:,0]),"std:", numpy.std(test_inputNN_selected[:,0])
print "para 1 test mean:",numpy.mean(test_inputNN_selected[:,1]),"std:", numpy.std(test_inputNN_selected[:,1])
print "para 2 test mean:",numpy.mean(test_inputNN_selected[:,2]),"std:", numpy.std(test_inputNN_selected[:,2])
print "para 3 test mean:",numpy.mean(test_inputNN_selected[:,3]),"std:", numpy.std(test_inputNN_selected[:,3])
print "para 4 test mean:",numpy.mean(test_inputNN_selected[:,4]),"std:", numpy.std(test_inputNN_selected[:,4])
print "para 5 test mean:",numpy.mean(test_inputNN_selected[:,5]),"std:", numpy.std(test_inputNN_selected[:,5])
print "para 6 test mean:",numpy.mean(test_inputNN_selected[:,6]),"std:", numpy.std(test_inputNN_selected[:,6])
print "para 7 test mean:",numpy.mean(test_inputNN_selected[:,7]),"std:", numpy.std(test_inputNN_selected[:,7])
print "para 8 test mean:",numpy.mean(test_inputNN_selected[:,8]),"std:", numpy.std(test_inputNN_selected[:,8])
print "para 9 test mean:",numpy.mean(test_inputNN_selected[:,9]),"std:", numpy.std(test_inputNN_selected[:,9])
print "para 10 test mean:",numpy.mean(test_inputNN_selected[:,10]),"std:", numpy.std(test_inputNN_selected[:,10])
print "para 11 test mean:",numpy.mean(test_inputNN_selected[:,11]),"std:", numpy.std(test_inputNN_selected[:,11])
print "para 12 test mean:",numpy.mean(test_inputNN_selected[:,12]),"std:", numpy.std(test_inputNN_selected[:,12])
print "para 13 test mean:",numpy.mean(test_inputNN_selected[:,13]),"std:", numpy.std(test_inputNN_selected[:,13])
print "para 14 test mean:",numpy.mean(test_inputNN_selected[:,14]),"std:", numpy.std(test_inputNN_selected[:,14])
print "para 15 test mean:",numpy.mean(test_inputNN_selected[:,15]),"std:", numpy.std(test_inputNN_selected[:,15])
print "para 16 test mean:",numpy.mean(test_inputNN_selected[:,16]),"std:", numpy.std(test_inputNN_selected[:,16])
print "para 0  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,0]),"std:", numpy.std(test_inputNN_125GeV_stand[:,0])
print "para 1  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,1]),"std:", numpy.std(test_inputNN_125GeV_stand[:,1])
print "para 2  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,2]),"std:", numpy.std(test_inputNN_125GeV_stand[:,2])
print "para 3  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,3]),"std:", numpy.std(test_inputNN_125GeV_stand[:,3])
print "para 4  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,4]),"std:", numpy.std(test_inputNN_125GeV_stand[:,4])
print "para 5  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,5]),"std:", numpy.std(test_inputNN_125GeV_stand[:,5])
print "para 6  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,6]),"std:", numpy.std(test_inputNN_125GeV_stand[:,6])
print "para 7  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,7]),"std:", numpy.std(test_inputNN_125GeV_stand[:,7])
print "para 8  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,8]),"std:", numpy.std(test_inputNN_125GeV_stand[:,8])
print "para 9  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,9]),"std:", numpy.std(test_inputNN_125GeV_stand[:,9])
print "para 10  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,10]),"std:", numpy.std(test_inputNN_125GeV_stand[:,10])
print "para 11  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,11]),"std:", numpy.std(test_inputNN_125GeV_stand[:,11])
print "para 12  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,12]),"std:", numpy.std(test_inputNN_125GeV_stand[:,12])
print "para 13  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,13]),"std:", numpy.std(test_inputNN_125GeV_stand[:,13])
print "para 14  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,14]),"std:", numpy.std(test_inputNN_125GeV_stand[:,14])
print "para 15  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,15]),"std:", numpy.std(test_inputNN_125GeV_stand[:,15])
print "para 16  125GeV mean:",numpy.mean(test_inputNN_125GeV_stand[:,16]),"std:", numpy.std(test_inputNN_125GeV_stand[:,16])
print "para 0  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,0]),"std:", numpy.std(test_inputNN_250GeV_stand[:,0])
print "para 1  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,1]),"std:", numpy.std(test_inputNN_250GeV_stand[:,1])
print "para 2  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,2]),"std:", numpy.std(test_inputNN_250GeV_stand[:,2])
print "para 3  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,3]),"std:", numpy.std(test_inputNN_250GeV_stand[:,3])
print "para 4  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,4]),"std:", numpy.std(test_inputNN_250GeV_stand[:,4])
print "para 5  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,5]),"std:", numpy.std(test_inputNN_250GeV_stand[:,5])
print "para 6  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,6]),"std:", numpy.std(test_inputNN_250GeV_stand[:,6])
print "para 7  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,7]),"std:", numpy.std(test_inputNN_250GeV_stand[:,7])
print "para 8  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,8]),"std:", numpy.std(test_inputNN_250GeV_stand[:,8])
print "para 9  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,9]),"std:", numpy.std(test_inputNN_250GeV_stand[:,9])
print "para 10  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,10]),"std:", numpy.std(test_inputNN_250GeV_stand[:,10])
print "para 11  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,11]),"std:", numpy.std(test_inputNN_250GeV_stand[:,11])
print "para 12  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,12]),"std:", numpy.std(test_inputNN_250GeV_stand[:,12]
print "para 13  250GeV mean:",numpy.mn(test_inputNN_250GeV_stand[:,13]),"std:", numpy.std(test_inputNN_250GeV_stand[:,13])
print "para 14  250GeV mean:",numy.mean(test_inputNN_250GeV_stand[:,14]),"std:", numpy.std(test_inputNN_250GeV_stand[:,14])
print "para 15  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,15]),"std:", numpy.std(test_inputNN_250GeV_stand[:,15])
print "para 16  250GeV mean:",numpy.mean(test_inputNN_250GeV_stand[:,16]),"std:", numpy.std(test_inputNN_250GeV_stand[:,16])
"""

#############    preparing the histograms       ###################
ROOT.TGaxis.SetMaxDigits(3)
#histogram of ditau mass using neural network and SVfit
histtitle = "reconstruct di-#tau mass using a neural network and SVfit"
histditaumass = ROOT.TH1D("ditaumass",histtitle,70,0,350)
histditaumass.SetTitleSize(0.3,"t")
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleSize(0.049)
histditaumass.GetYaxis().SetTitleOffset(0.75)
histditaumass.GetYaxis().SetLabelSize(0.049)
histditaumass.SetLineColor(2)
histditaumass.SetLineWidth(3)
histditaumass.SetStats(0)
#histditaumassgen = ROOT.TH1D("ditaumassgen","di-#tau_{gen} mass and visible mass",70,0,350)
histditaumassgen = ROOT.TH1D("ditaumassgen","mass distribution of di-#tau system",80,0,400)
histditaumassgen.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassgen.GetYaxis().SetTitle("number of occurence")
histditaumassgen.SetLineColor(2)
histditaumassgen.SetLineWidth(3)
histditaumassgen.SetStats(0)
#histditauvismass = ROOT.TH1D("ditauvismass","reconstructed di-#tau vismass using neural network",70,0,350)
histditauvismass = ROOT.TH1D("ditauvismass","reconstructed di-#tau vismass using neural network",80,0,400)
histditauvismass.SetLineColor(6)
histditauvismass.SetLineWidth(3)
histditauvismass.SetLineStyle(7)
histditauvismass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassnn","reconstructed di-#tau mass using neural network",70,0,350)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetLineWidth(3)
histditaumassnn.SetLineStyle(7)
histditaumassnn.SetStats(0)
#histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",70,0,350)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",80,0,400)
histditaumasssvfit.SetLineColor(8)
histditaumasssvfit.SetLineWidth(3)
histditaumasssvfit.SetLineStyle(2)
histditaumasssvfit.SetStats(0)
histditaumassnnoverfit = ROOT.TH1D("ditaumassnnoverfit","overfit of neural network",70,0,350)
histditaumassnnoverfit.SetLineColor(3)
histditaumassnnoverfit.SetLineStyle(2)
histditaumassnnoverfit.SetStats(0)
histditaucollmass = ROOT.TH1D("ditaucollmass","collinear ditau mass",80,0,400)
histditaucollmass.SetLineColor(7)
histditaucollmass.SetLineWidth(3)
histditaucollmass.SetLineStyle(3)
histditaucollmass.SetStats(0)

histditaumassnncorr = ROOT.TH2D("ditaumassnncorr","di-#tau_{gen} mass vs di-#tau mass",400,0,400,400,0,400)
histditaumassnncorr.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnncorr.GetYaxis().SetTitle("di-#tau mass [GeV]")
histditaumassnncorr.SetStats(0)
histditaumasssvfitcorr = ROOT.TH2D("ditaumasssvfitcorr","di-#tau_{gen} mass vs di-#tau mass",400,0,400,400,0,400)
histditaumasssvfitcorr.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumasssvfitcorr.GetYaxis().SetTitle("di-#tau mass [GeV]")
histditaumasssvfitcorr.SetStats(0)

profditaumassnncorrrms = ROOT.TProfile("profditaumassnncorrrms","di-#tau_{gen} mass vs mean(di-#tau mass)",350,0,350,"s")

histditaumassnnrms =ROOT.TH1D("ditaumassnnrms","RMS per di-#tau_{gen} mass",350,0,350)
histditaumassnnrms.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnnrms.GetYaxis().SetTitle("RMS")
histditaumassnnrms.SetStats(0)
histditaumassnnrms.SetMarkerStyle(7)

histditaumassnnres = ROOT.TH1D("resolution","relative difference per event using neural network",80,-1,1)
histditaumassnnres.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres.GetYaxis().SetTitle("number of occurence")
histditaumassnnres.SetLineWidth(3)
histditaumassnnrescomp = ROOT.TH1D("NN","relative difference per event",80,-1,1)
histditaumassnnrescomp.GetXaxis().SetTitle("relative difference per event")
histditaumassnnrescomp.GetYaxis().SetTitle("number of occurence")
histditaumassnnrescomp.SetLineColor(4)
histditaumassnnrescomp.SetLineWidth(3)
histditaumasssvfitres = ROOT.TH1D("SVfit","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres.SetLineColor(8)
histditaumasssvfitres.SetLineWidth(3)


histditaumassnncorrres = ROOT.TH2D("ditaumassnncorrres","relative difference per event",350,0,350,80,-1,1)
histditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnncorrres.GetYaxis().SetTitle("relative difference per event")
histditaumassnncorrres.SetStats(0)
histditaumasssvfitcorrres = ROOT.TH2D("ditaumasssvfitcorrres","relative difference per event",350,0,350,80,-1,1)
histditaumasssvfitcorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumasssvfitcorrres.GetYaxis().SetTitle("relative difference per event")
histditaumasssvfitcorrres.SetStats(0)

profditaumassnncorrres = ROOT.TProfile("profditaumassnncorrres","bias in the di-#tau mass reconstruction",350,0,350)
profditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumassnncorrres.GetYaxis().SetTitle("bias")
profditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
profditaumassnncorrres.SetStats(0)
profditaumassnncorrres.SetMarkerStyle(7)
histprofditaumassnncorrres = ROOT.TH1D("histprofditaumassnncorrres","bias in the di-#tau mass reconstruction",350,0,350)
histprofditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histprofditaumassnncorrres.GetYaxis().SetTitle("bias")
histprofditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
histprofditaumassnncorrres.SetStats(0)
histprofditaumassnncorrres.SetLineColor(4)
histprofditaumassnncorrres.SetMarkerStyle(7)
histprofditaumassnncorrres.SetMarkerColor(4)
histprofditaumasssvfitcorrres = ROOT.TH1D("histprofditaumasssvfitcorrres","mean of resolution per di-#tau_{gen} mass",350,0,350)
histprofditaumasssvfitcorrres.SetStats(0)
histprofditaumasssvfitcorrres.SetLineColor(8)
histprofditaumasssvfitcorrres.SetMarkerStyle(7)
histprofditaumasssvfitcorrres.SetMarkerColor(8)

profditaumassnncorrabsres = ROOT.TProfile("profditaumassnncorrabsres","average of the absolute relative differences per event",350,0,350)
profditaumassnncorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumassnncorrabsres.GetYaxis().SetTitle("average of the absolute relative difference per event")
profditaumassnncorrabsres.GetYaxis().SetTitleOffset(1.2)
profditaumassnncorrabsres.SetStats(0)
profditaumassnncorrabsres.SetMarkerStyle(7)
histprofditaumassnncorrabsres = ROOT.TH1D("histprofditaumassnncorrabsres","average of the absolute relative differences per event",350,0,350)
histprofditaumassnncorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histprofditaumassnncorrabsres.GetYaxis().SetTitle("average of the absolute relative differences per event")
histprofditaumassnncorrabsres.GetYaxis().SetTitleOffset(1.2)
histprofditaumassnncorrabsres.SetStats(0)
histprofditaumassnncorrabsres.SetLineColor(4)
histprofditaumassnncorrabsres.SetMarkerStyle(7)
histprofditaumassnncorrabsres.SetMarkerColor(4)
histprofditaumasssvfitcorrabsres = ROOT.TH1D("histprofditaumasssvfitcorrabsres","mean of |resolution| per di-#tau_{gen} mass",350,0,350)
histprofditaumasssvfitcorrabsres.SetStats(0)
histprofditaumasssvfitcorrabsres.SetLineColor(8)
histprofditaumasssvfitcorrabsres.SetMarkerStyle(7)
histprofditaumasssvfitcorrabsres.SetMarkerColor(8)

#ratio histograms
histditaumassnnratio = ROOT.TH1D("ditaumassregallratio","ratio between reconstruced and actual mass",70,0,350)
histditaumassnnratio.SetTitle("")
histditaumassnnratio.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnnratio.GetXaxis().SetLabelSize(0.115)
histditaumassnnratio.GetXaxis().SetTitleSize(0.115)
#histditaumassnnratio.GetXaxis().SetLabelSize(0.1)
#histditaumassnnratio.GetXaxis().SetTitleSize(0.1)
histditaumassnnratio.GetYaxis().SetTitle("ratio")
histditaumassnnratio.GetYaxis().SetLabelSize(0.115)
histditaumassnnratio.GetYaxis().SetTitleSize(0.115)
#histditaumassnnratio.GetYaxis().SetLabelSize(0.08)
#histditaumassnnratio.GetYaxis().SetTitleSize(0.08)
histditaumassnnratio.GetYaxis().SetTitleOffset(0.3)
histditaumassnnratio.GetYaxis().SetNdivisions(404)
histditaumassnnratio.GetYaxis().CenterTitle()
histditaumassnnratio.GetYaxis().SetRangeUser(0.0,2.0)
histditaumassnnratio.SetMarkerStyle(7)
histditaumassnnratio.SetMarkerColor(4)
histditaumassnnratio.SetStats(0)
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and actual mass",70,0,350)
histditaumasssvfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(8)
histditaumasssvfitratio.SetStats(0)
histditaumassnnoverfitratio = ROOT.TH1D("ditaumassregalloverfitratio","ratio between reconstruced and overfit mass",70,0,350)
histditaumassnnoverfitratio.SetMarkerStyle(7)
histditaumassnnoverfitratio.SetMarkerColor(8)
histditaumassnnoverfitratio.SetStats(0)


#histogram of ditau mass using neural network and SVfit 125 GeV
histditaumass125GeV = ROOT.TH1D("ditaumass125GeV","reconstruct di-#tau mass using neural network and SVfit",100,95,155)
histditaumass125GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass125GeV.GetYaxis().SetTitle("number of occurence")
histditaumass125GeV.SetLineColor(2)
histditaumass125GeV.SetLineWidth(3)
histditaumass125GeV.SetStats(0)
histditaumassnn125GeV = ROOT.TH1D("ditaumassnn125GeV","reconstruct di-#tau mass using neural network",100,95,155)
histditaumassnn125GeV.SetLineColor(4)
histditaumassnn125GeV.SetLineWidth(3)
histditaumassnn125GeV.SetLineStyle(7)
histditaumassnn125GeV.SetStats(0)
histditaumasssvfit125GeV = ROOT.TH1D("ditaumasssvfit125GeV","di-#tau mass using SVfit",100,95,155)
histditaumasssvfit125GeV.SetLineColor(8)
histditaumasssvfit125GeV.SetLineWidth(3)
histditaumasssvfit125GeV.SetLineStyle(2)
histditaumasssvfit125GeV.SetStats(0)
histditaumassnnres125GeV = ROOT.TH1D("NN125GeV","relative difference per event",80,-1,1)
histditaumassnnres125GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres125GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres125GeV.SetLineColor(4)
histditaumassnnres125GeV.SetLineWidth(3)
histditaumasssvfitres125GeV = ROOT.TH1D("SVfit125GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres125GeV.SetLineColor(8)
histditaumasssvfitres125GeV.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit 250 GeV
histditaumass250GeV = ROOT.TH1D("ditaumass250GeV","reconstruct di-#tau mass using neural network and SVfit",100,220,280)
histditaumass250GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass250GeV.GetYaxis().SetTitle("number of occurence")
histditaumass250GeV.SetLineColor(2)
histditaumass250GeV.SetLineWidth(3)
histditaumass250GeV.SetStats(0)
histditaumassnn250GeV = ROOT.TH1D("ditaumassnn250GeV","reconstructed di-#tau mass using neural network",100,220,280)
histditaumassnn250GeV.SetLineColor(4)
histditaumassnn250GeV.SetLineWidth(3)
histditaumassnn250GeV.SetLineStyle(7)
histditaumassnn250GeV.SetStats(0)
histditaumasssvfit250GeV = ROOT.TH1D("ditaumasssvfit250GeV","di-#tau mass using SVfit",100,220,280)
histditaumasssvfit250GeV.SetLineColor(8)
histditaumasssvfit250GeV.SetLineWidth(3)
histditaumasssvfit250GeV.SetLineStyle(2)
histditaumasssvfit250GeV.SetStats(0)
histditaumassnnres250GeV = ROOT.TH1D("NN250GeV","relative difference per event",80,-1,1)
histditaumassnnres250GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres250GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres250GeV.SetLineColor(4)
histditaumassnnres250GeV.SetLineWidth(3)
histditaumasssvfitres250GeV = ROOT.TH1D("SVfit250GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres250GeV.SetLineColor(8)
histditaumasssvfitres250GeV.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit 180 GeV
histditaumass180GeV = ROOT.TH1D("ditaumass180GeV","reconstruct di-#tau mass using neural network and SVfit",100,150,210)
histditaumass180GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass180GeV.GetYaxis().SetTitle("number of occurence")
histditaumass180GeV.SetLineColor(2)
histditaumass180GeV.SetLineWidth(3)
histditaumass180GeV.SetStats(0)
histditaumassnn180GeV = ROOT.TH1D("ditaumassnn180GeV","reconstructed di-#tau mass using neural network",100,150,210)
histditaumassnn180GeV.SetLineColor(4)
histditaumassnn180GeV.SetLineWidth(3)
histditaumassnn180GeV.SetLineStyle(7)
histditaumassnn180GeV.SetStats(0)
histditaumasssvfit180GeV = ROOT.TH1D("ditaumasssvfit180GeV","di-#tau mass using SVfit",100,150,210)
histditaumasssvfit180GeV.SetLineColor(8)
histditaumasssvfit180GeV.SetLineWidth(3)
histditaumasssvfit180GeV.SetLineStyle(2)
histditaumasssvfit180GeV.SetStats(0)
histditaumassnnres180GeV = ROOT.TH1D("NN180GeV","relative difference per event",80,-1,1)
histditaumassnnres180GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres180GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres180GeV.SetLineColor(4)
histditaumassnnres180GeV.SetLineWidth(3)
histditaumasssvfitres180GeV = ROOT.TH1D("SVfit180GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres180GeV.SetLineColor(8)
histditaumasssvfitres180GeV.SetLineWidth(3)

def neural_network(batch_size,epochs,output_name):
    print "NEURAL NETWORK"
    mass_model = Sequential()
    mass_model.add(Dense(200,input_dim=17,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    mass_model.compile(loss='mean_squared_error',optimizer='adam')
    history = mass_model.fit(train_inputNN_selected,train_ditaumass_selected,batch_size,epochs,validation_data = (test_inputNN_selected,test_ditaumass_selected),verbose = 2)
    mass_score = mass_model.evaluate(test_inputNN_selected,test_ditaumass_selected,batch_size,verbose=0)
    mass_score125GeV = mass_model.evaluate(test_inputNN_125GeV_stand,test_ditaumass_125GeV,batch_size,verbose=0)
    mass_score180GeV = mass_model.evaluate(test_inputNN_180GeV_stand,test_ditaumass_180GeV,batch_size,verbose=0)
    mass_score250GeV = mass_model.evaluate(test_inputNN_250GeV_stand,test_ditaumass_250GeV,batch_size,verbose=0)
    start_predictions = time.time()
    ditaumass_nn = mass_model.predict(test_inputNN_selected,batch_size,verbose=0)
    end_predictions = time.time()
    ditaumass_nn_125GeV = mass_model.predict(test_inputNN_125GeV_stand,batch_size,verbose=0)
    ditaumass_nn_180GeV = mass_model.predict(test_inputNN_180GeV_stand,batch_size,verbose=0)
    ditaumass_nn_250GeV = mass_model.predict(test_inputNN_250GeV_stand,batch_size,verbose=0)
    mass_model.summary()
    mass_model.save('nnmodel_thesis_MET')
    print "mass_model(",batch_size,epochs,")"
    print "loss (MSE):",mass_score
    print "loss (MSE) 125GeV:",mass_score125GeV
    print "loss (MSE) 180GeV:",mass_score180GeV
    print "loss (MSE) 250GeV:",mass_score250GeV
    print "time to predict 100000 events:",end_predictions - start_predictions,"s"
    print description_of_training
    failed_division = 0
    #preparing the histograms
    for j in ditaumass_nn:
        histditaumassnn.Fill(j)
    for s,ditaumass125GeV_value in enumerate(test_ditaumass_125GeV):
        histditaumassnn125GeV.Fill(ditaumass_nn_125GeV[s])
        res = (ditaumass125GeV_value - ditaumass_nn_125GeV[s])/ditaumass125GeV_value
        histditaumassnnres125GeV.Fill(res)
    for y,ditaumass180GeV_value in enumerate(test_ditaumass_180GeV):
        histditaumassnn180GeV.Fill(ditaumass_nn_180GeV[y])
        res = (ditaumass180GeV_value - ditaumass_nn_180GeV[y])/ditaumass180GeV_value
        histditaumassnnres180GeV.Fill(res)
    for x,ditaumass250GeV_value in enumerate(test_ditaumass_250GeV):
        histditaumassnn250GeV.Fill(ditaumass_nn_250GeV[x])
        res = (ditaumass250GeV_value - ditaumass_nn_250GeV[x])/ditaumass250GeV_value
        histditaumassnnres250GeV.Fill(res)
    #for d in ditaumass_nn_overfit:
    #    histditaumassnnoverfit.Fill(d)
    #for i,ditaumass_value in enumerate(test_ditaumass):
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        #for i,ditaumass_value in enumerate(ditaumass_cuts):
        res = (ditaumass_value - ditaumass_nn[i])/ditaumass_value
        histditaumassnnres.Fill(res)
        histditaumassnnrescomp.Fill(res)
        histditaumassnncorrres.Fill(ditaumass_value,res)
        profditaumassnncorrres.Fill(ditaumass_value,res)
        profditaumassnncorrabsres.Fill(ditaumass_value,abs(res))
    for g in range(len(ditaumass_nn)):
        #histditaumassnncorr.Fill(test_ditaumass[g],ditaumass_nn[g])
        histditaumassnncorr.Fill(test_ditaumass_selected[g],ditaumass_nn[g])
        profditaumassnncorrrms.Fill(test_ditaumass_selected[g],ditaumass_nn[g])
        #histditaumassnncorr.Fill(ditaumass_cuts[g],ditaumass_nn[g])
    for j in range(350):
        rms = profditaumassnncorrrms.GetBinError(j+1)
        histditaumassnnrms.SetBinContent(j+1,rms)
    for k in range(70):
        if histditaumass.GetBinContent(k+1) != 0:
            content_nn = histditaumassnn.GetBinContent(k+1)
            content_actual = histditaumass.GetBinContent(k+1)
            ratio = content_nn/content_actual
            error_nn = numpy.sqrt(content_nn)
            error_actual = numpy.sqrt(content_actual)
            error_ratio = ratio*numpy.sqrt((error_actual/content_actual)**2+(error_nn/content_nn)**2)
            histditaumassnnratio.SetBinError(k+1,error_ratio)
            histditaumassnnratio.SetBinContent(k+1,ratio)
        elif histditaumassnn.GetBinContent(k+1) == 0 and histditaumass.GetBinContent(k+1) == 0:
            histditaumassnnratio.SetBinContent(k+1,1.0)
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
    leg1 = ROOT.TLegend(0.6,0.7,0.87,0.87)
    leg1.AddEntry(loss_graph,"loss on train sample","P")
    leg1.AddEntry(val_loss_graph,"loss on test sample","P")
    leg1.Draw()
    output_plot_name = "%s_loss.png" %(output_name)
    loss_graph.Write()
    val_loss_graph.Write()
    canv1.Write()
    img1 = ROOT.TImage.Create()
    img1.FromPad(canv1)
    img1.WriteImage(output_plot_name)

        
def fill_histditaumass():
    for i,ditaumass_value in enumerate(test_ditaumass):
        histditaumass.Fill(ditaumass_value)

def fill_histditauvismass():
    for f,ditauvismass_value in enumerate(test_ditauvismass):
        histditauvismass.Fill(ditauvismass_value)

def fill_histditaucollinearmass():
    for d,ditaucollinearmass_value in enumerate(test_ditaucollinearmass):
        histditaucollmass.Fill(ditaucollinearmass_value)

def fill_histditaumass_selected():
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        histditaumass.Fill(ditaumass_value)
        histditaumassgen.Fill(ditaumass_value)

def fill_histditauvismass_selected():
    for i,ditauvismass_value in enumerate(test_ditauvismass_selected):
        histditauvismass.Fill(ditauvismass_value)

def fill_histditaucollinearmass_selected():
    for d,ditaucollinearmass_value in enumerate(test_ditaucollinearmass_selected):
        histditaucollmass.Fill(ditaucollinearmass_value)

def fill_histprofres():
    for h in range(350):
        corrres = profditaumassnncorrres.GetBinContent(h+1)
        corrres_error = profditaumassnncorrres.GetBinError(h+1)
        histprofditaumassnncorrres.SetBinContent(h+1,corrres)
        histprofditaumassnncorrres.SetBinError(h+1,corrres_error)

def fill_histprofabsres():
    for i in range(350):
        corrabsres = profditaumassnncorrabsres.GetBinContent(i+1)
        corrabsres_error = profditaumassnncorrabsres.GetBinError(i+1)
        histprofditaumassnncorrabsres.SetBinContent(i+1,corrabsres)
        histprofditaumassnncorrabsres.SetBinError(i+1,corrabsres_error)
def fill_histditaumass_125GeV():
    for i,ditaumass125GeV_value in enumerate(test_ditaumass_125GeV):
        histditaumass125GeV.Fill(ditaumass125GeV_value)
def fill_histditaumass_180GeV():
    for i,ditaumass180GeV_value in enumerate(test_ditaumass_180GeV):
        histditaumass180GeV.Fill(ditaumass180GeV_value)
def fill_histditaumass_250GeV():
    for i,ditaumass250GeV_value in enumerate(test_ditaumass_250GeV):
        histditaumass250GeV.Fill(ditaumass250GeV_value)

#####################  getting the SVfit histograms ####################
svfit = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/final_plots/ditau_mass_svfit_cuts_1e5_try7.root")
histsvfit = svfit.Get("ditaumasssvfit")
histsvfitratio = svfit.Get("ditaumasssvfitratio")
histsvfitres = svfit.Get("resolution")
histsvfitcorr = svfit.Get("ditaumasssvfitcorr")
histsvfitcorrres = svfit.Get("ditaumasssvfitcorrres")
profsvfitcorrres = svfit.Get("profditaumasssvfitcorrres")
profsvfitcorrabsres = svfit.Get("profditaumasssvfitcorrabsres")

for i in range(70):
    binvalue = histsvfit.GetBinContent(i+1)
    ratio = histsvfitratio.GetBinContent(i+1)
    ratio_error = histsvfitratio.GetBinError(i+1)
    histditaumasssvfit.SetBinContent(i+1,binvalue)
    histditaumasssvfitratio.SetBinContent(i+1,ratio)
    histditaumasssvfitratio.SetBinError(i+1,ratio_error)
for j in range(80):
    resvalue = histsvfitres.GetBinContent(j+1)
    histditaumasssvfitres.SetBinContent(j+1,resvalue)
for h in range(350):
    corrres = profsvfitcorrres.GetBinContent(h+1)
    corrres_error = profsvfitcorrres.GetBinError(h+1)
    corrabsres = profsvfitcorrabsres.GetBinContent(h+1)
    corrabsres_error = profsvfitcorrabsres.GetBinError(h+1)
    histprofditaumasssvfitcorrres.SetBinContent(h+1,corrres)
    histprofditaumasssvfitcorrres.SetBinError(h+1,corrres_error)
    histprofditaumasssvfitcorrabsres.SetBinContent(h+1,corrabsres)
    histprofditaumasssvfitcorrabsres.SetBinError(h+1,corrabsres_error)
for g in range(350):
    for f in range(80):
        binvalue_corrres = histsvfitcorrres.GetBinContent(g+1,f+1)
        histditaumasssvfitcorrres.SetBinContent(g+1,f+1,binvalue_corrres)
for r in range(400):
    for t in range(400):
        binvalue_corr = histsvfitcorr.GetBinContent(r+1,t+1)
        histditaumasssvfitcorr.SetBinContent(r+1,t+1,binvalue_corr)

svfit.Close()
svfit125GeV = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/final_plots/ditau_mass_svfit_cuts_125GeV_1e4.root")
histsvfit125GeV = svfit125GeV.Get("ditaumasssvfit")
histsvfitres125GeV = svfit125GeV.Get("svfitresolution")
for i in range(100):
    binvalue = histsvfit125GeV.GetBinContent(i+1)
    histditaumasssvfit125GeV.SetBinContent(i+1,binvalue)
for j in range(80):
    resvalue = histsvfitres125GeV.GetBinContent(j+1)
    histditaumasssvfitres125GeV.SetBinContent(j+1,resvalue)
svfit125GeV.Close()
svfit180GeV = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/final_plots/ditau_mass_svfit_cuts_180GeV_22e3.root")
histsvfit180GeV = svfit180GeV.Get("ditaumasssvfit")
histsvfitres180GeV = svfit180GeV.Get("resolution")
for i in range(100):
    binvalue = histsvfit180GeV.GetBinContent(i+1)
    histditaumasssvfit180GeV.SetBinContent(i+1,binvalue)
for j in range(80):
    resvalue = histsvfitres180GeV.GetBinContent(j+1)
    histditaumasssvfitres180GeV.SetBinContent(j+1,resvalue)
svfit180GeV.Close()
svfit250GeV = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/final_plots/ditau_mass_svfit_cuts_250GeV_35e3_20.root")
histsvfit250GeV = svfit250GeV.Get("ditaumasssvfit")
histsvfitres250GeV = svfit250GeV.Get("resolution")
for i in range(100):
    binvalue = histsvfit250GeV.GetBinContent(i+1)
    histditaumasssvfit250GeV.SetBinContent(i+1,binvalue)
for j in range(80):
    resvalue = histsvfitres250GeV.GetBinContent(j+1)
    histditaumasssvfitres250GeV.SetBinContent(j+1,resvalue)
svfit250GeV.Close()
###################################################################################

output_name = "ditau_mass_nn_final_39"
#output_name = "ditau_mass_distribution"
description_of_training = "standardized input data  INPUT: 1000 (classification of decaychannel) vis tau1 (pt,eta,phi,E,mass)+vis tau2 (pt,eta,phi,E,mass)+MET_ET+MET_Phi+collinear ditaumass"
output_file_name = "%s.txt" % (output_name)
output_root_name = "%s.root" % (output_name)
#rootfile = ROOT.TFile(output_root_name,"RECREATE")
#output_file = open(output_file_name,'w')
#sys.stdout = output_file

print "dataset length:",len(ditaumass)
print "train length:",len(train_ditaumass_selected)
print "test length:",len(test_ditaumass_selected)

fill_histditaumass_selected()
fill_histditauvismass_selected()
fill_histditaucollinearmass_selected()
fill_histditaumass_125GeV()
fill_histditaumass_180GeV()
fill_histditaumass_250GeV()
######################### run neural network  ######################

batch_size = 128
epochs = 200

start_nn = time.time()
neural_network(batch_size,epochs,output_name)
end_nn = time.time()

print "NN executon time:",(end_nn-start_nn)/3600,"h" 

fill_histprofres()
fill_histprofabsres()

#histogram of di-tau mass using regression all decays

canv2 = ROOT.TCanvas("di-tau mass using NN and SVfit")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.32,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.35,0.03)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum())
histditaumass.SetMaximum(max_bin*1.08)
histditaumass.Draw("HIST")
histditaumassnn.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
leg2 = ROOT.TLegend(0.13,0.62,0.35,0.87)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau_{NN} mass","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
leg2.SetTextSize(0.05)
leg2.Draw()
pad2.cd()
histditaumassnnratio.Draw("P")
histditaumasssvfitratio.Draw("P SAME")
unit_line = ROOT.TLine(0.0,1.0,350.0,1.0)
unit_line.SetLineColor(2)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
histditaumass.Write()
histditaumassnn.Write()
histditauvismass.Write()
histditaumasssvfit.Write()
canv2.Write()
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)


canv3 = ROOT.TCanvas("ditaumassnncorr")
histditaumassnncorr.Draw()
histditaumassnncorr.Write()
line = ROOT.TLine(0.0,0.0,400.0,400.0)
line.Draw("SAME")
line.SetLineWidth(2)
output_hist_corr_name = "%s_corr.png" %(output_name)
canv3.Write()
img3 = ROOT.TImage.Create()
img3.FromPad(canv3)
img3.WriteImage(output_hist_corr_name)

canv4 = ROOT.TCanvas("resolution")
histditaumassnnres.Draw()
histditaumassnnres.Write()
output_hist_res_name = "%s_res.png" %(output_name)
img4 = ROOT.TImage.Create()
img4.FromPad(canv4)
img4.WriteImage(output_hist_res_name)

canv5 = ROOT.TCanvas("ditaumass NN correlation with resolution")
histditaumassnncorrres.Draw()
histditaumassnncorrres.Write()
output_hist_corrres_name = "%s_corrres.png" %(output_name)
canv5.Write()
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_corrres_name)

canv6 = ROOT.TCanvas("ditaumass NN inputs")
histtrainditaumasscheck.Draw()
histtrainditaumasscheck.Write()
output_input_name = "%s_NNinput.png" %(output_name)
canv6.Write()
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage(output_input_name)

canv_use1 = ROOT.TCanvas("nn resolution use")
histditaumassnnrescomp.Draw()
ROOT.gPad.Update()
nn_statbox = histditaumassnnrescomp.FindObject("stats")
nn_color = histditaumassnnrescomp.GetLineColor()
nn_statbox.SetTextColor(1)
nn_statbox.SetLineColor(nn_color)
nn_statbox.SetOptStat(1101)
X1 = nn_statbox.GetX1NDC()
Y1 = nn_statbox.GetY1NDC()
X2 = nn_statbox.GetX2NDC()
Y2 = nn_statbox.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution use")
histditaumasssvfitres.Draw()
ROOT.gPad.Update()
svfit_statbox = histditaumasssvfitres.FindObject("stats")
svfit_color = histditaumasssvfitres.GetLineColor()
svfit_statbox.SetTextColor(1)
svfit_statbox.SetLineColor(svfit_color)
svfit_statbox.SetOptStat(1101)
svfit_statbox.SetX1NDC(X1)
svfit_statbox.SetX2NDC(X2)
svfit_statbox.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox.SetY2NDC(Y1)

canv7 = ROOT.TCanvas("resolution comparison")
max_bin = max(histditaumassnnrescomp.GetMaximum(),histditaumasssvfitres.GetMaximum())
histditaumassnnrescomp.SetMaximum(max_bin*1.08)
histditaumassnnrescomp.Draw()
histditaumasssvfitres.Draw("SAMES")
nn_statbox.Draw("SAME")
svfit_statbox.Draw("SAME")
output_res_compare_name = "%s_rescompar.png" %(output_name)
histditaumassnnrescomp.Write()
histditaumasssvfitres.Write()
canv7.Write()
img7 = ROOT.TImage.Create()
img7.FromPad(canv7)
img7.WriteImage(output_res_compare_name)

canv8 = ROOT.TCanvas("ditaumass profile resolution")
profditaumassnncorrres.Draw()
profditaumassnncorrres.Write()
output_profres_name = "%s_profres.png" %(output_name)
canv8.Write()
img8 = ROOT.TImage.Create()
img8.FromPad(canv8)
img8.WriteImage(output_profres_name)

canv9 = ROOT.TCanvas("ditaumass profile resolution comparison")
max_bin9 = max(histprofditaumassnncorrres.GetMaximum(),histprofditaumasssvfitcorrres.GetMaximum())
histprofditaumassnncorrres.SetMaximum(max_bin9*1.08)
histprofditaumassnncorrres.Draw()
histprofditaumasssvfitcorrres.Draw("SAME")
histprofditaumasssvfitcorrres.Write()
leg9 = ROOT.TLegend(0.13,0.77,0.4,0.87)
leg9.AddEntry(histprofditaumassnncorrres,"Neural Network","PL")
leg9.AddEntry(histprofditaumasssvfitcorrres,"SVfit","PL")
leg9.SetTextSize(0.04)
leg9.Draw()
output_profrescomp_name = "%s_profrescomp.png" %(output_name)
canv9.Write()
img9 = ROOT.TImage.Create()
img9.FromPad(canv9)
img9.WriteImage(output_profrescomp_name)

canv10 = ROOT.TCanvas("ditaumass profile abs(resolution)")
profditaumassnncorrabsres.Draw()
profditaumassnncorrabsres.Write()
output_profabsres_name = "%s_profabsres.png" %(output_name)
canv10.Write()
img10 = ROOT.TImage.Create()
img10.FromPad(canv10)
img10.WriteImage(output_profabsres_name)

canv11 = ROOT.TCanvas("ditaumass profile abs(resolution) comparisson")
max_bin11 = max(histprofditaumassnncorrabsres.GetMaximum(),histprofditaumasssvfitcorrabsres.GetMaximum())
histprofditaumassnncorrabsres.SetMaximum(max_bin11*1.08)
histprofditaumassnncorrabsres.Draw()
histprofditaumasssvfitcorrabsres.Draw("SAME")
histprofditaumasssvfitcorrabsres.Write()
leg11 = ROOT.TLegend(0.13,0.77,0.4,0.87)
leg11.AddEntry(histprofditaumassnncorrabsres,"Neural Network","PL")
leg11.AddEntry(histprofditaumasssvfitcorrabsres,"SVfit","PL")
leg11.SetTextSize(0.04)
leg11.Draw()
output_profabsrescomp_name = "%s_profabsrescomp.png" %(output_name)
canv11.Write()
img11 = ROOT.TImage.Create()
img11.FromPad(canv11)
img11.WriteImage(output_profabsrescomp_name)

canv12 = ROOT.TCanvas("ditaumass rms")
histditaumassnnrms.Draw("P")
histditaumassnnrms.Write()
output_rms_name = "%s_rms.png" %(output_name)
canv12.Write()
img12 = ROOT.TImage.Create()
img12.FromPad(canv12)
img12.WriteImage(output_rms_name)

canv13 = ROOT.TCanvas("di-tau mass using NN and SVfit including vismass")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.32,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.35,0.03)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum(),histditauvismass.GetMaximum())
histditaumass.SetMaximum(max_bin*1.3)
histditaumass.Draw("HIST")
histditauvismass.Draw("HIST SAME")
histditaumassnn.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
leg2 = ROOT.TLegend(0.67,0.6,0.97,0.87)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg2.AddEntry(histditauvismass,"visible di-#tau_{gen} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau_{NN} mass","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
leg2.SetTextSize(0.05)
leg2.Draw()
pad2.cd()
histditaumassnnratio.Draw("P")
histditaumasssvfitratio.Draw("P SAME")
unit_line = ROOT.TLine(0.0,1.0,350.0,1.0)
unit_line.SetLineColor(2)
unit_line.Draw("SAME")
output_allhist_name = "%s_all.png" %(output_name)
canv13.Write()
img13 = ROOT.TImage.Create()
img13.FromPad(canv13)
img13.WriteImage(output_allhist_name)

##### 125 GeV histograms
canv14 = ROOT.TCanvas("di-tau mass 125GeV using NN and SVfit")
max_bin125GeV = max(histditaumass125GeV.GetMaximum(),histditaumassnn125GeV.GetMaximum(),histditaumasssvfit125GeV.GetMaximum())
histditaumass125GeV.SetMaximum(max_bin125GeV*1.08)
histditaumass125GeV.Draw("HIST")
histditaumassnn125GeV.Draw("HIST SAME")
histditaumasssvfit125GeV.Draw("HIST SAME")
leg14 = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg14.AddEntry(histditaumass125GeV,"di-#tau_{gen} mass","PL")
leg14.AddEntry(histditaumassnn125GeV,"di-#tau_{NN} mass","PL")
leg14.AddEntry(histditaumasssvfit125GeV,"di-#tau_{SVfit} mass","PL")
leg14.SetTextSize(0.04)
leg14.Draw()
output_hist125GeV_name = "%s_125GeV.png" %(output_name)
histditaumass125GeV.Write()
histditaumassnn125GeV.Write()
histditaumasssvfit125GeV.Write()
canv14.Write()
img14 = ROOT.TImage.Create()
img14.FromPad(canv14)
img14.WriteImage(output_hist125GeV_name)

canv_use1 = ROOT.TCanvas("nn resolution 125GeV use")
histditaumassnnres125GeV.Draw()
ROOT.gPad.Update()
nn_statbox125GeV = histditaumassnnres125GeV.FindObject("stats")
nn_color = histditaumassnnres125GeV.GetLineColor()
nn_statbox125GeV.SetTextColor(1)
nn_statbox125GeV.SetLineColor(nn_color)
nn_statbox125GeV.SetOptStat(1101)
X1 = nn_statbox125GeV.GetX1NDC()
Y1 = nn_statbox125GeV.GetY1NDC()
X2 = nn_statbox125GeV.GetX2NDC()
Y2 = nn_statbox125GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 125GeV use")
histditaumasssvfitres125GeV.Draw()
ROOT.gPad.Update()
svfit_statbox125GeV = histditaumasssvfitres125GeV.FindObject("stats")
svfit_color = histditaumasssvfitres125GeV.GetLineColor()
svfit_statbox125GeV.SetTextColor(1)
svfit_statbox125GeV.SetLineColor(svfit_color)
svfit_statbox125GeV.SetOptStat(1101)
svfit_statbox125GeV.SetX1NDC(X1)
svfit_statbox125GeV.SetX2NDC(X2)
svfit_statbox125GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox125GeV.SetY2NDC(Y1)

canv15 = ROOT.TCanvas("resolution comparison 125GeV")
max_bin15 = max(histditaumassnnres125GeV.GetMaximum(),histditaumasssvfitres125GeV.GetMaximum())
histditaumassnnres125GeV.SetMaximum(max_bin15*1.08)
histditaumassnnres125GeV.Draw()
histditaumasssvfitres125GeV.Draw("SAMES")
nn_statbox125GeV.Draw("SAME")
svfit_statbox125GeV.Draw("SAME")
output_res125GeV_compare_name = "%s_125GeV_rescompar.png" %(output_name)
histditaumassnnres125GeV.Write()
histditaumasssvfitres125GeV.Write()
canv15.Write()
img15 = ROOT.TImage.Create()
img15.FromPad(canv15)
img15.WriteImage(output_res125GeV_compare_name)

##### 250 GeV histograms
canv16 = ROOT.TCanvas("di-tau mass 250GeV using NN and SVfit")
max_bin250GeV = max(histditaumass250GeV.GetMaximum(),histditaumassnn250GeV.GetMaximum(),histditaumasssvfit250GeV.GetMaximum())
histditaumass250GeV.SetMaximum(max_bin250GeV*1.08)
histditaumass250GeV.Draw("HIST")
histditaumassnn250GeV.Draw("HIST SAME")
histditaumasssvfit250GeV.Draw("HIST SAME")
leg16 = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg16.AddEntry(histditaumass250GeV,"di-#tau_{gen} mass","PL")
leg16.AddEntry(histditaumassnn250GeV,"di-#tau_{NN} mass","PL")
leg16.AddEntry(histditaumasssvfit250GeV,"di-#tau_{SVfit} mass","PL")
leg16.SetTextSize(0.04)
leg16.Draw()
output_hist250GeV_name = "%s_250GeV.png" %(output_name)
histditaumass250GeV.Write()
histditaumassnn250GeV.Write()
histditaumasssvfit250GeV.Write()
canv16.Write()
img16 = ROOT.TImage.Create()
img16.FromPad(canv16)
img16.WriteImage(output_hist250GeV_name)

canv_use1 = ROOT.TCanvas("nn resolution 250GeV use")
histditaumassnnres250GeV.Draw()
ROOT.gPad.Update()
nn_statbox250GeV = histditaumassnnres250GeV.FindObject("stats")
nn_color = histditaumassnnres250GeV.GetLineColor()
nn_statbox250GeV.SetTextColor(1)
nn_statbox250GeV.SetLineColor(nn_color)
nn_statbox250GeV.SetOptStat(1101)
X1 = nn_statbox250GeV.GetX1NDC()
Y1 = nn_statbox250GeV.GetY1NDC()
X2 = nn_statbox250GeV.GetX2NDC()
Y2 = nn_statbox250GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 250GeV use")
histditaumasssvfitres250GeV.Draw()
ROOT.gPad.Update()
svfit_statbox250GeV = histditaumasssvfitres250GeV.FindObject("stats")
svfit_color = histditaumasssvfitres250GeV.GetLineColor()
svfit_statbox250GeV.SetTextColor(1)
svfit_statbox250GeV.SetLineColor(svfit_color)
svfit_statbox250GeV.SetOptStat(1101)
svfit_statbox250GeV.SetX1NDC(X1)
svfit_statbox250GeV.SetX2NDC(X2)
svfit_statbox250GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox250GeV.SetY2NDC(Y1)

canv17 = ROOT.TCanvas("resolution comparison 250GeV")
max_bin17 = max(histditaumassnnres250GeV.GetMaximum(),histditaumasssvfitres250GeV.GetMaximum())
histditaumassnnres250GeV.SetMaximum(max_bin17*1.08)
histditaumassnnres250GeV.Draw()
histditaumasssvfitres250GeV.Draw("SAMES")
nn_statbox250GeV.Draw("SAME")
svfit_statbox250GeV.Draw("SAME")
output_res250GeV_compare_name = "%s_250GeV_rescompar.png" %(output_name)
histditaumassnnres250GeV.Write()
histditaumasssvfitres250GeV.Write()
canv17.Write()
img17 = ROOT.TImage.Create()
img17.FromPad(canv17)
img17.WriteImage(output_res250GeV_compare_name)

##### 180 GeV histograms
canv18 = ROOT.TCanvas("di-tau mass 180GeV using NN and SVfit")
max_bin180GeV = max(histditaumass180GeV.GetMaximum(),histditaumassnn180GeV.GetMaximum(),histditaumasssvfit180GeV.GetMaximum())
histditaumass180GeV.SetMaximum(max_bin180GeV*1.08)
histditaumass180GeV.Draw("HIST")
histditaumassnn180GeV.Draw("HIST SAME")
histditaumasssvfit180GeV.Draw("HIST SAME")
leg18 = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg18.AddEntry(histditaumass180GeV,"di-#tau_{gen} mass","PL")
leg18.AddEntry(histditaumassnn180GeV,"di-#tau_{NN} mass","PL")
leg18.AddEntry(histditaumasssvfit180GeV,"di-#tau_{SVfit} mass","PL")
leg18.SetTextSize(0.04)
leg18.Draw()
output_hist180GeV_name = "%s_180GeV.png" %(output_name)
histditaumass180GeV.Write()
histditaumassnn180GeV.Write()
histditaumasssvfit180GeV.Write()
canv18.Write()
img18 = ROOT.TImage.Create()
img18.FromPad(canv18)
img18.WriteImage(output_hist180GeV_name)

canv_use1 = ROOT.TCanvas("nn resolution 180GeV use")
histditaumassnnres180GeV.Draw()
ROOT.gPad.Update()
nn_statbox180GeV = histditaumassnnres180GeV.FindObject("stats")
nn_color = histditaumassnnres180GeV.GetLineColor()
nn_statbox180GeV.SetTextColor(1)
nn_statbox180GeV.SetLineColor(nn_color)
nn_statbox180GeV.SetOptStat(1101)
X1 = nn_statbox180GeV.GetX1NDC()
Y1 = nn_statbox180GeV.GetY1NDC()
X2 = nn_statbox180GeV.GetX2NDC()
Y2 = nn_statbox180GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 180GeV use")
histditaumasssvfitres180GeV.Draw()
ROOT.gPad.Update()
svfit_statbox180GeV = histditaumasssvfitres180GeV.FindObject("stats")
svfit_color = histditaumasssvfitres180GeV.GetLineColor()
svfit_statbox180GeV.SetTextColor(1)
svfit_statbox180GeV.SetLineColor(svfit_color)
svfit_statbox180GeV.SetOptStat(1101)
svfit_statbox180GeV.SetX1NDC(X1)
svfit_statbox180GeV.SetX2NDC(X2)
svfit_statbox180GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox180GeV.SetY2NDC(Y1)

canv19 = ROOT.TCanvas("resolution comparison 180GeV")
max_bin19 = max(histditaumassnnres180GeV.GetMaximum(),histditaumasssvfitres180GeV.GetMaximum())
histditaumassnnres180GeV.SetMaximum(max_bin19*1.08)
histditaumassnnres180GeV.Draw()
histditaumasssvfitres180GeV.Draw("SAMES")
nn_statbox180GeV.Draw("SAME")
svfit_statbox180GeV.Draw("SAME")
output_res180GeV_compare_name = "%s_180GeV_rescompar.png" %(output_name)
histditaumassnnres180GeV.Write()
histditaumasssvfitres180GeV.Write()
canv19.Write()
img19 = ROOT.TImage.Create()
img19.FromPad(canv19)
img19.WriteImage(output_res180GeV_compare_name)

##### gen mass and visible gen mass
canv20 = ROOT.TCanvas("gen mass and visible gen mass")
max_bin20 = max(histditaumassgen.GetMaximum(), histditauvismass.GetMaximum())
histditaumassgen.SetMaximum(max_bin20*1.08)
histditaumassgen.Draw("HIST")
histditauvismass.Draw("HIST SAME")
histditaucollmass.Draw("HIST SAME")
histditausvfit.Draw("HIST SAME")
leg20 = ROOT.TLegend(0.55,0.67,0.87,0.87)
leg20.AddEntry(histditaumassgen,"di-#tau_{gen} mass","PL")
leg20.AddEntry(histditauvismass,"visible di-#tau_{gen} mass","PL")
leg20.SetTextSize(0.04)
leg20.Draw()
output_gen_name = "%s_gen.png" %(output_name)
img20 = ROOT.TImage.Create()
img20.FromPad(canv20)
img20.WriteImage(output_gen_name)

"""
#gem mass and visible gen mass and collinear gen mass and SVfit
canv20 = ROOT.TCanvas("gen mass and visible gen mass")
max_bin20 = max(histditaumassgen.GetMaximum(), histditauvismass.GetMaximum(), histditaucollmass.GetMaximum(),histditaumasssvfit.GetMaximum())
histditaumassgen.SetMaximum(max_bin20*1.08)
histditaumassgen.Draw("HIST")
histditauvismass.Draw("HIST SAME")
histditaucollmass.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
leg20 = ROOT.TLegend(0.53,0.67,0.87,0.87)
leg20.AddEntry(histditaumassgen,"di-#tau_{gen} mass","PL")
leg20.AddEntry(histditauvismass,"visible di-#tau_{gen} mass","PL")
leg20.AddEntry(histditaucollmass,"collinear di-#tau_{gen} mass","PL")
leg20.AddEntry(histditaumasssvfit,"SVfit","PL")
leg20.SetTextSize(0.04)
leg20.Draw()
output_gen_name = "%s_gen.png" %(output_name)
img20 = ROOT.TImage.Create()
img20.FromPad(canv20)
img20.WriteImage(output_gen_name)
"""

canv21 = ROOT.TCanvas("ditaumasssvfitcorr")
histditaumasssvfitcorr.Draw()
histditaumasssvfitcorr.Write()
line = ROOT.TLine(0.0,0.0,400.0,400.0)
line.Draw("SAME")
line.SetLineWidth(2)
output_hist_svfit_corr_name = "%s_svfit_corr.png" %(output_name)
canv21.Write()
img21 = ROOT.TImage.Create()
img21.FromPad(canv21)
img21.WriteImage(output_hist_svfit_corr_name)

canv22 = ROOT.TCanvas("ditaumass SVfit correlation with resolution")
histditaumasssvfitcorrres.Draw()
histditaumasssvfitcorrres.Write()
output_hist_svfit_corrres_name = "%s_svfit_corrres.png" %(output_name)
canv22.Write()
img22 = ROOT.TImage.Create()
img22.FromPad(canv22)
img22.WriteImage(output_hist_svfit_corrres_name)

output_file.close()
rootfile.Close()

