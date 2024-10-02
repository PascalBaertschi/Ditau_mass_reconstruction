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
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
import multiprocessing


#################  choose size and decaymode of used dataset ######################

decaymode = 'all'
fixed_dataset_length = 800000
#fixed_dataset_length = 791166
#fixed_dataset_length = 6000000
fixed_train_length = 553700
fixed_test_length = 237300
fixed_test_length_select = 40000
#fixed_dataset_length = 10000
fixed_train_length_lowmass = 553700
fixed_test_length_highmass = 237300
#fixed_train_length_lowmass = 700
#fixed_test_length_highmass = 300

###################################################################################
list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfit_1e6.csv"
#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_8e5_%s.csv" %(decaymode)
#dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_5.5e6.csv",delim_whitespace=False,header=None)
#dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_withtaunu4Vec_2e6.csv",delim_whitespace=False,header=None)

#dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_recovistau_1e6.csv",delim_whitespace=False,header=None)
dataframe_ditaumass = pandas.read_csv(list_name,delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values
dataset_total_length = len(dataset_ditaumass[:,0])
train_nn_lowmass = []
test_nn_highmass = []
ditaumass_train_lowmass = []
ditauvismass_train_lowmass = []
ditaumass_test_highmass = []
ditauvismass_test_highmass = []


inputNN = []
inputSVfit = []
ditaumass = []
ditauvismass = []
decaymode_count = 0
lowmass_count = 0
highmass_count = 0
highmass_extra_count = 0
inputNN_highmass = []
ditaumass_high = []
ditauvismass_high = []
inputsample_weights = []
highmass_index_list = []
ditaumass_cut = []
ditauvismass_cut = []
test_inputNN_cut = []


for i in range(0,dataset_total_length):
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
    #vistau1_nu_pt = dataset_ditaumass[i,7]
    #vistau1_nu_eta = dataset_ditaumass[i,8]
    #vistau1_nu_phi = dataset_ditaumass[i,9]
    #vistau1_nu_mass = dataset_ditaumass[i,10]
    #vistau2_pt = dataset_ditaumass[i,11]
    #vistau2_eta = dataset_ditaumass[i,12]
    #vistau2_phi = dataset_ditaumass[i,13]
    #vistau2_mass = dataset_ditaumass[i,14]
    #vistau2_att = dataset_ditaumass[i,15]
    #vistau2_prongs = dataset_ditaumass[i,16]
    #vistau2_pi0 = dataset_ditaumass[i,17]
    #vistau2_nu_pt = dataset_ditaumass[i,18]
    #vistau2_nu_eta = dataset_ditaumass[i,19]
    #vistau2_nu_phi = dataset_ditaumass[i,20]
    #vistau2_nu_mass = dataset_ditaumass[i,21]
    #nu_pt = dataset_ditaumass[i,22]
    #nu_eta = dataset_ditaumass[i,23]
    #nu_phi = dataset_ditaumass[i,24]
    #nu_mass = dataset_ditaumass[i,25]
    #genMissingET_MET = dataset_ditaumass[i,26]
    #genMissingET_Phi = dataset_ditaumass[i,27]
    #MissingET_MET = dataset_ditaumass[i,28]
    #MissingET_Phi = dataset_ditaumass[i,29]
    #genMissingET_MET = dataset_ditaumass[i,18]
    #genMissingET_Phi = dataset_ditaumass[i,19]
    #MissingET_MET = dataset_ditaumass[i,20]
    #MissingET_Phi = dataset_ditaumass[i,21]
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
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    if (decaymode_count < fixed_dataset_length):
        decaymode_count += 1
        inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
        ditaumass.append(ditaumass_value)
        ditauvismass.append(ditauvismass_value)
    """
    if (decaymode_count < fixed_dataset_length) and vistau1_pt!=0 and vistau2_pt!=0:
        ditaumass_collinear = vismass/numpy.sqrt(vistau1_pt/(vistau1_pt+pt_nu)*vistau2_pt/(vistau2_pt+pt_nu))
        #if vistau1_att in (1,2) and vistau2_att in (1,2):
        #    inputNN.append([1.0,0.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
        #    #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass,1,0,0,0])
        #elif vistau1_att in (1,2) and vistau2_att == 3:
        #    inputNN.append([0.0,1.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
        #    #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass,0,1,0,0])
        #elif vistau1_att == 3 and vistau2_att in (1,2):
        #    inputNN.append([0.0,0.0,1.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
        #    #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass,0,0,1,0])
        #elif vistau1_att == 3 and vistau2_att == 3:
        #    inputNN.append([0.0,0.0,0.0,1.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
        #    #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass,0,0,0,1])
        inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
        ditaumass.append(ditaumass_value)
        ditauvismass.append(ditauvismass_value)
        decaymode_count += 1
    """    
    """
        if v_mot.M() > 100.0:
            highmass_count +=1
            highmass_index_list.append(1.0)
        else:
            highmass_index_list.append(0.0)
    elif decaymode_count == fixed_dataset_length:
        if highmass_extra_count < highmass_count:
            if v_mot.M() > 100.0:
                inputNN_highmass.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
                ditaumass_high.append(ditaumass_value)
                ditauvismass_high.append(ditauvismass_value)
                highmass_extra_count += 1
        else:
            break
    
    if v_mot.M() < 50:
        if lowmass_count < fixed_train_length_lowmass:
            train_nn_lowmass.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass_train_lowmass.append(ditaumass_value)
            ditauvismass_train_lowmass.append(ditauvismass_value)
            lowmass_count +=1
    if v_mot.M() > 50:
        if highmass_count < fixed_test_length_highmass:
            test_nn_highmass.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass_test_highmass.append(ditaumass_value)
            ditauvismass_test_highmass.append(ditauvismass_value)
            highmass_count +=1
    #if lowmass_count == fixed_train_length_lowmass and highmass_count == fixed_test_length_highmass:
    #    break
    """
"""    
for i in range(0,dataset_total_length):
    vistau1_pt = dataset_ditaumass[i,0]
    vistau1_eta = dataset_ditaumass[i,1]
    vistau1_phi = dataset_ditaumass[i,2]
    vistau1_mass = dataset_ditaumass[i,3]
    vistau1_att = dataset_ditaumass[i,4]
    vistau1_prongs = dataset_ditaumass[i,5]
    vistau1_pi0 = dataset_ditaumass[i,6]
    vistau1_nu_pt = dataset_ditaumass[i,7]
    vistau1_nu_eta = dataset_ditaumass[i,8]
    vistau1_nu_phi = dataset_ditaumass[i,9]
    vistau1_nu_mass = dataset_ditaumass[i,10]
    vistau2_pt = dataset_ditaumass[i,11]
    vistau2_eta = dataset_ditaumass[i,12]
    vistau2_phi = dataset_ditaumass[i,13]
    vistau2_mass = dataset_ditaumass[i,14]
    vistau2_att = dataset_ditaumass[i,15]
    vistau2_prongs = dataset_ditaumass[i,16]
    vistau2_pi0 = dataset_ditaumass[i,17]
    vistau2_nu_pt = dataset_ditaumass[i,18]
    vistau2_nu_eta = dataset_ditaumass[i,19]
    vistau2_nu_phi = dataset_ditaumass[i,20]
    vistau2_nu_mass = dataset_ditaumass[i,21]
    nu_pt = dataset_ditaumass[i,22]
    nu_eta = dataset_ditaumass[i,23]
    nu_phi = dataset_ditaumass[i,24]
    nu_mass = dataset_ditaumass[i,25]
    genMissingET_MET = dataset_ditaumass[i,26]
    genMissingET_Phi = dataset_ditaumass[i,27]
    MissingET_MET = dataset_ditaumass[i,28]
    MissingET_Phi = dataset_ditaumass[i,29]
    v_vistau1 = ROOT.TLorentzVector()
    v_vistau1.SetPtEtaPhiM(vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass)
    v_vistau2 = ROOT.TLorentzVector()
    v_vistau2.SetPtEtaPhiM(vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass)
    v_vistau1_nu = ROOT.TLorentzVector()
    v_vistau1_nu.SetPtEtaPhiM(vistau1_nu_pt,vistau1_nu_eta,vistau1_nu_phi,vistau1_nu_mass)
    v_vistau2_nu = ROOT.TLorentzVector()
    v_vistau2_nu.SetPtEtaPhiM(vistau2_nu_pt,vistau2_nu_eta,vistau2_nu_phi,vistau2_nu_mass)
    v_tau1 = v_vistau1 + v_vistau1_nu
    v_tau2 = v_vistau2 + v_vistau2_nu
    tau1_pt = v_tau1.Pt()
    tau1_eta = v_tau1.Eta()
    tau1_phi = v_tau1.Phi()
    tau1_mass = v_tau1.M()
    tau2_pt = v_tau2.Pt()
    tau2_eta = v_tau2.Eta()
    tau2_phi = v_tau2.Phi()
    tau2_mass = v_tau2.M()
    v_nu = ROOT.TLorentzVector()
    v_nu.SetPtEtaPhiM(nu_pt,nu_eta,nu_phi,nu_mass)
    v_mot = v_vistau1+v_vistau2+v_nu
    v_vismot = v_vistau1+v_vistau2
    ditaumass_value = v_mot.M()
    ditauvismass_value = v_vismot.M()
    ditaupt_value = v_mot.Pt()
    vismass = v_vismot.M()
    p_vis = v_vismot.P()
    pt_vis = v_vismot.Pt()
    pt_nu = v_nu.Pt()
    pt = v_mot.Pt()
    nu_px = v_nu.Px()
    nu_py = v_nu.Py()
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    if tau1_pt > 20 and tau2_pt > 20 and tau1_eta < 2.3 and tau2_eta < 2.3 and MissingET_MET > 20:
        if vistau1_att == 3 and vistau2_att == 3:
            if vistau1_pt > 45 and vistau2_pt > 45 and abs(vistau1_eta) < 2.1 and abs(vistau2_eta) < 2.1: 
                ditaumass_cut.append(ditaumass_value)
                ditauvismass_cut.append(ditauvismass_value)
                #decaymode_count += 1
                test_inputNN_cut.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,nu_px,nu_py,p_vis,vismass])
            else:
                continue
        elif vistau1_att in (1,2) and vistau2_att in (1,2):
            if vistau1_pt > vistau2_pt:
                if vistau1_pt > 20 and vistau2_pt > 10 and abs(vistau1_eta)<2.4 and abs(vistau2_eta) < 2.4:
                    ditaumass_cut.append(ditaumass_value)
                    ditauvismass_cut.append(ditauvismass_value)
                    #decaymode_count += 1
                    test_inputNN_cut.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,nu_px,nu_py,p_vis,vismass])
                else:
                    continue
            elif vistau2_pt > vistau1_pt:
                if vistau1_pt > 10 and vistau2_pt > 20 and abs(vistau1_eta)<2.4 and abs(vistau2_eta) < 2.4:
                    ditaumass_cut.append(ditaumass_value)
                    ditauvismass_cut.append(ditauvismass_value)
                    #decaymode_count += 1
                    test_inputNN_cut.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,nu_px,nu_py,p_vis,vismass])
                else:
                    continue
        elif vistau1_att in (1,2) and vistau2_att == 3:
            if vistau1_pt > 20 and abs(vistau1_eta) < 2.1 and vistau2_pt > 30 and abs(vistau2_eta) < 2.3:
                ditaumass_cut.append(ditaumass_value)
                ditauvismass_cut.append(ditauvismass_value)
                #decaymode_count += 1
                test_inputNN_cut.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,nu_px,nu_py,p_vis,vismass])
            else:
                continue
        elif vistau2_att in (1,2) and vistau1_att == 3:
            if vistau2_pt > 20 and abs(vistau2_eta) < 2.1 and vistau1_pt > 30 and abs(vistau1_eta) < 2.3:
                ditaumass_cut.append(ditaumass_value)
                ditauvismass_cut.append(ditauvismass_value)
                #decaymode_count += 1
                test_inputNN_cut.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,nu_px,nu_py,p_vis,vismass])
            else:
                continue
    else:
        continue
"""

inputNN = numpy.array(inputNN,numpy.float64)
#test_inputNN_cut = numpy.array(test_inputNN_cut,numpy.float64)
#inputNN_highmass = numpy.array(inputNN_highmass,numpy.float64)
#train_nn_lowmass = numpy.array(train_nn_lowmass,numpy.float64)
#test_nn_highmass = numpy.array(test_nn_highmass,numpy.float64)
#train_nn_lowmass[:,:] = preprocessing.scale(train_nn_lowmass[:,:])
#test_nn_highmass[:,:] = preprocessing.scale(test_nn_highmass[:,:])

#extra_dataset_length = len(inputNN_highmass[:,0])
#train_extra_length = int(round(extra_dataset_length*0.7))
#test_extra_length = int(round(extra_dataset_length*0.3))
dataset_length = len(inputNN[:,0])
train_length = int(round(dataset_length*0.7))
test_length = int(round(dataset_length*0.3))

train_inputNN = inputNN[0:train_length,:]
#train_inputNN = inputNN[0:fixed_train_length,:]
test_inputNN = inputNN[train_length:dataset_length,:]
train_inputNN_forselection = inputNN[fixed_test_length:,:]
test_inputNN_selected = inputNN[0:fixed_test_length,:]
#test_inputNN_selected = inputNN[0:fixed_test_length_select,:]
train_ditaumass_forselection = ditaumass[fixed_test_length:]
test_ditaumass_selected = ditaumass[0:fixed_test_length]
#test_ditaumass_selected = ditaumass[0:fixed_test_length_select]
train_ditaumass = ditaumass[0:train_length]
#train_ditaumass = ditaumass[0:fixed_train_length]
test_ditaumass = ditaumass[train_length:dataset_length]
train_ditauvismass = ditauvismass[0:train_length]
test_ditauvismass = ditauvismass[train_length:dataset_length]
#indexlist_train_highmass = highmass_index_list[0:train_length]
#indexlist_test_highmass = highmass_index_list[train_length:dataset_length]

######### make flat mass distribution with less events ######################
#histtrainditaumass = ROOT.TH1D("trainditaumass","trainditaumass",90,10,100)
histtrainditaumass = ROOT.TH1D("trainditaumass","trainditaumass",100,10,110)

min_bincontent = 10000
for i in train_ditaumass_forselection:
    histtrainditaumass.Fill(i)

#for j in range(90):
for j in range(100):
    bincontent = histtrainditaumass.GetBinContent(j+1)
    if bincontent < min_bincontent:
        min_bincontent = bincontent

#histtrainditaumasscalc = ROOT.TH1D("trainditaumasscalc","trainditaumasscalc",90,10,100)
histtrainditaumasscalc = ROOT.TH1D("trainditaumasscalc","trainditaumasscalc",100,10,110)
print "min bincontent:",min_bincontent
train_inputNN_selected = []
train_ditaumass_selected = []
for k,ditaumass_loopvalue in enumerate(train_ditaumass_forselection):
    bin_index = histtrainditaumasscalc.GetXaxis().FindBin(ditaumass_loopvalue)
    bin_content = histtrainditaumasscalc.GetBinContent(bin_index)
    if bin_content < min_bincontent:
        histtrainditaumasscalc.SetBinContent(bin_index,bin_content+1)
        train_inputNN_selected.append(train_inputNN_forselection[k,:])
        train_ditaumass_selected.append(ditaumass_loopvalue)
    else:
        continue

train_inputNN_selected = numpy.array(train_inputNN_selected,numpy.float64)

#histtrainditaumasscheck = ROOT.TH1D("trainditaumasscheck","trainditaumasscheck",90,10,100)
histtrainditaumasscheck = ROOT.TH1D("trainditaumasscheck","trainditaumasscheck",100,10,110)
for j in train_ditaumass_selected:
    histtrainditaumasscheck.Fill(j)

"""
#####################   weights for more higher masses ########################
train_inputNN_highmass = inputNN_highmass[0:train_extra_length,:]
test_inputNN_highmass = inputNN_highmass[train_extra_length:extra_dataset_length,:]
train_ditaumass_high = ditaumass_high[0:train_extra_length]
test_ditaumass_high = ditaumass_high[train_extra_length:extra_dataset_length]
train_ditauvismass_high = ditauvismass_high[0:train_extra_length]
test_ditauvismass_high = ditauvismass_high[train_extra_length:extra_dataset_length]

train_length_combined = train_length+train_extra_length
test_length_combined = test_length + test_extra_length
index_train_combined_step = int(train_length_combined/train_extra_length)
index_test_combined_step = int(test_length_combined/test_extra_length)
#print "train dataset length:",train_length
#print "train extra length:",train_extra_length
#print "train length combined:",train_length_combined
#print "index train combined step:",index_train_combined_step
#print "test dataset length:",test_length
#print "test extra length:",test_extra_length
#print "test length combined:",test_length_combined
#print "index test combined step:",index_test_combined_step

train_inputNN_combined = []
test_inputNN_combined = []
train_ditaumass_combined = []
test_ditaumass_combined = []
train_ditauvismass_combined = []
test_ditauvismass_combined = []
weightlist_combined_train = []
weightlist_combined_test = []
index_train_combined = 0
index_test_combined = 0
index_train_dataset = 0
index_test_dataset = 0
index_train_highmass = 0
index_test_highmass = 0 

for j in range(train_length_combined):
    if j == index_train_combined and index_train_highmass < train_extra_length:
        train_inputNN_combined.append(train_inputNN_highmass[index_train_highmass,:])
        train_ditaumass_combined.append(train_ditaumass_high[index_train_highmass])
        train_ditauvismass_combined.append(train_ditauvismass_high[index_train_highmass])
        index_train_highmass += 1
        index_train_combined += index_train_combined_step
        weightlist_combined_train.append(0.5)
    else:
        train_inputNN_combined.append(train_inputNN[index_train_dataset,:])
        train_ditaumass_combined.append(train_ditaumass[index_train_dataset])
        train_ditauvismass_combined.append(train_ditauvismass[index_train_dataset])
        if indexlist_train_highmass[index_train_dataset] == 1.0:
            weightlist_combined_train.append(0.5)
        else:
            weightlist_combined_train.append(1.0)
        index_train_dataset += 1

for j in range(test_length_combined):
    if j == index_test_combined and index_test_highmass < test_extra_length:
        test_inputNN_combined.append(test_inputNN_highmass[index_test_highmass,:])
        test_ditaumass_combined.append(test_ditaumass_high[index_test_highmass])
        test_ditauvismass_combined.append(test_ditauvismass_high[index_test_highmass])
        index_test_highmass += 1
        index_test_combined += index_test_combined_step
        weightlist_combined_test.append(0.5)
    else:
        test_inputNN_combined.append(test_inputNN[index_test_dataset,:])
        test_ditaumass_combined.append(test_ditaumass[index_test_dataset])
        test_ditauvismass_combined.append(test_ditauvismass[index_test_dataset])
        if indexlist_test_highmass[index_test_dataset] == 1.0:
            weightlist_combined_test.append(0.5)
        else:
            weightlist_combined_test.append(1.0)
        index_test_dataset += 1


train_inputNN_combined = numpy.array(train_inputNN_combined,numpy.float64)
test_inputNN_combined = numpy.array(test_inputNN_combined,numpy.float64)
weightlist_combined_train = numpy.array(weightlist_combined_train,numpy.float64)
weightlist_combined_test = numpy.array(weightlist_combined_test,numpy.float64)
overfit_inputNN_combined = train_inputNN_combined[0:len(test_ditaumass_combined),:]

train_inputNN_combined[:,:] = preprocessing.scale(train_inputNN_combined[:,:])
test_inputNN_combined[:,:] = preprocessing.scale(test_inputNN_combined[:,:])
overfit_inputNN_combined[:,:] = preprocessing.scale(overfit_inputNN_combined[:,:])

##################################################################################################
"""
### Standardize the whole input data  ####
#inputNN[:,:] = preprocessing.scale(inputNN[:,:])
##########################################

train_inputNN = inputNN[0:train_length,:]
train_ditaumass = ditaumass[0:train_length]
train_ditauvismass = ditauvismass[0:train_length]
#train_inputNN = inputNN[0:fixed_train_length,:]
#train_ditaumass = ditaumass[0:fixed_train_length]
#train_ditauvismass = ditauvismass[0:fixed_train_length]
test_inputNN = inputNN[train_length:dataset_length,:]
test_ditaumass = ditaumass[train_length:dataset_length]
test_ditauvismass = ditauvismass[train_length:dataset_length]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
overfit_ditaumass = ditaumass[0:len(test_ditaumass)]
overfit_ditauvismass = ditauvismass[0:len(test_ditauvismass)]

#### Standardize according to the training data ####
test_inputNN_stand = test_inputNN
for j in range(len(train_inputNN[0,:])):
    mean = numpy.mean(train_inputNN[:,j])
    std = numpy.std(train_inputNN[:,j])
    for i in range(0,len(test_ditaumass)):
        value = test_inputNN[i,j]
        new_value = (value-mean)/std
        test_inputNN_stand[i,j] = new_value

test_inputNN = numpy.array(test_inputNN_stand,numpy.float64) 
train_inputNN[:,:] = preprocessing.scale(train_inputNN[:,:])
#######################################################

"""
########## weights for making a flat mass ##################
histditaumass_weights_train = ROOT.TH1D("ditaumass_weights_train","di-#tau mass for training",500,0,500)
for k in train_ditaumass:
    #for k in train_ditauvismass:
    histditaumass_weights_train.Fill(k)
binweightlist_train = []
weightlist_train = []
for i in range(1,501):
    bin_content = histditaumass_weights_train.GetBinContent(i)
    if bin_content != 0:
        weight = 1./bin_content
    else:
        weight = 1.
    binweightlist_train.append(weight)
for j,mass_per_event in enumerate(train_ditaumass):
    #for j,mass_per_event in enumerate(train_ditauvismass):
    if mass_per_event < 500.0:
        bin_index = histditaumass_weights_train.GetXaxis().FindBin(mass_per_event)
        weightlist_train.append(binweightlist_train[bin_index])
    else:
        weightlist_train.append(1.0)

histditaumass_weights_test = ROOT.TH1D("ditaumass_weights_test","di-#tau mass for testing",500,0,500)
for k in test_ditaumass:
    #for k in test_ditauvismass:
    histditaumass_weights_test.Fill(k)
binweightlist_test = []
weightlist_test = []
for i in range(1,501):
    bin_content = histditaumass_weights_test.GetBinContent(i)
    if bin_content != 0:
        weight = 1./bin_content
    else:
        weight = 1.
    binweightlist_test.append(weight)
for j,mass_per_event in enumerate(test_ditaumass):
    #for j,mass_per_event in enumerate(test_ditauvismass):
    if mass_per_event < 500.0:
        bin_index = histditaumass_weights_test.GetXaxis().FindBin(mass_per_event)
        weightlist_test.append(binweightlist_test[bin_index])
    else:
        weightlist_test.append(1.0)

histditaumass_weights_overfit = ROOT.TH1D("ditaumass_weights_overfit","di-#tau mass for overfit_test",500,0,500)
for k in overfit_ditaumass:
    #for k in overfit_ditauvismass:
    histditaumass_weights_overfit.Fill(k)
binweightlist_overfit = []
weightlist_overfit = []
for i in range(1,501):
    bin_content = histditaumass_weights_overfit.GetBinContent(i)
    if bin_content != 0:
        weight = 1./bin_content
    else:
        weight = 1.
    binweightlist_overfit.append(weight)
for j,mass_per_event in enumerate(overfit_ditaumass):
    #for j,mass_per_event in enumerate(overfit_ditauvismass):
    if mass_per_event < 500.0:
        bin_index = histditaumass_weights_overfit.GetXaxis().FindBin(mass_per_event)
        weightlist_overfit.append(binweightlist_overfit[bin_index])
    else:
        weightlist_overfit.append(1.0)

weightlist = weightlist_train+weightlist_test
weightlist = numpy.array(weightlist,numpy.float64)
weightlist_train = numpy.array(weightlist_train,numpy.float64)
weightlist_test = numpy.array(weightlist_test,numpy.float64)
weightlist_overfit = numpy.array(weightlist_overfit,numpy.float64)
#########################################################################
"""
#train_inputNN[:,:] = preprocessing.scale(train_inputNN[:,:])
#test_inputNN[:,:] = preprocessing.scale(test_inputNN[:,:])
#overfit_inputNN[:,:] = preprocessing.scale(overfit_inputNN[:,:])
#test_inputNN_cut[:,:] = preprocessing.scale(test_inputNN_cut[:,:])
#train_inputNN_selected[:,:] = preprocessing.scale(train_inputNN_selected[:,:])
#test_inputNN_selected[:,:] = preprocessing.scale(test_inputNN_selected[:,:])

#Defining how many Digits are shown for each axis labels
ROOT.TGaxis.SetMaxDigits(4)
#histogram of ditau mass using neural network and SVfit
histtitle = "reconstruct di-#tau mass using a neural network"
histditaumass = ROOT.TH1D("ditaumass",histtitle,100,0,100)
#histditaumass = ROOT.TH1D("ditaumass",histtitle,150,0,100)
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleSize(0.049)
histditaumass.GetYaxis().SetTitleOffset(0.75)
histditaumass.GetYaxis().SetLabelSize(0.049)
histditaumass.SetLineColor(2)
histditaumass.SetLineWidth(3)
histditaumass.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","di-#tau_{gen} mass and visible mass",100,0,100)
histditauvismass.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditauvismass.GetYaxis().SetTitle("number of occurence")
histditauvismass.SetLineColor(6)
histditauvismass.SetLineWidth(3)
histditauvismass.SetLineStyle(7)
histditauvismass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassnn","reconstructed di-#tau mass using neural network",100,0,100)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetLineWidth(3)
histditaumassnn.SetLineStyle(7)
histditaumassnn.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",100,0,100)
histditaumasssvfit.SetLineColor(3)
histditaumasssvfit.SetStats(0)
histditaumassnnoverfit = ROOT.TH1D("ditaumassnnoverfit","overfit of neural network",100,0,100)
histditaumassnnoverfit.SetLineColor(3)
histditaumassnnoverfit.SetLineStyle(2)
histditaumassnnoverfit.SetStats(0)

histditaumassnncorr = ROOT.TH2D("ditaumassnncorr","di-#tau_{gen} mass vs di-#tau mass",100,0,100,100,0,100)
histditaumassnncorr.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnncorr.GetYaxis().SetTitle("di-#tau mass [GeV]")
histditaumassnncorr.SetStats(0)

histditaumassnnres040 = ROOT.TH1D("resolution0to40","resolution of di-#tau mass using neural network range 0-40 GeV",100,-3,3)
histditaumassnnres040.GetXaxis().SetTitle("resolution")
histditaumassnnres040.GetYaxis().SetTitle("number of occurence")
histditaumassnnres4080 = ROOT.TH1D("resolution40to80","resolution of di-#tau mass using neural network range 40-80 GeV",100,-3,3)
histditaumassnnres4080.GetXaxis().SetTitle("resolution")
histditaumassnnres4080.GetYaxis().SetTitle("number of occurence")
histditaumassnnres80100 = ROOT.TH1D("resolution80to100","resolution of di-#tau mass using neural network range 80-100 GeV",100,-3,3)
histditaumassnnres80100.GetXaxis().SetTitle("resolution")
histditaumassnnres80100.GetYaxis().SetTitle("number of occurecne")
histditaumassnnresabove100 = ROOT.TH1D("resolutionabove100","resolution of di-#tau mass using neural network range above 100 GeV",100,-3,3)
histditaumassnnresabove100.GetXaxis().SetTitle("resolution")
histditaumassnnresabove100.GetYaxis().SetTitle("number of occurence")
histditaumassnnres = ROOT.TH1D("resolution","resolution of di-#tau mass",100,-3,3)
histditaumassnnres.GetXaxis().SetTitle("resolution")
histditaumassnnres.GetYaxis().SetTitle("number of occurence")


#ratio histograms
histditaumassnnratio = ROOT.TH1D("ditaumassregallratio","ratio between di-#tau mass and di-#tau_{gen} mass",100,0,100)
#histditaumassnnratio = ROOT.TH1D("ditaumassregallratio","ratio between reconstruced and actual mass",90,10,100)
histditaumassnnratio.SetTitle("")
histditaumassnnratio.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
#histditaumassnnratio.GetXaxis().SetLabelSize(0.08)
#histditaumassnnratio.GetXaxis().SetTitleSize(0.08)
histditaumassnnratio.GetXaxis().SetLabelSize(0.115)
histditaumassnnratio.GetXaxis().SetTitleSize(0.115)
histditaumassnnratio.GetYaxis().SetTitle("ratio")
#histditaumassnnratio.GetYaxis().SetLabelSize(0.08)
#histditaumassnnratio.GetYaxis().SetTitleSize(0.08)
histditaumassnnratio.GetYaxis().SetLabelSize(0.115)
histditaumassnnratio.GetYaxis().SetTitleSize(0.115)
histditaumassnnratio.GetYaxis().SetTitleOffset(0.3)
histditaumassnnratio.GetYaxis().SetNdivisions(404)
histditaumassnnratio.GetYaxis().CenterTitle()
histditaumassnnratio.GetYaxis().SetRangeUser(0.0,2.0)
histditaumassnnratio.SetMarkerStyle(7)
histditaumassnnratio.SetMarkerColor(4)
histditaumassnnratio.SetStats(0)
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and di-#tau_{gen} mass",100,0,100)
#histditaumasssvfitratio.SetTitle("")
#histditaumasssvfitratio.GetXaxis().SetTitle("di-#tau mass [GeV]")
#histditaumasssvfitratio.GetXaxis().SetLabelSize(0.08)
#histditaumasssvfitratio.GetXaxis().SetTitleSize(0.08)
#histditaumasssvfitratio.GetYaxis().SetTitle("ratio")
#histditaumasssvfitratio.GetYaxis().SetLabelSize(0.08)
#histditaumasssvfitratio.GetYaxis().SetTitleSize(0.08)
#histditaumasssvfitratio.GetYaxis().SetTitleOffset(0.5)
#histditaumasssvfitratio.GetYaxis().SetNdivisions(504)
#histditaumasssvfitratio.GetYaxis().CenterTitle()
#histditaumasssvfitratio.GetYaxis().SetRangeUser(0.0,5.0)
histditaumasssvfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(3)
histditaumasssvfitratio.SetStats(0)
histditaumassnnoverfitratio = ROOT.TH1D("ditaumassregalloverfitratio","ratio between reconstruced and overfit mass",100,0,100)
histditaumassnnoverfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(4)
histditaumassnnoverfitratio.SetStats(0)



def neural_network(batch_size,epochs,output_name):
    print "NEURAL NETWORK"
    mass_model = Sequential()
    #mass_model.add(Dropout(0.3,input_shape=(12,)))
    #mass_model.add(Dense(60,input_dim=17,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,input_dim=12,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(0.1))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(0.1))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(0.1))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(0.1))
    mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    
    def step_decay(epoch):
        initial_lrate = 0.005
        drop = 0.7
        epochs_drop = 20.0
        lrate = initial_lrate*math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate
    #lrate = [LearningRateScheduler(step_decay)]
    #adam = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = decay_rate)
    #adam = Adam(lr = 0.0, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = 0.0)
    mass_model.compile(loss='mean_squared_error',optimizer='adam')
    #history = mass_model.fit(inputNN,ditaumass,batch_size,epochs,shuffle=False,validation_split = 0.3,verbose = 2)
    #history = mass_model.fit(train_inputNN,train_ditaumass,batch_size,epochs,shuffle=False,validation_data = (test_inputNN_cut,ditaumass_cut),verbose =2)
    #history = mass_model.fit(inputNN,ditaumass,batch_size,epochs,validation_split = 0.3,verbose = 2)
    #mass_score = mass_model.evaluate(test_inputNN_cut,ditaumass_cut,batch_size,verbose=0)
    #history = mass_model.fit(train_inputNN_combined,train_ditaumass_combined,batch_size,epochs,sample_weight = weightlist_combined_train,validation_data =(test_inputNN_combined,test_ditaumass_combined),verbose = 2)
    #history = mass_model.fit(train_inputNN,train_ditaumass,batch_size,epochs,sample_weight = weightlist_train,validation_data = (test_inputNN,test_ditaumass),verbose = 2)
    #history = mass_model.fit(train_inputNN,train_ditauvismass,batch_size,epochs, sample_weight = weightlist_train,validation_data = (test_inputNN,test_ditauvismass),verbose = 2)
    history = mass_model.fit(train_inputNN,train_ditaumass,batch_size,epochs,validation_data = (test_inputNN,test_ditaumass),verbose = 2)
    #history = mass_model.fit(train_inputNN_selected,train_ditaumass_selected,batch_size,epochs,validation_data = (test_inputNN_selected,test_ditaumass_selected),verbose = 2)
    #mass_score = mass_model.evaluate(test_inputNN,test_ditaumass,batch_size,sample_weight = weightlist_test,verbose=0)
    #mass_score = mass_model.evaluate(test_inputNN_selected,test_ditaumass_selected,batch_size,verbose=0)
    mass_score = mass_model.evaluate(test_inputNN,test_ditaumass,batch_size,verbose=0)
    #mass_score = mass_model.evaluate(test_inputNN_combined,test_ditaumass_combined,batch_size,sample_weight = weightlist_combined_test,verbose=0)
    #ditaumass_nn = mass_model.predict(test_inputNN_cut,batch_size,verbose=0)
    #ditaumass_nn = mass_model.predict(test_inputNN_selected,batch_size,verbose=0)
    ditaumass_nn = mass_model.predict(test_inputNN,batch_size,verbose=0)
    #ditaumass_nn_overfit = mass_model.predict(overfit_inputNN,batch_size,verbose=0)
    #ditaumass_nn = mass_model.predict(test_inputNN_combined,batch_size,verbose=0)
    #ditaumass_nn_overfit = mass_model.predict(overfit_inputNN_combined,batch_size,verbose=0)
    #history = mass_model.fit(train_nn_lowmass,ditaumass_train_lowmass,batch_size,epochs,validation_data = (test_nn_highmass,ditaumass_test_highmass),verbose = 2)
    #mass_score = mass_model.evaluate(test_nn_highmass,ditaumass_test_highmass,batch_size,verbose=0)
    #ditaumass_nn = mass_model.predict(test_nn_highmass,batch_size,verbose=0)
    mass_model.summary()
    print "mass_model(",batch_size,epochs,")"
    print "loss (MSE):",mass_score
    print description_of_training
    failed_division = 0
    #preparing the histograms
    for j in ditaumass_nn:
        histditaumassnn.Fill(j)
    #for d in ditaumass_nn_overfit:
    #    histditaumassnnoverfit.Fill(d)
    for i,ditaumass_value in enumerate(test_ditaumass):
        res = (ditaumass_value - ditaumass_nn[i])/ditaumass_value
        histditaumassnnres.Fill(res)
        if ditaumass_value < 40:
            histditaumassnnres040.Fill(res)
        elif ditaumass_value > 40 and ditaumass_value < 80:
            histditaumassnnres4080.Fill(res)
        elif ditaumass_value > 80 and ditaumass_value < 100:
            histditaumassnnres80100.Fill(res)
        elif ditaumass_value > 100:
            histditaumassnnresabove100.Fill(res)
    for g in range(len(ditaumass_nn)):
        histditaumassnncorr.Fill(test_ditaumass[g],ditaumass_nn[g])
        #histditaumassnn.Fill(ditaumass_nn[g],weightlist_combined_test[g])
        #histditaumassnnoverfit.Fill(ditaumass_nn_overfit[g],weightlist_combined_test[g])
        #histditaumassnncorr.Fill(test_ditaumass_combined[g],ditaumass_nn[g],weightlist_combined_test[g])
        #histditaumassnn.Fill(ditaumass_nn[g],weightlist_test[g])
        #histditaumassnnoverfit.Fill(ditaumass_nn_overfit[g],weightlist_overfit[g])
        #histditaumassnncorr.Fill(test_ditaumass[g],ditaumass_nn[g],weightlist_test[g])
    #for g in range(len(ditaumass_nn)):
    #    histditaumassnncorr.Fill(ditaumass_test_highmass[g],ditaumass_nn[g])
    for k in range(100):
        if histditaumass.GetBinContent(k+1) != 0:
            #if histditauvismass.GetBinContent(k+1) != 0:
            content_nn = histditaumassnn.GetBinContent(k+1)
            content_actual = histditaumass.GetBinContent(k+1)
            #content_actual = histditauvismass.GetBinContent(k+1)
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
    histditaumassnn.Write()
    histditaumassnnoverfit.Write()
    histditaumassnnratio.Write()
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
    loss_graph.Write()
    val_loss_graph.Write()
    canv1.Write()
    img1 = ROOT.TImage.Create()
    img1.FromPad(canv1)
    img1.WriteImage(output_plot_name)

        
def fill_histditaumass():
    for i,ditaumass_value in enumerate(test_ditaumass):
        histditaumass.Fill(ditaumass_value)
        #histditaumass.Fill(ditaumass_value,weightlist_test[i])

def fill_histditauvismass():
    for f,ditauvismass_value in enumerate(test_ditauvismass):
        histditauvismass.Fill(ditauvismass_value)
        #histditauvismass.Fill(ditauvismass_value,weightlist_test[f])

def fill_histditaumass_combined():
    for i,ditaumass_value in enumerate(test_ditaumass_combined):
        histditaumass.Fill(ditaumass_value,weightlist_combined_test[i])

def fill_histditauvismass_combined():
    for f,ditauvismass_value in enumerate(test_ditauvismass_combined):
        histditauvismass.Fill(ditauvismass_value,weightlist_combined_test[f])

def fill_histditaumass_highmass():
    for g in ditaumass_test_highmass:
        histditaumass.Fill(g)

def fill_histditauvismass_highmass():
    for f in ditauvismass_test_highmass:
        histditauvismass.Fill(f)

def fill_histditaumass_cut():
    for i,ditaumass_value in enumerate(ditaumass_cut):
	histditaumass.Fill(ditaumass_value)
def fill_histditauvismass_cut():
    for i,ditauvismass_value in enumerate(ditauvismass_cut):
	histditauvismass.Fill(ditauvismass_value)

def fill_histditaumass_selected():
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        histditaumass.Fill(ditaumass_value)
"""
#####################  getting the SVfit histogram ####################
svfit = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/batch_nnvssvfit_output/ditau_mass_svfit_try116_cuts.root")
histsvfit = svfit.Get("ditaumasssvfit")
histsvfitratio = svfit.Get("ditaumasssvfitratio")
for i in range(150):
    binvalue = histsvfit.GetBinContent(i)
    histditaumasssvfit.SetBinContent(i,binvalue)
for j in range(150):
    ratio = histsvfitratio.GetBinContent(j)
    ratio_error = histsvfitratio.GetBinError(j)
    histditaumasssvfitratio.SetBinContent(j,ratio)
    histditaumasssvfitratio.SetBinError(j,ratio_error)
svfit.Close()
###################################################################################
"""
"""
nn_best = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/thesis_plots/ditau_mass_nn_drellyan_best14.root")
histnnbest = nn_best.Get("ditaumassnn")
histnnbestratio = nn_best.Get("ditaumassregallratio")
for i in range(100):
    binvalue = histnnbest.GetBinContent(i+1)
    histditaumassnn.SetBinContent(i+1,binvalue)
    ratio = histnnbestratio.GetBinContent(i+1)
    ratio_error = histnnbestratio.GetBinError(i+1)
    histditaumassnnratio.SetBinContent(i+1,ratio)
    histditaumassnnratio.SetBinError(i+1,ratio_error)
nn_best.Close()
"""

#output_name = "ditau_mass_nnvssvfit_cuts_try1"
output_name = "ditau_mass_nn_drellyan_best26"
#output_name = "ditau_mass_nn_larger_range_try1"
description_of_training = "Standardized input data according to training data INPUT:vis tau1 (pt,eta,phi,mass)+vis tau2 (pt,eta,phi,mass)+METx+METy+p vis+vismass"
#description_of_training = "input data  INPUT: 1000 (classification of decaychannel) vis tau1 (pt,eta,phi,E,mass)+vis tau2 (pt,eta,phi,E,mass)+MET_ET+MET_Phi+ditaumass_collinear"
output_file_name = "%s.txt" % (output_name)
output_root_name = "%s.root" % (output_name)
rootfile = ROOT.TFile(output_root_name,"RECREATE")
output_file = open(output_file_name,'w')
sys.stdout = output_file

print "dataset length:",dataset_length
print "train length:",train_length
print "test length:",test_length

fill_histditaumass()
fill_histditauvismass()
#fill_histditaumass_selected()
#fill_histditaumass_combined()
#fill_histditauvismass_combined()
#fill_histditaumass_highmass()
#fill_histditauvismass_highmass()
#fill_histditaumass_cut()
#fill_histditauvismass_cut()

histditaumass.Write()
histditauvismass.Write()

######################### run neural network  ######################

batch_size = 128
epochs = 100

start_nn = time.time()
neural_network(batch_size,epochs,output_name)
end_nn = time.time()

print "NN executon time:",(end_nn-start_nn)/3600,"h" 


#histogram of di-tau mass using regression all decays

canv2 = ROOT.TCanvas("di-tau mass using NN and SVfit")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.32,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.35,0.03)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum())
histditaumass.SetMaximum(max_bin*1.08)
histditaumass.Draw("HIST")
#histditauvismass.Draw("HIST")
histditaumassnn.Draw("HIST SAME")
#histditaumassnnoverfit.Draw("HIST SAME")
#histditaumasssvfit.Draw("SAME")
#histditauvismass.Draw("SAME")
leg2 = ROOT.TLegend(0.75,0.67,0.97,0.87)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
#leg2.AddEntry(histditauvismass,"di-#tau_{vis} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau mass","PL")
#leg2.AddEntry(histditaumassnnoverfit,"overfit-test","PL")
#leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
leg2.SetTextSize(0.05)
leg2.Draw()
pad2.cd()
histditaumassnnratio.Draw("P")
unit_line = ROOT.TLine(0.0,1.0,100.0,1.0)
unit_line.SetLineColor(2)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
histditaumassnn.Write()
canv2.Write()
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)

canv3 = ROOT.TCanvas("ditaumassnncorr")
histditaumassnncorr.Draw()
line = ROOT.TLine(0.0,0.0,100.0,100.0)
line.Draw("SAME")
output_hist_corr_name = "%s_corr.png" %(output_name)
histditaumassnncorr.Write()
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

canv5 = ROOT.TCanvas("mass and visible mass")
max_bin5 = max(histditaumass.GetMaximum(),histditauvismass.GetMaximum())
histditauvismass.SetMaximum(max_bin5*1.08)
histditauvismass.Draw()
histditaumass.Draw("SAME")
output_hist_genmass_name = "%s_gen.png" %(output_name)
leg5 = ROOT.TLegend(0.55,0.67,0.87,0.87)
leg5.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg5.AddEntry(histditauvismass,"visible di-#tau_{gen} mass","PL")
leg5.SetTextSize(0.04)
leg5.Draw()
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_genmass_name)

"""
canv5 = ROOT.TCanvas("resolution0to40")
histditaumassnnres040.Draw()
histditaumassnnres040.Write()
output_hist_res040_name = "%s_res040.png" %(output_name)
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_res040_name)

canv6 = ROOT.TCanvas("resolution40to80")
histditaumassnnres4080.Draw()
histditaumassnnres4080.Write()
output_hist_res4080_name = "%s_res4080.png" %(output_name)
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage(output_hist_res4080_name)

canv7 = ROOT.TCanvas("resolution80to100")
histditaumassnnres80100.Draw()
histditaumassnnres80100.Write()
output_hist_res80100_name = "%s_res80100.png" %(output_name)
img7 = ROOT.TImage.Create()
img7.FromPad(canv7)
img7.WriteImage(output_hist_res80100_name)

canv8 = ROOT.TCanvas("resolutionabove100")
histditaumassnnresabove100.Draw()
histditaumassnnresabove100.Write()
output_hist_resabove100_name = "%s_resabove100.png" %(output_name)
img8 = ROOT.TImage.Create()
img8.FromPad(canv8)
img8.WriteImage(output_hist_resabove100_name)
"""
output_file.close()
rootfile.Close()
