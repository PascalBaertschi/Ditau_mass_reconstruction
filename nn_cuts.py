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


#################  choose size of used dataset ######################
#fixed_dataset_length = 2640000
#fixed_dataset_length = 1910000
fixed_dataset_length = 100000000
fixed_train_length = 560000
fixed_test_length = 100000
fulllep_fixed_length = int(round(0.1226*fixed_dataset_length))
semilep_fixed_length = int(round(0.4553*fixed_dataset_length))
fullhad_fixed_length = int(round(0.422* fixed_dataset_length))
###################################################################################
#list_name_normal = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_morelowmass.csv"
#list_name_normal = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_tight.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_tight_morelowmass.csv"
#list_name_normal = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_notsotight.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_notsotight_morelowmass.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_skim_correct40.csv"
list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_correct29_nocut.csv"
#dataframe_ditaumass_normal = pandas.read_csv(list_name_normal,delim_whitespace=False,header=None)
#dataframe_ditaumass_skim = pandas.read_csv(list_name_skim,delim_whitespace=False,header=None)
dataframe_ditaumass = pandas.read_csv(list_name_skim,delim_whitespace=False,header=None)
#dataframe_ditaumass = dataframe_ditaumass_normal.append(dataframe_ditaumass_skim)
dataframe_ditaumass_shuffled = dataframe_ditaumass.sample(frac=1,random_state =1337)
dataset_ditaumass = dataframe_ditaumass_shuffled.values
dataset_total_length = len(dataset_ditaumass[:,0])

histwholedatacheck = ROOT.TH1D("wholedatacheck","datacheck",300,0,300)
histtestdata = ROOT.TH1D("testdatacheck","test sample check",300,0,300)
inputNN = []
ditaumass = []
ditauvismass = []
decaymode_count = 0
decaymode_count_fulllep = 0
decaymode_count_semilep = 0
decaymode_count_fullhad = 0
ptzero_count = 0

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
    ditaumass_collinear = vismass/numpy.sqrt(vistau1_pt/(vistau1_pt+pt_nu)*vistau2_pt/(vistau2_pt+pt_nu))
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    histwholedatacheck.Fill(ditaumass_value)
    if tau1_pt == 0 or tau2_pt == 0 or vistau1_pt == 0 or vistau2_pt == 0:
        ptzero_count += 1
    if decaymode_count < fixed_dataset_length and ditaumass_value > 80.0:
        if vistau1_att in (1,2) and vistau2_att in (1,2) and decaymode_count_fulllep < fulllep_fixed_length:
            inputNN.append([1.0,0.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_fulllep += 1
            decaymode_count += 1
        elif vistau1_att in (1,2) and vistau2_att == 3 and decaymode_count_semilep < semilep_fixed_length:
            inputNN.append([0.0,1.0,0.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_semilep += 1
            decaymode_count += 1
        elif vistau1_att == 3 and vistau2_att in (1,2) and decaymode_count_semilep < semilep_fixed_length:
            inputNN.append([0.0,0.0,1.0,0.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_semilep += 1
            decaymode_count += 1
        elif vistau1_att == 3 and vistau2_att == 3 and decaymode_count_fullhad < fullhad_fixed_length:
            inputNN.append([0.0,0.0,0.0,1.0,vistau1_pt,vistau1_eta,vistau1_phi,vistau1_E,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_E,vistau2_mass, genMissingET_MET,genMissingET_Phi,ditaumass_collinear])
            #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_fullhad += 1 
            decaymode_count += 1


inputNN = numpy.array(inputNN,numpy.float64)
"""
mean = numpy.mean(inputNN[:,4])
std = numpy.std(inputNN[:,4])
print "mean pt vistau1:",mean
print "std pt vistau1:", std
#inputNN[:,:] = preprocessing.scale(inputNN[:,:])
check_list = []
for i in inputNN[:,4]:
    pt_new = (i-mean)/std
    check_list.append(pt_new)
check_list = numpy.array(check_list,numpy.float64)
print "STANDARDIZATION"
print "mean pt vistau1:",numpy.mean(check_list)
print "std pt vistau1:", numpy.std(check_list)
"""
dataset_length = len(inputNN[:,0])

train_inputNN = inputNN[fixed_test_length:,:]
train_ditaumass = ditaumass[fixed_test_length:]
train_ditauvismass = ditauvismass[fixed_test_length:]
test_inputNN = inputNN[0:fixed_test_length,:]
test_ditaumass = ditaumass[0:fixed_test_length]
test_ditauvismass = ditauvismass[0:fixed_test_length]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
overfit_ditaumass = train_ditaumass[0:len(test_ditaumass)]
overfit_ditauvismass = train_ditauvismass[0:len(test_ditauvismass)]

train_inputNN_forselection = inputNN[fixed_test_length:,:]
train_ditaumass_forselection = ditaumass[fixed_test_length:]
test_inputNN_selected = inputNN[0:fixed_test_length,:]
test_ditaumass_selected = ditaumass[0:fixed_test_length]
test_ditauvismass_selected = ditauvismass[0:fixed_test_length]

######### make flat mass distribution with less events ######################
min_bincontent = 4900
#min_bincontent = 5500
histtrainditaumasscalc = ROOT.TH1D("trainditaumasscalc","trainditaumasscalc",300,0,300)


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

histtrainditaumasscheck = ROOT.TH1D("trainditaumasscheck","train sample of neural network",350,0,350)
histtrainditaumasscheck.GetXaxis().SetTitle("di-#tau mass [GeV]")
histtrainditaumasscheck.GetYaxis().SetTitle("number of occurence")
histtrainditaumasscheck.GetYaxis().SetTitleOffset(1.2)
histtrainditaumasscheck.SetStats(0)

for j in train_ditaumass_selected:
    histtrainditaumasscheck.Fill(j)

overfit_inputNN_selected = train_inputNN_selected[0:len(test_ditaumass_selected),:]
overfit_ditaumass_selected = train_ditaumass_selected[0:len(test_ditaumass_selected)]


###########     standardization of input data     ##################
test_inputNN_selected_stand = test_inputNN_selected
for j in range(4,len(train_inputNN_selected[0,:])):
    mean = numpy.mean(train_inputNN_selected[:,j])
    std = numpy.std(train_inputNN_selected[:,j])
    for i in range(0,len(test_ditaumass_selected)):
        value = test_inputNN_selected[i,j]
        new_value = (value-mean)/std
        test_inputNN_selected_stand[i,j] = new_value

test_inputNN_selected = numpy.array(test_inputNN_selected_stand,numpy.float64)        
train_stand = preprocessing.scale(train_inputNN_selected[:,4:])
train_inputNN_set = []
test_inputNN_set = []
for j in range(0,len(train_ditaumass_selected)):
    train_inputNN_set.append([train_inputNN_selected[j,0],train_inputNN_selected[j,1],train_inputNN_selected[j,2],train_inputNN_selected[j,3],train_stand[j,0],train_stand[j,1],train_stand[j,2],train_stand[j,3],train_stand[j,4],train_stand[j,5],train_stand[j,6],train_stand[j,7],train_stand[j,8],train_stand[j,9],train_stand[j,10],train_stand[j,11],train_stand[j,12]])

train_inputNN_selected = numpy.array(train_inputNN_set,numpy.float64)

###########     standardization of input data     ##################
#train_inputNN_selected[:,:] = preprocessing.normalize(train_inputNN_selected[:,:])
#test_inputNN_selected[:,:] = preprocessing.normalize(test_inputNN_selected[:,:])
#train_inputNN[:,:] = preprocessing.scale(train_inputNN[:,:])
#test_inputNN[:,:] = preprocessing.scale(test_inputNN[:,:])
#overfit_inputNN[:,:] = preprocessing.scale(overfit_inputNN[:,:])
#train_inputNN_selected[:,:] = preprocessing.scale(train_inputNN_selected[:,:])
#test_inputNN_selected[:,:] = preprocessing.scale(test_inputNN_selected[:,:])
#overfit_inputNN_selected[:,:] = preprocessing.scale(overfit_inputNN_selected[:,:])
#############    preparing the histograms       ###################

#histogram of ditau mass using neural network and SVfit
histtitle = "reconstructed di-#tau mass using neural network and SVfit"
histditaumass = ROOT.TH1D("ditaumass",histtitle,70,0,350)
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleOffset(1.2)
histditaumass.SetLineColor(2)
histditaumass.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","reconstructed di-#tau vismass using neural network",70,0,350)
histditauvismass.SetLineColor(6)
histditauvismass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassnn","reconstructed di-#tau mass using neural network",70,0,350)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",70,0,350)
histditaumasssvfit.SetLineColor(3)
histditaumasssvfit.SetStats(0)
histditaumassnnoverfit = ROOT.TH1D("ditaumassnnoverfit","overfit of neural network",70,0,350)
histditaumassnnoverfit.SetLineColor(3)
histditaumassnnoverfit.SetLineStyle(2)
histditaumassnnoverfit.SetStats(0)

histditaumassnncorr = ROOT.TH2D("ditaumassnncorr","di-#tau_{gen} mass vs di-#tau mass",400,0,400,400,0,400)
histditaumassnncorr.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnncorr.GetYaxis().SetTitle("di-#tau mass [GeV]")
histditaumassnncorr.SetStats(0)

profditaumassnncorrrms = ROOT.TProfile("profditaumassnncorrrms","di-#tau_{gen} mass vs mean(reconstructed di-#tau mass",350,0,350,"s")

histditaumassnnrms =ROOT.TH1D("ditaumassnnrms","RMS per di-#tau_{gen} mass",350,0,350)
histditaumassnnrms.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnnrms.GetYaxis().SetTitle("RMS")
histditaumassnnrms.SetStats(0)
histditaumassnnrms.SetMarkerStyle(7)

histditaumassnnres = ROOT.TH1D("resolution","resolution of di-#tau mass using neural network",80,-1,1)
histditaumassnnres.GetXaxis().SetTitle("resolution")
histditaumassnnres.GetYaxis().SetTitle("number of occurence")
histditaumassnnres.GetYaxis().SetTitleOffset(1.4)
histditaumassnnres.SetLineWidth(3)
histditaumassnnrescomp = ROOT.TH1D("NN","resolution comparison for NN and SVfit",80,-1,1)
histditaumassnnrescomp.GetXaxis().SetTitle("resolution")
histditaumassnnrescomp.GetYaxis().SetTitle("number of occurence")
histditaumassnnrescomp.GetYaxis().SetTitleOffset(1.4)
histditaumassnnrescomp.SetLineColor(4)
histditaumassnnrescomp.SetLineWidth(3)
histditaumasssvfitres = ROOT.TH1D("SVfit","resolution of di-#tau mass using SVfit",80,-1,1)
histditaumasssvfitres.SetLineColor(3)
histditaumasssvfitres.SetLineWidth(3)


histditaumassnncorrres = ROOT.TH2D("ditaumassnncorrres","resolution per di-#tau_{gen} mass",350,0,350,80,-1,1)
histditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnncorrres.GetYaxis().SetTitle("resolution")
histditaumassnncorrres.SetStats(0)

profditaumassnncorrres = ROOT.TProfile("profditaumassnncorrres","mean of resolution per di-#tau_{gen} mass",350,0,350)
profditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumassnncorrres.GetYaxis().SetTitle("mean(resolution)")
profditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
profditaumassnncorrres.SetStats(0)
profditaumassnncorrres.SetMarkerStyle(7)
histprofditaumassnncorrres = ROOT.TH1D("histprofditaumassnncorrres","mean of resolution per di-#tau_{gen} mass",350,0,350)
histprofditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histprofditaumassnncorrres.GetYaxis().SetTitle("mean(resolution)")
histprofditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
histprofditaumassnncorrres.SetStats(0)
histprofditaumassnncorrres.SetLineColor(4)
histprofditaumassnncorrres.SetMarkerStyle(7)
histprofditaumassnncorrres.SetMarkerColor(4)
histprofditaumasssvfitcorrres = ROOT.TH1D("histprofditaumasssvfitcorrres","mean of resolution per di-#tau_{gen} mass",350,0,350)
histprofditaumasssvfitcorrres.SetStats(0)
histprofditaumasssvfitcorrres.SetLineColor(3)
histprofditaumasssvfitcorrres.SetMarkerStyle(7)
histprofditaumasssvfitcorrres.SetMarkerColor(3)

profditaumassnncorrabsres = ROOT.TProfile("profditaumassnncorrabsres","mean of |resolution| per di-#tau_{gen} mass",350,0,350)
profditaumassnncorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumassnncorrabsres.GetYaxis().SetTitle("mean(|resolution|)")
profditaumassnncorrabsres.GetYaxis().SetTitleOffset(1.2)
profditaumassnncorrabsres.SetStats(0)
profditaumassnncorrabsres.SetMarkerStyle(7)
histprofditaumassnncorrabsres = ROOT.TH1D("histprofditaumassnncorrabsres","mean of |resolution| per di-#tau_{gen} mass",350,0,350)
histprofditaumassnncorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histprofditaumassnncorrabsres.GetYaxis().SetTitle("mean(|resolution|)")
histprofditaumassnncorrabsres.GetYaxis().SetTitleOffset(1.2)
histprofditaumassnncorrabsres.SetStats(0)
histprofditaumassnncorrabsres.SetLineColor(4)
histprofditaumassnncorrabsres.SetMarkerStyle(7)
histprofditaumassnncorrabsres.SetMarkerColor(4)
histprofditaumasssvfitcorrabsres = ROOT.TH1D("histprofditaumasssvfitcorrabsres","mean of |resolution| per di-#tau_{gen} mass",350,0,350)
histprofditaumasssvfitcorrabsres.SetStats(0)
histprofditaumasssvfitcorrabsres.SetLineColor(3)
histprofditaumasssvfitcorrabsres.SetMarkerStyle(7)
histprofditaumasssvfitcorrabsres.SetMarkerColor(3)

#ratio histograms
histditaumassnnratio = ROOT.TH1D("ditaumassregallratio","ratio between reconstruced and actual mass",70,0,350)
histditaumassnnratio.SetTitle("")
histditaumassnnratio.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnnratio.GetXaxis().SetLabelSize(0.08)
histditaumassnnratio.GetXaxis().SetTitleSize(0.08)
histditaumassnnratio.GetYaxis().SetTitle("ratio")
histditaumassnnratio.GetYaxis().SetLabelSize(0.08)
histditaumassnnratio.GetYaxis().SetTitleSize(0.08)
histditaumassnnratio.GetYaxis().SetTitleOffset(0.5)
histditaumassnnratio.GetYaxis().SetNdivisions(404)
histditaumassnnratio.GetYaxis().CenterTitle()
histditaumassnnratio.GetYaxis().SetRangeUser(0.0,2.0)
histditaumassnnratio.SetMarkerStyle(7)
histditaumassnnratio.SetMarkerColor(4)
histditaumassnnratio.SetStats(0)
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and actual mass",70,0,350)
histditaumasssvfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(3)
histditaumasssvfitratio.SetStats(0)
histditaumassnnoverfitratio = ROOT.TH1D("ditaumassregalloverfitratio","ratio between reconstruced and overfit mass",70,0,350)
histditaumassnnoverfitratio.SetMarkerStyle(7)
histditaumassnnoverfitratio.SetMarkerColor(8)
histditaumassnnoverfitratio.SetStats(0)



def neural_network(batch_size,epochs,output_name):
    print "NEURAL NETWORK"
    mass_model = Sequential()
    mass_model.add(Dense(200,input_dim=17,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    #mass_model.add(Dense(200,input_dim=17,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    
    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate*math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate
    #learning_rate = 0.001
    #decay_rate =  0.00001
    #adam = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = decay_rate)
    #adam = Adam(lr = 0.0, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = 0.0)
    mass_model.compile(loss='mean_squared_error',optimizer='adam')
    #mass_model.compile(loss='mean_squared_error',optimizer=adam)
    #lrate = [LearningRateScheduler(step_decay)]
    #history = mass_model.fit(train_inputNN,train_ditaumass,batch_size,epochs,validation_data = (test_inputNN,test_ditaumass),verbose = 2)
    #mass_score = mass_model.evaluate(test_inputNN,test_ditaumass,batch_size,verbose=0)
    #ditaumass_nn = mass_model.predict(test_inputNN,batch_size,verbose=0)
    #ditaumass_nn_overfit = mass_model.predict(overfit_inputNN,batch_size,verbose=0)
    #history = mass_model.fit(train_inputNN_selected,train_ditaumass_selected,batch_size,epochs,validation_data = (test_inputNN_selected,test_ditaumass_selected),callbacks = lrate,verbose = 2)
    history = mass_model.fit(train_inputNN_selected,train_ditaumass_selected,batch_size,epochs,validation_data = (test_inputNN_selected,test_ditaumass_selected),verbose = 2)
    mass_score = mass_model.evaluate(test_inputNN_selected,test_ditaumass_selected,batch_size,verbose=0)
    ditaumass_nn = mass_model.predict(test_inputNN_selected,batch_size,verbose=0)
    #ditaumass_nn_overfit = mass_model.predict(overfit_inputNN_selected,batch_size,verbose=0)
    #history = mass_model.fit(train_inputNN_selected,train_ditaumass_selected,batch_size,epochs,validation_data = (inputNN_cuts,ditaumass_cuts),verbose = 2)
    #mass_score = mass_model.evaluate(inputNN_cuts,ditaumass_cuts,batch_size,verbose=0)
    #ditaumass_nn = mass_model.predict(inputNN_cuts,batch_size,verbose=0)
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

def fill_histditauvismass():
    for f,ditauvismass_value in enumerate(test_ditauvismass):
        histditauvismass.Fill(ditauvismass_value)

def fill_histditaumass_selected():
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        histditaumass.Fill(ditaumass_value)

def fill_histditauvismass_selected():
    for i,ditauvismass_value in enumerate(test_ditauvismass_selected):
        histditauvismass.Fill(ditauvismass_value)

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


#####################  getting the SVfit histograms ####################
svfit = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/final_plots/ditau_mass_svfit_nocuts_1e5_32.root")
histsvfit = svfit.Get("ditaumasssvfit")
histsvfitratio = svfit.Get("ditaumasssvfitratio")
histsvfitres = svfit.Get("resolution")
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

svfit.Close()
###################################################################################

output_name = "ditau_mass_nn_final_nocuts_9"
#output_name = "ditau_mass_nn_cuts_final_a4test"
#description_of_training = "Standardized input data  INPUT:vis tau1 (pt,eta,phi,mass)+vis tau2 (pt,eta,phi,mass)+METx+METy+p vis+vismass"
description_of_training = "standardized input data  INPUT: 1000 (classification of decaychannel) vis tau1 (pt,eta,phi,E,mass)+vis tau2 (pt,eta,phi,E,mass)+MET_ET+MET_Phi+collinear ditaumass"
output_file_name = "%s.txt" % (output_name)
output_root_name = "%s.root" % (output_name)
rootfile = ROOT.TFile(output_root_name,"RECREATE")
output_file = open(output_file_name,'w')
sys.stdout = output_file

print "dataset length:",len(ditaumass)
#print "train length:",len(train_ditaumass)
#print "test length:",len(test_ditaumass)
print "train length:",len(train_ditaumass_selected)
print "test length:",len(test_ditaumass_selected)


#fill_histditaumass()
#fill_histditauvismass()
fill_histditaumass_selected()
fill_histditauvismass_selected()

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
pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.3,0.02)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum())
histditaumass.SetMaximum(max_bin*1.2)
histditaumass.Draw("HIST")
#histditauvismass.Draw("HIST")
histditaumassnn.Draw("HIST SAME")
#histditaumassnnoverfit.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
#histditauvismass.Draw("SAME")
leg2 = ROOT.TLegend(0.1,0.68,0.35,0.9)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
#leg2.AddEntry(histditauvismass,"di-#tau_{vis} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau_{NN} mass","PL")
#leg2.AddEntry(histditaumassnnoverfit,"overfit-test","PL")
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
nn_statbox.SetTextColor(nn_color)
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
svfit_statbox.SetTextColor(svfit_color)
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
leg9 = ROOT.TLegend(0.1,0.8,0.3,0.9)
leg9.AddEntry(histprofditaumassnncorrres,"NN","PL")
leg9.AddEntry(histprofditaumasssvfitcorrres,"SVfit","PL")
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
leg11 = ROOT.TLegend(0.1,0.8,0.3,0.9)
leg11.AddEntry(histprofditaumassnncorrabsres,"NN","PL")
leg11.AddEntry(histprofditaumasssvfitcorrabsres,"SVfit","PL")
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
canv10.Write()
img12 = ROOT.TImage.Create()
img12.FromPad(canv12)
img12.WriteImage(output_rms_name)

canv13 = ROOT.TCanvas("di-tau mass using NN and SVfit including vismass")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.3,0.02)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum(),histditauvismass.GetMaximum())
histditaumass.SetMaximum(max_bin*1.3)
histditaumass.Draw("HIST")
histditauvismass.Draw("HIST SAME")
histditaumassnn.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
leg2 = ROOT.TLegend(0.1,0.63,0.35,0.9)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg2.AddEntry(histditauvismass,"di-#tau_{vis} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau_{NN} mass","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
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


output_file.close()
rootfile.Close()

