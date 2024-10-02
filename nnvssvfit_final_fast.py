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

# importing the python binding to the C++ class from ROOT
class SVfitAlgo(ROOT.SVfitStandaloneAlgorithm):
    '''Just an additional wrapper, not really needed :-)
    We just want to illustrate the fact that you could
    use such a wrapper to add functions, attributes, etc,
    in an improved interface to the original C++ class.
    '''
    def __init__(self, *args):
        super(SVfitAlgo, self).__init__(*args)


class measuredTauLepton(ROOT.svFitStandalone.MeasuredTauLepton):
    '''
       decayType : {
                    0:kUndefinedDecayType,
                    1:kTauToHadDecay,
                    2:kTauToElecDecay,
                    3:kTauToMuDecay,
                    4:kPrompt
                   }
    '''
    def __init__(self, decayType, pt, eta, phi, mass, decayMode=-1):
        super(measuredTauLepton, self).__init__(decayType, pt, eta, phi, mass, decayMode)


#################  choose size and decaymode of used dataset ######################

decaymode = 'all'
fixed_dataset_length = 791000
#fixed_dataset_length = 1000

###################################################################################
list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_8e5_%s.csv" %(decaymode)

#dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_5.5e6.csv",delim_whitespace=False,header=None)
dataframe_ditaumass = pandas.read_csv(list_name,delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values
dataset_total_length = len(dataset_ditaumass[:,0])


histCOV00 = ROOT.TH1D("COV00","histogram of COV00",100,-20,20)
histCOV11 = ROOT.TH1D("COV11","histogram of COV11",100,-20,20)

for i in range(0,dataset_total_length):
    genMissingET_MET = dataset_ditaumass[i,18]
    genMissingET_Phi = dataset_ditaumass[i,19]
    MissingET_MET = dataset_ditaumass[i,20]
    MissingET_Phi = dataset_ditaumass[i,21]
    genMETpx = genMissingET_MET*numpy.cos(genMissingET_Phi)
    genMETpy = genMissingET_MET*numpy.sin(genMissingET_Phi)
    METpx = MissingET_MET*numpy.cos(MissingET_Phi)
    METpy = MissingET_MET*numpy.sin(MissingET_Phi)
    METCOV00 = METpx-genMETpx
    METCOV11 = METpy-genMETpy
    histCOV00.Fill(METCOV00)
    histCOV11.Fill(METCOV11)

histCOV00.Fit("gaus")
COV00_fit = histCOV00.GetFunction("gaus")
COV00_width = COV00_fit.GetParameter(2)
histCOV11.Fit("gaus")
COV11_fit = histCOV11.GetFunction("gaus")
COV11_width = COV11_fit.GetParameter(2)
COV_width_mean = (COV00_width+COV11_width)/2
METCOV = ROOT.TMath.Power(COV_width_mean,2.)

inputNN = []
inputSVfit = []
ditaumass = []
ditauvismass = []
decaymode_count = 0

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
    genMissingET_MET = dataset_ditaumass[i,18]
    genMissingET_Phi = dataset_ditaumass[i,19]
    MissingET_MET = dataset_ditaumass[i,20]
    MissingET_Phi = dataset_ditaumass[i,21]

    genMETpx = genMissingET_MET*numpy.cos(genMissingET_Phi)
    genMETpy = genMissingET_MET*numpy.sin(genMissingET_Phi)
    METpx = MissingET_MET*numpy.cos(MissingET_Phi)
    METpy = MissingET_MET*numpy.sin(MissingET_Phi)
    METCOV00 = METpx-genMETpx
    METCOV11 = METpy-genMETpy
    METCOV00 = ROOT.TMath.Power(METpx-genMETpx,2.)
    METCOV11 = ROOT.TMath.Power(METpy-genMETpy,2.)

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
    vismass = v_vismot.M()
    p_vis = v_vismot.P()
    pt_vis = v_vismot.Pt()
    pt_nu = v_nu.Pt()
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    if decaymode_count < fixed_dataset_length:
        inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
        ditaumass.append(ditaumass_value)
        ditauvismass.append(ditauvismass_value)
        decaymode_count += 1
        inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
    else:
        break
        

    """
    if decaymode == 'all':
        if decaymode_count < fixed_dataset_length:
            inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count += 1
            if vistau1_pt == 0.0 or vistau2_pt == 0.0:
                continue
            else:
                inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
        else:
            break
    elif decaymode == 'leptonic':
        if decaymode_count < fixed_dataset_length:
            if vistau1_att in (1,2) and vistau2_att in (1,2):
                inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
                ditaumass.append(ditaumass_value)
                ditauvismass.append(ditauvismass_value)
                decaymode_count += 1
                if vistau1_pt == 0.0 or vistau2_pt == 0.0:
                    continue
                else:
                    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
        else:
            break
    elif decaymode == 'semileptonic':
        if decaymode_count < fixed_dataset_length:
            if (vistau1_att in (1,2) and vistau2_att == 3) or (vistau1_att == 3 and vistau2_att in (1,2)):
                inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
                ditaumass.append(ditaumass_value)
                ditauvismass.append(ditauvismass_value)
                decaymode_count += 1
                if vistau1_pt == 0.0 or vistau2_pt == 0.0:
                    continue
                else:
                    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
        else:
            break
    elif decaymode == 'hadronic':
        if decaymode_count < fixed_dataset_length:
            if (vistau1_att == 3 and vistau2_att == 3):
                inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
                ditaumass.append(ditaumass_value)
                ditauvismass.append(ditauvismass_value)
                decaymode_count += 1
                if vistau1_pt == 0.0 or vistau2_pt == 0.0:
                    continue
                else:
                    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
        else:
            break
    """

inputNN = numpy.array(inputNN,numpy.float64)
ditaumass = numpy.array(ditaumass,numpy.float64)
inputSVfit = numpy.array(inputSVfit,numpy.float64)

dataset_length = len(inputNN[:,0])
train_length = int(round(dataset_length*0.7))
test_length = int(round(dataset_length*0.3))
dataset_svfit_length = len(inputSVfit[:,0])
train_svfit_length = dataset_svfit_length - test_length


train_inputNN = inputNN[0:train_length,:]
train_ditaumass = ditaumass[0:train_length]
test_inputNN = inputNN[train_length:dataset_length,:]
test_ditaumass = ditaumass[train_length:dataset_length]
test_ditauvismass = ditauvismass[train_length:dataset_length]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
inputSVfit = inputSVfit[train_svfit_length:dataset_svfit_length,:]
inputNN = preprocessing.scale(inputNN)


#histogram of ditau mass using neural network and SVfit
histtitle = "reconstructed di-#tau mass using neural network and SVfit %s decays" % (decaymode)
histditaumass = ROOT.TH1D("ditaumass",histtitle,100,0,100)
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleOffset(1.2)
histditaumass.SetLineColor(2)
histditaumass.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","di-#tau visible mass",100,0,100)
histditauvismass.SetLineColor(6)
histditauvismass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassregall","reconstructed di-#tau mass using neural network",100,0,100)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",100,0,100)
histditaumasssvfit.SetLineColor(3)
histditaumasssvfit.SetStats(0)
histditaumassnnoverfit = ROOT.TH1D("ditaumassnnoverfit","di-#tau mass overfit of NN",100,0,100)
histditaumassnnoverfit.SetLineColor(1)
histditaumassnnoverfit.SetLineStyle(2)
histditaumassnnoverfit.SetStats(0)


####SVfit delta histogram
histditaumasssvfitdelta = ROOT.TH1D("ditaumasssvfitdelta","delta between reconstructed di-#tau mass using SVfit and actual mass",100,-10,50)
histditaumasssvfitdelta.GetXaxis().SetTitle("#Deltam [GeV]")
histditaumasssvfitdelta.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitdelta.SetStats(0)
histditaumasssvfitres = ROOT.TH1D("ditaumasssvfitres","resolution of reconstructed di-#tau mass using SVfit",100,-5,5)
histditaumasssvfitres.GetXaxis().SetTitle("resolution")
histditaumasssvfitres.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitres.SetStats(0)


#ratio histograms
histditaumassnnratio = ROOT.TH1D("ditaumassregallratio","ratio between reconstruced and actual mass",100,0,100)
histditaumassnnratio.SetTitle("")
histditaumassnnratio.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassnnratio.GetXaxis().SetLabelSize(0.08)
histditaumassnnratio.GetXaxis().SetTitleSize(0.08)
histditaumassnnratio.GetYaxis().SetTitle("ratio")
histditaumassnnratio.GetYaxis().SetLabelSize(0.08)
histditaumassnnratio.GetYaxis().SetTitleSize(0.08)
histditaumassnnratio.GetYaxis().SetTitleOffset(0.5)
histditaumassnnratio.GetYaxis().SetNdivisions(504)
histditaumassnnratio.GetYaxis().CenterTitle()
histditaumassnnratio.GetYaxis().SetRangeUser(0.0,5.0)
histditaumassnnratio.SetMarkerStyle(7)
histditaumassnnratio.SetStats(0)
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and actual mass",100,0,100)
histditaumasssvfitratio.SetTitle("")
histditaumasssvfitratio.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumasssvfitratio.GetXaxis().SetLabelSize(0.08)
histditaumasssvfitratio.GetXaxis().SetTitleSize(0.08)
histditaumasssvfitratio.GetYaxis().SetTitle("ratio")
histditaumasssvfitratio.GetYaxis().SetLabelSize(0.08)
histditaumasssvfitratio.GetYaxis().SetTitleSize(0.08)
histditaumasssvfitratio.GetYaxis().SetTitleOffset(0.5)
histditaumasssvfitratio.GetYaxis().SetNdivisions(504)
histditaumasssvfitratio.GetYaxis().CenterTitle()
histditaumasssvfitratio.GetYaxis().SetRangeUser(0.0,5.0)
histditaumasssvfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(3)
histditaumasssvfitratio.SetStats(0)
histditaumassnnoverfitratio = ROOT.TH1D("ditaumassregalloverfitratio","ratio between reconstruced and overfit mass",100,0,100)
histditaumassnnoverfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(4)
histditaumassnnoverfitratio.SetStats(0)

def neural_network(batch_size,epochs,output_name,return_list_nn,return_list_nn_loss):
    start_nn = time.time()
    print "NEURAL NETWORK started"
    mass_model = Sequential()
    mass_model.add(Dense(200,input_dim=12,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    mass_model.compile(loss='mean_squared_error',optimizer='adam')
    history = mass_model.fit(inputNN,ditaumass,batch_size,epochs, validation_split = 0.3,verbose = 2)
    mass_score = mass_model.evaluate(test_inputNN,test_ditaumass,batch_size,verbose=0)
    ditaumass_nn = mass_model.predict(test_inputNN,batch_size,verbose=0)
    ditaumass_nn_overfit = mass_model.predict(overfit_inputNN,batch_size,verbose=0)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    for i in range(len(ditaumass_nn)):
        return_list_nn.append([float(ditaumass_nn[i]),float(ditaumass_nn_overfit[i])])
    for j in range(epochs):
        return_list_nn_loss.append([loss[j],val_loss[j]])
    mass_model.summary()
    print "mass_model(",batch_size,epochs,")"
    print "loss (MSE):",mass_score
    print description_of_training
    end_nn = time.time()
    print "NN executon time:",(end_nn-start_nn)/3600,"h"

def reconstruct_mass(inputSVfit):
    vistau1_pt = inputSVfit[0]
    vistau1_eta = inputSVfit[1]
    vistau1_phi = inputSVfit[2]
    vistau1_mass = inputSVfit[3]
    vistau1_att = inputSVfit[4]
    vistau1_prongs = inputSVfit[5]
    vistau1_pi0 = inputSVfit[6] 
    vistau2_pt = inputSVfit[7]
    vistau2_eta = inputSVfit[8]
    vistau2_phi = inputSVfit[9]
    vistau2_mass = inputSVfit[10]
    vistau2_att = inputSVfit[11]
    vistau2_prongs = inputSVfit[12]
    vistau2_pi0 = inputSVfit[13]
    METx = inputSVfit[14]
    METy = inputSVfit[15]
    COVMET00 = inputSVfit[16]
    COVMET11 = inputSVfit[17]
    COVMET = inputSVfit[18]
    ditaumass = inputSVfit[19]
    ditauvismass = inputSVfit[20]
    # define MET covariance
    covMET = ROOT.TMatrixD(2, 2)
    covMET[0][0] = COVMET
    covMET[1][0] = 0.0
    covMET[0][1] = 0.0
    covMET[1][1] = COVMET
    vistau1_decaymode = int(5*(vistau1_prongs-1)+vistau1_pi0)
    vistau2_decaymode = int(5*(vistau2_prongs-1)+vistau2_pi0)
    # define lepton four vectors (pt,eta,phi,mass)
    measuredTauLeptons = ROOT.std.vector('svFitStandalone::MeasuredTauLepton')()
    if vistau1_att in (1,2) and vistau2_att in (1,2):
        k = 3.0
    if (vistau1_att in (1,2) and vistau2_att ==3) or (vistau1_att == 3 and vistau2_att in (1,2)):
        k = 4.0
    if vistau1_att == 3 and vistau2_att == 3:
        k = 5.0
    if vistau1_att == 1 :
        measuredTauLeptons.push_back(ROOT.svFitStandalone.MeasuredTauLepton(ROOT.svFitStandalone.kTauToElecDecay,vistau1_pt , vistau1_eta,vistau1_phi,vistau1_mass))
    if vistau1_att == 2:
        measuredTauLeptons.push_back(ROOT.svFitStandalone.MeasuredTauLepton(ROOT.svFitStandalone.kTauToMuDecay,vistau1_pt , vistau1_eta,vistau1_phi,vistau1_mass))
    if vistau1_att == 3:
        measuredTauLeptons.push_back(ROOT.svFitStandalone.MeasuredTauLepton(ROOT.svFitStandalone.kTauToHadDecay, vistau1_pt, vistau1_eta, vistau1_phi,vistau1_mass,vistau1_decaymode))
    if vistau2_att == 1:
        measuredTauLeptons.push_back(ROOT.svFitStandalone.MeasuredTauLepton(ROOT.svFitStandalone.kTauToElecDecay,vistau2_pt , vistau2_eta,vistau2_phi,vistau2_mass))
    if vistau2_att == 2:
        measuredTauLeptons.push_back(ROOT.svFitStandalone.MeasuredTauLepton(ROOT.svFitStandalone.kTauToMuDecay,vistau2_pt , vistau2_eta,vistau2_phi,vistau2_mass))
    if vistau2_att == 3:
        measuredTauLeptons.push_back(ROOT.svFitStandalone.MeasuredTauLepton(ROOT.svFitStandalone.kTauToHadDecay, vistau2_pt, vistau2_eta, vistau2_phi,vistau2_mass,vistau2_decaymode))
    verbosity = 0
    algo = ROOT.SVfitStandaloneAlgorithm(measuredTauLeptons, METx, METy, covMET, verbosity)
    algo.addLogM(False)
    inputFileName_visPtResolution = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/SVfit_standalone/data/svFitVisMassAndPtResolutionPDF.root"
    ROOT.TH1.AddDirectory(False)
    inputFile_visPtResolution = ROOT.TFile(inputFileName_visPtResolution)
    algo.shiftVisPt(True, inputFile_visPtResolution)
    algo.integrateMarkovChain()
    mass = algo.getMCQuantitiesAdapter().getMass()  # full mass of tau lepton pair in units of GeV
    inputFile_visPtResolution.Close()
    return [mass,ditaumass,ditauvismass,vistau1_att,vistau1_decaymode,vistau2_att,vistau2_decaymode,k]

def SVfit(processes,return_list_svfit):
    print "SVfit started"
    start_svfit = time.time()
    pool = multiprocessing.Pool(processes)
    ditaumass_svfit = pool.map(reconstruct_mass,inputSVfit)
    results_length = len(numpy.array(ditaumass_svfit,numpy.float64)[:,0])
    for i in range(results_length):
        return_list_svfit.append(ditaumass_svfit[i][:])
    end_svfit = time.time()        
    print "SVfit execution time:",(end_svfit-start_svfit)/3600 ,"h"


def fill_histditaumass():
    for g in test_ditaumass:
        histditaumass.Fill(g)

def fill_histditauvismass():
    for k in test_ditauvismass:
        histditauvismass.Fill(k)

####################################      control center         ################################################
#################################################################################################################

output_name = "ditau_mass_nnvssvfit_final_%s_decays_11" % (decaymode)
description_of_training = "Standardized input data 100 Epochs INPUT:vis tau1 (pt,eta,phi,mass)+vis tau2 (pt,eta,phi,mass)+METx+METy+p vis+vismass"
output_name_root = "%s.root" %(output_name)
rootfile = ROOT.TFile(output_name_root,"RECREATE")
output_file_name = "%s.txt" % (output_name)
output_file = open(output_file_name,'w')
sys.stdout = output_file

### type in shell for batch : qsub -pe smp 21 -N nnsvfit9 -q long.q batch_nnvssvfit_final_fast.sh ditau_mass_nnvssvfit_final_all_decays_9

SVfit_processes = 25
batch_size = 60
epochs = 100

print "dataset total length:",dataset_total_length
print "dataset length:",dataset_length
print "train length:",train_length
print "test length:",test_length


fill_histditaumass()
fill_histditauvismass()

histditaumass.Write()
histditauvismass.Write()

manager = multiprocessing.Manager()
return_list_nn = manager.list()
return_list_nn_loss = manager.list()
return_list_svfit = manager.list()

p1 = multiprocessing.Process(target = SVfit, args = (SVfit_processes,return_list_svfit,))
p2 = multiprocessing.Process(target = neural_network, args = (batch_size,epochs,output_name,return_list_nn,return_list_nn_loss,))
jobs = [p1,p2]
for t in jobs:
    t.start()
for p in jobs:
    p.join()

return_list_svfit = numpy.array(return_list_svfit,numpy.float64)
return_list_nn = numpy.array(return_list_nn,numpy.float64) 
return_list_nn_loss = numpy.array(return_list_nn_loss,numpy.float64)
########################            neural network histograms                 ################################

ditaumass_nn = return_list_nn[:,0]
ditaumass_nn_overfit = return_list_nn[:,1]
loss = return_list_nn_loss[:,0]
val_loss = return_list_nn_loss[:,1]
failed_division_nn = 0 

for j in ditaumass_nn:
    histditaumassnn.Fill(j)
for d in ditaumass_nn_overfit:
    histditaumassnnoverfit.Fill(d)
for k in range(100):
    if histditaumass.GetBinContent(k) != 0:
        ratio = histditaumassnn.GetBinContent(k)/histditaumass.GetBinContent(k)
        content_nn = histditaumassnn.GetBinContent(k)
        content_actual = histditaumass.GetBinContent(k)
        error_nn = numpy.sqrt(content_nn)
        error_actual = numpy.sqrt(content_actual)
        error_ratio = ratio*numpy.sqrt((error_actual/content_actual)**2+(error_nn/content_nn)**2)
        histditaumassnnratio.SetBinError(k,error_ratio)
        histditaumassnnratio.SetBinContent(k,ratio)
    elif histditaumassnn.GetBinContent(k) == 0 and histditaumass.GetBinContent(k) == 0:
        histditaumassnnratio.SetBinContent(k,1.0)
        histditaumassnnratio.SetBinContent(k,0)
    else:
        failed_division_nn +=1
histditaumassnn.Write()
histditaumassnnoverfit.Write()
histditaumassnnratio.Write()
print "failed divisions ratio neural network:",failed_division_nn

epochs_range = numpy.array([float(i) for i in range(1,epochs+1)])
loss_graph = ROOT.TGraph(epochs,epochs_range,numpy.array(loss))
loss_graph.SetTitle("model loss")
loss_graph.GetXaxis().SetTitle("epochs")
loss_graph.GetYaxis().SetTitle("loss")
loss_graph.SetMarkerColor(4)
loss_graph.SetMarkerSize(0.8)
loss_graph.SetMarkerStyle(21)
val_loss_graph = ROOT.TGraph(epochs,epochs_range,numpy.array(val_loss))
val_loss_graph.SetMarkerColor(2)
val_loss_graph.SetMarkerSize(0.8)
val_loss_graph.SetMarkerStyle(21)

#plots of the loss of train and test sample
canv1 = ROOT.TCanvas("loss di-tau mass")
loss_graph.Draw("AP")
val_loss_graph.Draw("P")
leg1 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg1.AddEntry(loss_graph,"loss on train sample","P")
leg1.AddEntry(val_loss_graph,"loss on test sample","P")
leg1.Draw()
output_plot_name = "%s_loss.png" %(output_name)
canv1.Write()
img1 = ROOT.TImage.Create()
img1.FromPad(canv1)
img1.WriteImage(output_plot_name)


############################             SVfit histograms                ################################

ditaumass_svfit_calc = return_list_svfit[:,0]
ditaumass_actual = return_list_svfit[:,1]
ditauvismass_actual = return_list_svfit[:,2]
for j in range(len(ditaumass_svfit_calc)):
    vistau1_att = return_list_svfit[j,3]
    vistau1_decaymode = return_list_svfit[j,4]
    vistau2_att = return_list_svfit[j,5]
    vistau2_decaymode = return_list_svfit[j,6]
    k = return_list_svfit[j,7]
    ditaumass_svfit_calc_loop = return_list_svfit[j,0]
    ditaumass_actual_loop = return_list_svfit[j,1]
    ditauvismass_actual_loop = return_list_svfit[j,2]
    histditaumasssvfitdelta.Fill(ditaumass_actual_loop-ditaumass_svfit_calc_loop)
    histditaumasssvfitres.Fill((ditaumass_actual_loop-ditaumass_svfit_calc_loop)/ditaumass_svfit_calc_loop)

for j in ditaumass_svfit_calc:
    histditaumasssvfit.Fill(j)

failed_division_svfit = 0
for k in range(0,100):
    if histditaumass.GetBinContent(k) != 0:
        ratio = histditaumasssvfit.GetBinContent(k)/histditaumass.GetBinContent(k)
        content_svfit = histditaumasssvfit.GetBinContent(k)
        content_actual = histditaumass.GetBinContent(k)
        error_svfit = numpy.sqrt(content_svfit)
        error_actual = numpy.sqrt(content_actual)
        error_ratio = ratio*numpy.sqrt((error_actual/content_actual)**2+(error_svfit/content_svfit)**2)
        histditaumasssvfitratio.SetBinError(k,error_ratio)
        histditaumasssvfitratio.SetBinContent(k,ratio)
    elif histditaumass.GetBinContent(k) == 0 and histditaumasssvfit.GetBinContent(k) == 0:
        histditaumasssvfitratio.SetBinContent(k,1.0)
    else:
        failed_division_svfit +=1

print "failed divisions ratio svfit:",failed_division_svfit
histditaumasssvfit.Write()
histditaumasssvfitratio.Write()

canv3 = ROOT.TCanvas("ditaumass SVfit delta")
histditaumasssvfitdelta.Draw()
output_hist_delta_name = "%s_delta.png" %(output_name)
canv3.Write()
img3 = ROOT.TImage.Create()
img3.FromPad(canv3)
img3.WriteImage(output_hist_delta_name)

canv4 = ROOT.TCanvas("ditaumass SVfit resolution")
histditaumasssvfitres.Draw()
output_hist_res_name = "%s_res.png" %(output_name)
canv4.Write()
img4 = ROOT.TImage.Create()
img4.FromPad(canv4)
img4.WriteImage(output_hist_res_name)

##############    putting neural network and SVfit in same histogram    ####################

canv2 = ROOT.TCanvas("di-tau mass using NN and SVfit")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.3,0.02)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum(),histditaumassnnoverfit.GetMaximum())
histditaumass.SetMaximum(max_bin*1.08)
histditaumass.Draw()
histditaumassnn.Draw("SAME")
histditaumassnnoverfit.Draw("SAME")
histditaumasssvfit.Draw("SAME")
histditauvismass.Draw("SAME")
leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumass,"actual mass","PL")
leg2.AddEntry(histditauvismass,"actual vismass","PL")
leg2.AddEntry(histditaumassnn,"neural network","PL")
leg2.AddEntry(histditaumassnnoverfit,"overfit-test","PL")
leg2.AddEntry(histditaumasssvfit,"SVfit","PL")
leg2.Draw()
pad2.cd()
histditaumassnnratio.Draw("P")
histditaumasssvfitratio.Draw("P SAME")
leg3 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg3.AddEntry(histditaumassnnratio,"neural network","P")
leg3.AddEntry(histditaumasssvfitratio,"SVfit","P")
leg3.Draw()
unit_line = ROOT.TLine(0.0,1.0,100.0,1.0)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
canv2.Write()
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)

output_file.close()
rootfile.Close()

