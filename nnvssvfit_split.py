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

histCOV00 = ROOT.TH1D("COV00","histogram of COV00",100,-20,20)
#histCOV00.GetXaxis().SetTitle("recoMETpx-genMETpx")
#histCOV00.GetYaxis().SetTitle("number of occurence")
#histCOV00.SetStats(0)

histCOV11 = ROOT.TH1D("COV11","histogram of COV11",100,-20,20)
#histCOV11.GetXaxis().SetTitle("recoMETpy-genMETpy")
#histCOV11.GetYaxis().SetTitle("number of occurence")
#histCOV11.SetStats(0)


#dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_2e6.csv",delim_whitespace=False,header=None)
dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_2e4_over80.csv",delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values
dataset_length = len(dataset_ditaumass[:,0])

for i in range(0,dataset_length):
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
#COV00_width = COV00_fit.GetParameter(2)*2*numpy.sqrt(2*numpy.log(2))
COV00_width = COV00_fit.GetParameter(2)
histCOV11.Fit("gaus")
COV11_fit = histCOV11.GetFunction("gaus")
#COV11_width = COV11_fit.GetParameter(2)*2*numpy.sqrt(2*numpy.log(2))
COV11_width = COV11_fit.GetParameter(2)
COV_width_mean = (COV00_width+COV11_width)/2
METCOV = ROOT.TMath.Power(COV_width_mean,2.)

dataset_ditaumass = dataset_ditaumass[0:10000,:]
dataset_length = len(dataset_ditaumass[:,0])
train_length = int(round(dataset_length*0.7))
test_length = int(round(dataset_length*0.3))

inputNN = []
inputSVfit = []
ditaumass = []
inputSVfit_90GeV = []
count_90GeV = 0

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
    #inputNN.append([v_vismot.Px(),v_vismot.Py(),v_vismot.Pz(),v_vismot.E(),v_nu.Px(),v_nu.Py(),p_vis,vismass])
    #inputNN.append([v_vistau1.Px(),v_vistau1.Py(),v_vistau1.Pz(),v_vistau1.E(),v_vistau2.Px(),v_vistau2.Py(),v_vistau2.Pz(),v_vistau2.E(),v_nu.Px(),v_nu.Py(),pt_nu,p_vis,vismass])
    #inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
    inputNN.append([v_vistau1.Px(),v_vistau1.Py(),v_vistau1.Pz(),v_vistau1.E(),vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,v_vistau2.Px(),v_vistau2.Py(),v_vistau2.Pz(),v_vistau2.E(),vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),pt_nu,p_vis,vismass,pt_vis])
    ditaumass.append(ditaumass_value)
    if vistau1_pt == 0.0 or vistau2_pt == 0.0:
       continue
    else:
        if ditaumass_value > 86.0 and ditaumass_value < 96.0:
            count_90GeV += 1
            #inputSVfit_90GeV.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,v_nu.Px(),v_nu.Py(),METCOV00,METCOV11,ditaumass_value,ditauvismass_value])
            inputSVfit_90GeV.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
        #inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,v_nu.Px(),v_nu.Py(),METCOV00,METCOV11,ditaumass_value,ditauvismass_value])
        inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])


print "number of events between 86 GeV and 96 GeV",count_90GeV
inputNN = numpy.array(inputNN,numpy.float64)
ditaumass = numpy.array(ditaumass,numpy.float64)
#inputSVfit = numpy.array(inputSVfit,numpy.float64)
inputNN[:,:] = preprocessing.scale(inputNN[:,:])

train_inputNN = inputNN[0:train_length,:]
train_ditaumass = ditaumass[0:train_length]
test_inputNN = inputNN[train_length:dataset_length,:]
test_ditaumass = ditaumass[train_length:dataset_length]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
inputSVfit = inputSVfit[train_length:dataset_length][:]

#histogram of the haronic decaymodes
histhaddecaymodes = ROOT.TH1D("haddecaymodes","decaymodes of #tau",10,0,10)
histhaddecaymodes.GetYaxis().SetTitle("number of occurence")
histhaddecaymodes.SetStats(0)
histhaddecaymodes.SetLineColor(2)
histhaddecaymodes.GetXaxis().SetBinLabel(1, "#tau^{-} #rightarrow e^{-} #bar{#nu_{e}} #nu_{#tau}")
histhaddecaymodes.GetXaxis().SetBinLabel(2, "#tau^{-} #rightarrow #mu^{-} #bar{#nu_{#mu}} #nu_{#tau}")
histhaddecaymodes.GetXaxis().SetBinLabel(3,"#tau^{-} #rightarrow #pi^{-} #nu_{#tau}")
histhaddecaymodes.GetXaxis().SetBinLabel(4, "#tau^{-} #rightarrow #pi^{-} #pi^{0} #nu_{#tau}")
histhaddecaymodes.GetXaxis().SetBinLabel(5, "#tau^{-} #rightarrow #pi^{-} #pi^{0} #pi^{0} #nu_{#tau}")
histhaddecaymodes.GetXaxis().SetBinLabel(6,"#tau^{-} #rightarrow #pi^{-} #pi^{+} #pi^{-} #nu_{#tau}")
histhaddecaymodes.GetXaxis().SetBinLabel(7,"#tau^{-} #rightarrow #pi^{-} #pi^{+} #pi^{-} #pi^{0} #nu_{#tau}")
histhaddecaymodes.GetXaxis().SetBinLabel(8,"other decays")
histhaddecaymodes.GetXaxis().SetLabelSize(0.04)

#histogram of ditau mass using neural network and SVfit
#histditaumass = ROOT.TH1D("ditaumass","reconstructed di-#tau mass using neural network and SVfit",100,0,100)
histditaumass = ROOT.TH1D("ditaumass","reconstructed di-#tau mass using SVfit",100,0,200)
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleOffset(1.2)
histditaumass.SetLineColor(2)
histditaumass.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","sadsad",100,0,200)
histditauvismass.SetLineColor(4)
histditauvismass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassregall","reconstructed di-#tau mass using neural network",100,0,100)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",100,0,200)
histditaumasssvfit.SetLineColor(3)
histditaumasssvfit.SetStats(0)
histditaumassnnoverfit = ROOT.TH1D("ditaumassnnoverfit","dsfdsfg",100,0,100)
histditaumassnnoverfit.SetLineColor(3)
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
histditaumasssvfitratiocomp = ROOT.TH1D("ditaumasssvfitratiocomp","ratio of reconstructed di-#tau mass using SVfit over actual mass",30,0,3)
histditaumasssvfitratiocomp.GetXaxis().SetTitle("m_{#tau#tau}/m_{#tau#tau}^{true}")
histditaumasssvfitratiocomp.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitratiocomp.SetStats(0)
histditaumasssvfitratiocomplep = ROOT.TH1D("ditaumasssvfitratiocomplep","ratio of reconstructed di-#tau mass using SVfit over actual mass leptonic",30,0,3)
histditaumasssvfitratiocomplep.GetXaxis().SetTitle("m_{#tau#tau}/m_{#tau#tau}^{true}")
histditaumasssvfitratiocomplep.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitratiocomplep.SetStats(0)
histditaumasssvfitratiocompsemilep = ROOT.TH1D("ditaumasssvfitratiocompsemilep","ratio of reconstructed di-#tau mass using SVfit over actual mass semileptonic",30,0,3)
histditaumasssvfitratiocompsemilep.GetXaxis().SetTitle("m_{#tau#tau}/m_{#tau#tau}^{true}")
histditaumasssvfitratiocompsemilep.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitratiocompsemilep.SetStats(0)
histditaumasssvfitratiocomphad = ROOT.TH1D("ditaumasssvfitratiocomphad","ratio of reconstructed di-#tau mass using SVfit over actual mass hadronic",30,0,3)
histditaumasssvfitratiocomphad.GetXaxis().SetTitle("m_{#tau#tau}/m_{#tau#tau}^{true}")
histditaumasssvfitratiocomphad.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitratiocomphad.SetStats(0)


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
histditaumassnnratio.SetMarkerColor(4)
histditaumassnnratio.SetStats(0)
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and actual mass",100,0,200)
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
histditaumasssvfitratio.SetStats(0)
histditaumassnnoverfitratio = ROOT.TH1D("ditaumassregalloverfitratio","ratio between reconstruced and overfit mass",100,0,100)
histditaumassnnoverfitratio.SetMarkerStyle(7)
histditaumassnnoverfitratio.SetMarkerColor(2)
histditaumassnnoverfitratio.SetStats(0)

###############leptonic decays ########################
histditaumasslep = ROOT.TH1D("ditaumasslep","reconstructed di-#tau mass using SVfit leptonic decays",100,0,200)
histditaumasslep.GetXaxis().SetTitle("")
histditaumasslep.GetXaxis().SetLabelSize(0)
histditaumasslep.GetYaxis().SetTitle("number of occurence")
histditaumasslep.GetYaxis().SetTitleOffset(1.2)
histditaumasslep.SetLineColor(2)
histditaumasslep.SetStats(0)
histditauvismasslep = ROOT.TH1D("ditauvismasslep","sadsad",100,0,200)
histditauvismasslep.SetLineColor(4)
histditauvismasslep.SetStats(0)
histditaumasssvfitlep = ROOT.TH1D("ditaumasssvfitlep","di-#tau mass using SVfit",100,0,200)
histditaumasssvfitlep.SetLineColor(3)
histditaumasssvfitlep.SetStats(0)
histditaumasssvfitratiolep = ROOT.TH1D("ditaumasssvfitratiolep","ratio between svfit and actual mass",100,0,200)
histditaumasssvfitratiolep.SetTitle("")
histditaumasssvfitratiolep.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumasssvfitratiolep.GetXaxis().SetLabelSize(0.08)
histditaumasssvfitratiolep.GetXaxis().SetTitleSize(0.08)
histditaumasssvfitratiolep.GetYaxis().SetTitle("ratio")
histditaumasssvfitratiolep.GetYaxis().SetLabelSize(0.08)
histditaumasssvfitratiolep.GetYaxis().SetTitleSize(0.08)
histditaumasssvfitratiolep.GetYaxis().SetTitleOffset(0.5)
histditaumasssvfitratiolep.GetYaxis().SetNdivisions(504)
histditaumasssvfitratiolep.GetYaxis().CenterTitle()
histditaumasssvfitratiolep.GetYaxis().SetRangeUser(0.0,5.0)
histditaumasssvfitratiolep.SetMarkerStyle(7)
histditaumasssvfitratiolep.SetStats(0)

##################### semileptonic decays #############################
histditaumasssemilep = ROOT.TH1D("ditaumasssemilep","reconstructed di-#tau mass using SVfit semileptonic decays",100,0,200)
histditaumasssemilep.GetXaxis().SetTitle("")
histditaumasssemilep.GetXaxis().SetLabelSize(0)
histditaumasssemilep.GetYaxis().SetTitle("number of occurence")
histditaumasssemilep.GetYaxis().SetTitleOffset(1.2)
histditaumasssemilep.SetLineColor(2)
histditaumasssemilep.SetStats(0)
histditauvismasssemilep = ROOT.TH1D("ditauvismasssemilep","sadsad",100,0,200)
histditauvismasssemilep.SetLineColor(4)
histditauvismasssemilep.SetStats(0)
histditaumasssvfitsemilep = ROOT.TH1D("ditaumasssvfitsemilep","di-#tau mass using SVfit",100,0,200)
histditaumasssvfitsemilep.SetLineColor(3)
histditaumasssvfitsemilep.SetStats(0)
histditaumasssvfitratiosemilep = ROOT.TH1D("ditaumasssvfitratiosemilep","ratio between svfit and actual mass",100,0,200)
histditaumasssvfitratiosemilep.SetTitle("")
histditaumasssvfitratiosemilep.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumasssvfitratiosemilep.GetXaxis().SetLabelSize(0.08)
histditaumasssvfitratiosemilep.GetXaxis().SetTitleSize(0.08)
histditaumasssvfitratiosemilep.GetYaxis().SetTitle("ratio")
histditaumasssvfitratiosemilep.GetYaxis().SetLabelSize(0.08)
histditaumasssvfitratiosemilep.GetYaxis().SetTitleSize(0.08)
histditaumasssvfitratiosemilep.GetYaxis().SetTitleOffset(0.5)
histditaumasssvfitratiosemilep.GetYaxis().SetNdivisions(504)
histditaumasssvfitratiosemilep.GetYaxis().CenterTitle()
histditaumasssvfitratiosemilep.GetYaxis().SetRangeUser(0.0,5.0)
histditaumasssvfitratiosemilep.SetMarkerStyle(7)
histditaumasssvfitratiosemilep.SetStats(0)

############################### hadronic decays ###############################
histditaumasshad = ROOT.TH1D("ditaumasshad","reconstructed di-#tau mass using SVfit hadronic decays",100,0,200)
histditaumasshad.GetXaxis().SetTitle("")
histditaumasshad.GetXaxis().SetLabelSize(0)
histditaumasshad.GetYaxis().SetTitle("number of occurence")
histditaumasshad.GetYaxis().SetTitleOffset(1.2)
histditaumasshad.SetLineColor(2)
histditaumasshad.SetStats(0)
histditauvismasshad = ROOT.TH1D("ditauvismasshad","sadsad",100,0,200)
histditauvismasshad.SetLineColor(4)
histditauvismasshad.SetStats(0)
histditaumasssvfithad = ROOT.TH1D("ditaumasssvfithad","di-#tau mass using SVfit",100,0,200)
histditaumasssvfithad.SetLineColor(3)
histditaumasssvfithad.SetStats(0)
histditaumasssvfitratiohad = ROOT.TH1D("ditaumasssvfitratiohad","ratio between svfit and actual mass",100,0,200)
histditaumasssvfitratiohad.SetTitle("")
histditaumasssvfitratiohad.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumasssvfitratiohad.GetXaxis().SetLabelSize(0.08)
histditaumasssvfitratiohad.GetXaxis().SetTitleSize(0.08)
histditaumasssvfitratiohad.GetYaxis().SetTitle("ratio")
histditaumasssvfitratiohad.GetYaxis().SetLabelSize(0.08)
histditaumasssvfitratiohad.GetYaxis().SetTitleSize(0.08)
histditaumasssvfitratiohad.GetYaxis().SetTitleOffset(0.5)
histditaumasssvfitratiohad.GetYaxis().SetNdivisions(504)
histditaumasssvfitratiohad.GetYaxis().CenterTitle()
histditaumasssvfitratiohad.GetYaxis().SetRangeUser(0.0,5.0)
histditaumasssvfitratiohad.SetMarkerStyle(7)
histditaumasssvfitratiohad.SetStats(0)



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

"""
def SVfit(event_start,event_stop,output_name):
    print "SVFIT"
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
        sum_values = METx+METy

        # define MET covariance
        covMET = ROOT.TMatrixD(2, 2)
        covMET[0][0] = 787.352
        covMET[1][0] = -178.63
        covMET[0][1] = -178.63
        covMET[1][1] = 179.545
        k = 0
        vistau1_decaymode = int(5*(vistau1_prongs-1)+vistau1_pi0)
        vistau2_decaymode = int(5*(vistau2_prongs-1)+vistau2_pi0)
        # define lepton four vectors (pt,eta,phi,mass)
        measuredTauLeptons = ROOT.std.vector('svFitStandalone::MeasuredTauLepton')()
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
        if vistau1_att in (1,2) and vistau2_att in (1,2):
            k = 3.0
        if (vistau1_att in (1,2) and vistau2_att ==3) or (vistau1_att == 3 and vistau2_att in (1,2)):
            k = 4.0
        if vistau1_att == 3 and vistau2_att == 3:
            k = 5.0
        # define algorithm (set the debug level to 3 for testing)
        verbosity = 0
        algo = ROOT.SVfitStandaloneAlgorithm(measuredTauLeptons, METx, METy, covMET, verbosity)
        algo.addLogM(True, k)
        inputFileName_visPtResolution = "SVfit_standalone/data/svFitVisMassAndPtResolutionPDF.root"
        ROOT.TH1.AddDirectory(False)
        inputFile_visPtResolution = ROOT.TFile(inputFileName_visPtResolution)
        algo.shiftVisPt(True, inputFile_visPtResolution)
        algo.integrateMarkovChain()
        mass = algo.getMCQuantitiesAdapter().getMass()  # full mass of tau lepton pair in units of GeV
        transverseMass = algo.getMCQuantitiesAdapter().getTransverseMass()  # transverse mass of tau lepton pair in units of GeV
        inputFile_visPtResolution.Close()
        return mass

    
    def process_reconstruct_mass(process_number,split_length):
        ditaumass_split = []
        for g in range(split_length):
            ditaumass = reconstruct_mass(inputSVfit_split[process_number][g][0],inputSVfit_split[process_number][g][1],inputSVfit_split[process_number][g][2],inputSVfit_split[process_number][g][3],inputSVfit_split[process_number][g][4],inputSVfit_split[process_number][g][5],inputSVfit_split[process_number][g][6],inputSVfit_split[process_number][g][7],inputSVfit_split[process_number][g][8],inputSVfit_split[process_number][g][9],inputSVfit_split[process_number][g][10],inputSVfit_split[process_number][g][11],inputSVfit_split[process_number][g][12],inputSVfit_split[process_number][g][13],inputSVfit_split[process_number][g][14],inputSVfit_split[process_number][g][15])
            ditaumass_split.append(ditaumass)
        #q.put(ditaumass_split)
        result = numpy.array(ditaumass_split)
        for i in range(len(ditaumass_split)):            
            ditaumass_svfit.append(ditaumass_split[i])
        print "ditaumass split:",ditaumass_split

    def multi_run_wrapper(args):
        return process_reconstruct_mass(*args)
    
    nProcesses = 8
    inputSVfit_split = numpy.array_split(inputSVfit[0:100][:], nProcesses)
    q = multiprocessing.Queue()
    ditaumass_svfit = []
    jobs = []

    def easy_function(input1):
        return input1[0]+input1[1]+input1[2]
    
    input_list = []
    for i in range(100):
        input_list.append([random.random(),random.random(),random.random()])


    pool = multiprocessing.Pool(processes=50)
    result = pool.map(reconstruct_mass,inputSVfit[0:100][:])
    #result = pool.map(easy_function,input_list)
    print result

    print pool.map(process_reconstruct_mass,
    for j in range(len(inputSVfit_split)):
        p = multiprocessing.Process(target=process_reconstruct_mass, args=(j,len(inputSVfit_split[j]),results))
        print p
        jobs.append(p)
        p.start()
        #ditaumass_loop = q.get()
        #print ditaumass_loop
        #for k in range(len(ditaumass_loop)):
        #    ditaumass_svfit.append(ditaumass_loop[k])
    for job in jobs:
        job.join()
    print results[:]
    print ditaumass_svfit
 
    time_per_event = 0
    for j in range(event_start,event_stop):
        if inputSVfit[j,0] == 0 or inputSVfit[j,7] == 0:
            continue
        else:
            start = time.time()
            ditaumass_svfit = reconstruct_mass(inputSVfit[j,0],inputSVfit[j,1],inputSVfit[j,2],inputSVfit[j,3],inputSVfit[j,4],inputSVfit[j,5],inputSVfit[j,6],inputSVfit[j,7],inputSVfit[j,8],inputSVfit[j,9],inputSVfit[j,10],inputSVfit[j,11],inputSVfit[j,12],inputSVfit[j,13],inputSVfit[j,14],inputSVfit[j,15])
            histditaumasssvfit.Fill(ditaumass_svfit)
            end = time.time()
            time_per_event += end-start
   
    failed_division = 0
    for k in range(0,100):
        if histditaumass.GetBinContent(k) != 0:
            histditaumasssvfitratio.SetBinContent(k,histditaumasssvfit.GetBinContent(k)/histditaumass.GetBinContent(k))
        elif histditaumass.GetBinContent(k) == 0 and histditaumasssvfit.GetBinContent(k) == 0:
            histditaumasssvfitratio.SetBinContent(k,1.0)
        else:
            failed_division +=1
    print "number of failed divisions:",failed_division
    print "SVfit execution time per event:", time_per_event/(event_stop-event_start),"s"
"""

def neural_network(batch_size,epochs,learning_rate,decay_rate,output_name):
    print "NEURAL NETWORK"
    mass_model = Sequential()
    mass_model.add(Dense(200,input_dim=22,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))

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
    lrate = [LearningRateScheduler(step_decay)]
    adam = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = decay_rate)
    #adam = Adam(lr = 0.0, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,decay = 0.0)
    mass_model.compile(loss='mean_squared_error',optimizer=adam)

    history = mass_model.fit(inputNN,ditaumass,batch_size,epochs, validation_split = 0.3,verbose = 2)
    #history = mass_model.fit(inputNN,ditaumass,batch_size,epochs, validation_split = 0.3,callbacks = lrate,verbose = 2)
    test_inputNN_nn = test_inputNN[event_start:event_stop,:]
    overfit_inputNN_nn = overfit_inputNN[event_start:event_stop,:]
    mass_score = mass_model.evaluate(test_inputNN,test_ditaumass,batch_size,verbose=0)
    ditaumass_nn = mass_model.predict(test_inputNN_nn,batch_size,verbose=0)
    ditaumass_nn_overfit = mass_model.predict(overfit_inputNN_nn,batch_size,verbose=0)
   
    mass_model.summary()
    print "mass_model(",batch_size,epochs,")"
    print "loss (MSE):",mass_score
    print description_of_training
    failed_division = 0
    #preparing the histograms
    for j in ditaumass_nn:
        histditaumassnn.Fill(j)
    for d in ditaumass_nn_overfit:
        histditaumassnnoverfit.Fill(d)
    for k in range(0,100):
        if histditaumass.GetBinContent(k) != 0:
            histditaumassnnratio.SetBinContent(k,histditaumassnn.GetBinContent(k)/histditaumass.GetBinContent(k))
        elif histditaumassnn.GetBinContent(k) == 0 and histditaumass.GetBinContent(k) == 0:
            histditaumassnnratio.SetBinContent(k,1.0)
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
    #covMET[0][0] = COVMET00
    covMET[0][0] = COVMET
    covMET[1][0] = 0.0
    covMET[0][1] = 0.0
    covMET[1][1] = COVMET
    #covMET[1][1] = COVMET11
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
    # define algorithm (set the debug level to 3 for testing)
    verbosity = 0
    algo = ROOT.SVfitStandaloneAlgorithm(measuredTauLeptons, METx, METy, covMET, verbosity)
    #algo.addLogM(True, k)
    algo.addLogM(False)
    inputFileName_visPtResolution = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/SVfit_standalone/data/svFitVisMassAndPtResolutionPDF.root"
    ROOT.TH1.AddDirectory(False)
    inputFile_visPtResolution = ROOT.TFile(inputFileName_visPtResolution)
    algo.shiftVisPt(True, inputFile_visPtResolution)
    algo.integrateMarkovChain()
    mass = algo.getMCQuantitiesAdapter().getMass()  # full mass of tau lepton pair in units of GeV
    transverseMass = algo.getMCQuantitiesAdapter().getTransverseMass()  # transverse mass of tau lepton pair in units of GeV
    inputFile_visPtResolution.Close()
    return [mass,ditaumass,ditauvismass,vistau1_att,vistau1_decaymode,vistau2_att,vistau2_decaymode,k]



###################################################################################
output_name = "ditau_mass_svfit_try111"
#output_name = "ditau_mass_standardized100_try262"
#description_of_training = "Standardized input data 100 Epochs try 262 LR=0.001 INPUT:vis tau1 (px,py,pz,E,pt,eta,phi,mass)+vis tau2 (px,py,pz,E,pt,eta,phi,mass)+METx+METy+MET pt+p vis+vismass+pt vis"
output_file_name = "%s.txt" % (output_name)
output_file = open(output_file_name,'w')
sys.stdout = output_file

print "dataset length:",dataset_length
print "train length:",train_length
print "test length:",test_length
print "SVfit length:",len(inputSVfit[:][0])
#for g in test_ditaumass[0:1000]:
#        histditaumass.Fill(g)

#################             run SVfit          #################### #####
start_svfit = time.time()
pool = multiprocessing.Pool(processes=30)
ditaumass_svfit = pool.map(reconstruct_mass,inputSVfit)
ditaumass_svfit = numpy.array(ditaumass_svfit,numpy.float64)

end_svfit = time.time()
ditaumass_svfit_calc = ditaumass_svfit[:,0]
ditaumass_actual = ditaumass_svfit[:,1]
ditauvismass_actual = ditaumass_svfit[:,2]
for j in range(len(ditaumass_svfit_calc)):
    vistau1_att = ditaumass_svfit[j,3]
    vistau1_decaymode = ditaumass_svfit[j,4]
    vistau2_att = ditaumass_svfit[j,5]
    vistau2_decaymode = ditaumass_svfit[j,6]
    k = ditaumass_svfit[j,7]
    ditaumass_svfit_calc_loop = ditaumass_svfit[j,0]
    ditaumass_actual_loop = ditaumass_svfit[j,1]
    ditauvismass_actual_loop = ditaumass_svfit[j,2]
    histditaumasssvfitdelta.Fill(ditaumass_actual_loop-ditaumass_svfit_calc_loop)
    histditaumasssvfitres.Fill((ditaumass_actual_loop-ditaumass_svfit_calc_loop)/ditaumass_svfit_calc_loop)
    histditaumasssvfitratiocomp.Fill(ditaumass_svfit_calc_loop/ditaumass_actual_loop)
    if k == 3:
       histditaumasslep.Fill(ditaumass_actual_loop)
       histditauvismasslep.Fill(ditauvismass_actual_loop)
       histditaumasssvfitlep.Fill(ditaumass_svfit_calc_loop)
       histditaumasssvfitratiocomplep.Fill(ditaumass_svfit_calc_loop/ditaumass_actual_loop)
       for i in range(0,200):
           if histditaumasslep.GetBinContent(i) != 0:
               histditaumasssvfitratiolep.SetBinContent(i,histditaumasssvfitlep.GetBinContent(i)/histditaumasslep.GetBinContent(i))
           elif histditaumasslep.GetBinContent(i) == 0 and histditaumasssvfitlep.GetBinContent(i) == 0:
               histditaumasssvfitratiolep.SetBinContent(i,1.0)
    if k == 4:
       histditaumasssemilep.Fill(ditaumass_actual_loop)
       histditauvismasssemilep.Fill(ditauvismass_actual_loop)
       histditaumasssvfitsemilep.Fill(ditaumass_svfit_calc_loop)
       histditaumasssvfitratiocompsemilep.Fill(ditaumass_svfit_calc_loop/ditaumass_actual_loop)
       for i in range(0,200):
           if histditaumasssemilep.GetBinContent(i) != 0:
               histditaumasssvfitratiosemilep.SetBinContent(i,histditaumasssvfitsemilep.GetBinContent(i)/histditaumasssemilep.GetBinContent(i))
           elif histditaumasssemilep.GetBinContent(i) == 0 and histditaumasssvfitsemilep.GetBinContent(i) == 0:
               histditaumasssvfitratiosemilep.SetBinContent(i,1.0)
    if k == 5:
       histditaumasshad.Fill(ditaumass_actual_loop)
       histditauvismasshad.Fill(ditauvismass_actual_loop)
       histditaumasssvfithad.Fill(ditaumass_svfit_calc_loop)
       histditaumasssvfitratiocomphad.Fill(ditaumass_svfit_calc_loop/ditaumass_actual_loop)
       for i in range(0,200):
           if histditaumasshad.GetBinContent(i) != 0:
               histditaumasssvfitratiohad.SetBinContent(i,histditaumasssvfithad.GetBinContent(i)/histditaumasshad.GetBinContent(i))
           elif histditaumasshad.GetBinContent(i) == 0 and histditaumasssvfithad.GetBinContent(i) == 0:
               histditaumasssvfitratiohad.SetBinContent(i,1.0)


    if vistau1_att == 1:
       histhaddecaymodes.Fill(0)
    if vistau1_att == 2:
       histhaddecaymodes.Fill(1)
    if vistau1_att == 3:
       if vistau1_decaymode == 0:
          histhaddecaymodes.Fill(2)
       elif vistau1_decaymode == 1:
          histhaddecaymodes.Fill(3)
       elif vistau1_decaymode == 2:
          histhaddecaymodes.Fill(4)
       elif vistau1_decaymode == 10:
          histhaddecaymodes.Fill(5)
       elif vistau1_decaymode == 11:
          histhaddecaymodes.Fill(6)
       else:
          histhaddecaymodes.Fill(7)
    if vistau2_att == 1:
       histhaddecaymodes.Fill(0)
    if vistau2_att == 2:
       histhaddecaymodes.Fill(1)
    if vistau2_att == 3:
       if vistau2_decaymode == 0:
          histhaddecaymodes.Fill(2)
       elif vistau2_decaymode == 1:
          histhaddecaymodes.Fill(3)
       elif vistau2_decaymode == 2:
          histhaddecaymodes.Fill(4)
       elif vistau2_decaymode == 10:
          histhaddecaymodes.Fill(5)
       elif vistau2_decaymode == 11:
          histhaddecaymodes.Fill(6)
       else:
          histhaddecaymodes.Fill(7)

for j in ditaumass_svfit_calc:
    histditaumasssvfit.Fill(j)
for g in ditaumass_actual:
    histditaumass.Fill(g)
for h in ditauvismass_actual:
    histditauvismass.Fill(h)   
failed_division = 0
for k in range(0,200):
    if histditaumass.GetBinContent(k) != 0:
        histditaumasssvfitratio.SetBinContent(k,histditaumasssvfit.GetBinContent(k)/histditaumass.GetBinContent(k))
    elif histditaumass.GetBinContent(k) == 0 and histditaumasssvfit.GetBinContent(k) == 0:
        histditaumasssvfitratio.SetBinContent(k,1.0)
    else:
        failed_division +=1
print "SVfit execution time:",(end_svfit-start_svfit)/3600 ,"h"
"""
#### run neural network  #####
batch_size = 60
epochs = 100
learning_rate = 0.001
#decay_rate = 0.00012
decay_rate = 0.0

start_nn = time.time()
#neural_network(batch_size,epochs,learning_rate,decay_rate,output_name)
end_nn = time.time()

print "NN executon time:",(end_nn-start_nn)/3600,"h" 
"""

#histogram of di-tau mass using regression all decays
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
#histditaumassnn.Draw("SAME")
#histditaumassnnoverfit.Draw("SAME")
histditaumasssvfit.Draw("SAME")
histditauvismass.Draw("SAME")
leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumass,"actual mass","PL")
leg2.AddEntry(histditauvismass,"actual vismass","PL")
#leg2.AddEntry(histditaumassnn,"neural network","PL")
#leg2.AddEntry(histditaumassnnoverfit,"overfit-test","PL")
leg2.AddEntry(histditaumasssvfit,"SVfit","PL")
leg2.Draw()
pad2.cd()
#histditaumassnnratio.Draw("P")
histditaumasssvfitratio.Draw("P")
#histditaumasssvfitratio.Draw("P SAME")
leg3 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg3.AddEntry(histditaumassnnratio,"neural network","P")
leg3.AddEntry(histditaumasssvfitratio,"SVfit","P")
#leg3.Draw()
unit_line = ROOT.TLine(0.0,1.0,200.0,1.0)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)

canv3 = ROOT.TCanvas("hadronic decay modes")
histhaddecaymodes.Draw()
output_hist_modes_name = "%s_decaymodes.png" %(output_name)
img3 = ROOT.TImage.Create()
img3.FromPad(canv3)
img3.WriteImage(output_hist_modes_name)

canv4 = ROOT.TCanvas("ditau mass SVfit leptonic decays")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.3,0.02)
pad1.Draw()
pad2.Draw()
pad1.cd()
histditaumasslep.Draw()
histditaumasssvfitlep.Draw("SAME")
histditauvismasslep.Draw("SAME")
leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumasslep,"actual mass","PL")
leg2.AddEntry(histditauvismasslep,"actual vismass","PL")
leg2.AddEntry(histditaumasssvfitlep,"SVfit","PL")
leg2.Draw()
pad2.cd()
histditaumasssvfitratiolep.Draw("P")
unit_line = ROOT.TLine(0.0,1.0,200.0,1.0)
unit_line.Draw("SAME")
output_hist_lep_name = "%s_lep.png" %(output_name)
img4 = ROOT.TImage.Create()
img4.FromPad(canv4)
img4.WriteImage(output_hist_lep_name)

canv5 = ROOT.TCanvas("ditau mass SVfit semileptonic decays")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.3,0.02)
pad1.Draw()
pad2.Draw()
pad1.cd()
histditaumasssemilep.Draw()
histditaumasssvfitsemilep.Draw("SAME")
histditauvismasssemilep.Draw("SAME")
leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumasssemilep,"actual mass","PL")
leg2.AddEntry(histditauvismasssemilep,"actual vismass","PL")
leg2.AddEntry(histditaumasssvfitsemilep,"SVfit","PL")
leg2.Draw()
pad2.cd()
histditaumasssvfitratiosemilep.Draw("P")
unit_line = ROOT.TLine(0.0,1.0,200.0,1.0)
unit_line.Draw("SAME")
output_hist_semilep_name = "%s_semilep.png" %(output_name)
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_semilep_name)

canv6 = ROOT.TCanvas("ditau mass SVfit hadronic decays")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.3,0.02)
pad1.Draw()
pad2.Draw()
pad1.cd()
histditaumasshad.Draw()
histditaumasssvfithad.Draw("SAME")
histditauvismasshad.Draw("SAME")
leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumasshad,"actual mass","PL")
leg2.AddEntry(histditauvismasshad,"actual vismass","PL")
leg2.AddEntry(histditaumasssvfithad,"SVfit","PL")
leg2.Draw()
pad2.cd()
histditaumasssvfitratiohad.Draw("P")
unit_line = ROOT.TLine(0.0,1.0,200.0,1.0)
unit_line.Draw("SAME")
output_hist_had_name = "%s_had.png" %(output_name)
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage(output_hist_had_name)

canv7 = ROOT.TCanvas("ditaumass SVfit delta")
histditaumasssvfitdelta.Draw()
output_hist_delta_name = "%s_delta.png" %(output_name)
img7 = ROOT.TImage.Create()
img7.FromPad(canv7)
img7.WriteImage(output_hist_delta_name)

canv8 = ROOT.TCanvas("ditaumass SVfit resolution")
histditaumasssvfitres.Draw()
output_hist_res_name = "%s_res.png" %(output_name)
img8 = ROOT.TImage.Create()
img8.FromPad(canv8)
img8.WriteImage(output_hist_res_name)

canv9 = ROOT.TCanvas("ditaumass SVfit ratio")
#canv9.SetLogx()
canv9.SetLogy()
histditaumasssvfitratiocomp.Draw()
output_hist_ratio_name = "%s_ratio.png" %(output_name)
img9 = ROOT.TImage.Create()
img9.FromPad(canv9)
img9.WriteImage(output_hist_ratio_name)

canv10 = ROOT.TCanvas("ditaumass SVfit ratio leptonic")
canv10.SetLogy()
histditaumasssvfitratiocomplep.Draw()
output_hist_ratiolep_name = "%s_ratiolep.png" %(output_name)
img10 = ROOT.TImage.Create()
img10.FromPad(canv10)
img10.WriteImage(output_hist_ratiolep_name)

canv11 = ROOT.TCanvas("ditaumass SVfit ratio semileptonic")
canv11.SetLogy()
histditaumasssvfitratiocompsemilep.Draw()
output_hist_ratiosemilep_name = "%s_ratiosemilep.png" %(output_name)
img11 = ROOT.TImage.Create()
img11.FromPad(canv11)
img11.WriteImage(output_hist_ratiosemilep_name)

canv12 = ROOT.TCanvas("ditaumass SVfit ratio hadronic")
canv12.SetLogy()
histditaumasssvfitratiocomphad.Draw()
output_hist_ratiohad_name = "%s_ratiohad.png" %(output_name)
img12 = ROOT.TImage.Create()
img12.FromPad(canv12)
img12.WriteImage(output_hist_ratiohad_name)

output_file.close()


#canv13 = ROOT.TCanvas("COV00")
#histCOV00.Draw()
#img13 = ROOT.TImage.Create()
#img13.FromPad(canv13)
#img13.WriteImage("COV00_histogram.png")

#canv14 = ROOT.TCanvas("COV11")
#histCOV11.Draw()
#img14 = ROOT.TImage.Create()
#img14.FromPad(canv14)
#img14.WriteImage("COV11_histogram.png")
