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


dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_withtaunu4Vec_2e6.csv",delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values
dataset_length = len(dataset_ditaumass[:,0])


histCOV00 = ROOT.TH1D("COV00","histogram of COV00",100,-20,20)
histCOV11 = ROOT.TH1D("COV11","histogram of COV11",100,-20,20)

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
COV00_width = COV00_fit.GetParameter(2)
histCOV11.Fit("gaus")
COV11_fit = histCOV11.GetFunction("gaus")
COV11_width = COV11_fit.GetParameter(2)
COV_width_mean = (COV00_width+COV11_width)/2
METCOV = ROOT.TMath.Power(COV_width_mean,2.)

#################  choose size and decaymode of used dataset ######################
decaymode = 'all'
#fixed_dataset_length = 791000
fixed_dataset_length = 500000
#fixed_dataset_length = 1000

#####################################################################
dataset_length = len(dataset_ditaumass[:,0])

histratiovisptvspt = ROOT.TH1D("ratiovisptvspt","ratio between di-#tau vis pt and pt",50,0,5)
histratiovisptvspt.GetXaxis().SetTitle("ratio di-#tau vis-pt/pt")
histratiovisptvspt.GetYaxis().SetTitle("number of occurence")
histratiovisptvspt.SetLineColor(2)
histratiovisptvspt.SetStats(0)
histratiovisptvsptlep = ROOT.TH1D("ratiovisptvsptlep","ratio between vis pt and pt lep",50,0,5)
histratiovisptvsptlep.SetLineColor(3)
histratiovisptvsptlep.SetStats(0)
histratiovisptvsptsemilep = ROOT.TH1D("ratiovisptvsptsemilep","ratio between vis pt and pt semilep",50,0,5)
histratiovisptvsptsemilep.SetLineColor(4)
histratiovisptvsptsemilep.SetStats(0)
histratiovisptvspthad = ROOT.TH1D("ratiovisptvspthad","ratio between vis pt and pt had",50,0,5)
histratiovisptvspthad.SetLineColor(6)
histratiovisptvspthad.SetStats(0)

histratiovismassvsmass = ROOT.TH1D("ratiovismassvsmass","ratio between di-#tau vis mass and mass",50,0,1.5)
histratiovismassvsmass.GetXaxis().SetTitle("ratio di-#tau vis-mass/mass")
histratiovismassvsmass.GetYaxis().SetTitle("number of occurence")
histratiovismassvsmass.SetLineColor(2)
histratiovismassvsmass.SetStats(0)
histratiovismassvsmasslep = ROOT.TH1D("ratiovismassvsmasslep","ratio between vis mass and mass lep",50,0,1.5)
histratiovismassvsmasslep.SetLineColor(3)
histratiovismassvsmasslep.SetStats(0)
histratiovismassvsmasssemilep = ROOT.TH1D("ratiovismassvsmasssemilep","ratio between vis mass and mass semilep",50,0,1.5)
histratiovismassvsmasssemilep.SetLineColor(4)
histratiovismassvsmasssemilep.SetStats(0)
histratiovismassvsmasshad = ROOT.TH1D("ratiovismassvsmasshad","ratio between vis mass and mass had",50,0,1.5)
histratiovismassvsmasshad.SetLineColor(6)
histratiovismassvsmasshad.SetStats(0)

histtauratiovisptvspt = ROOT.TH1D("tauratiovisptvspt","ratio between #tau vis-pt and pt",50,0,2)
histtauratiovisptvspt.GetXaxis().SetTitle("ratio #tau vis-pt/pt")
histtauratiovisptvspt.GetYaxis().SetTitle("number of occurence")
histtauratiovisptvspt.SetLineColor(2)
histtauratiovisptvspt.SetStats(0)
histtauratiovisptvsptlep = ROOT.TH1D("tauratiovisptvsptlep","ratio between vis-pt and pt leptonic decays",50,0,2)
histtauratiovisptvsptlep.SetLineColor(3)
histtauratiovisptvsptlep.SetStats(0)
histtauratiovisptvspthad = ROOT.TH1D("tauratiovisptvspthad","ratio between vis-pt and pt hadronic decays",50,0,2)
histtauratiovisptvspthad.SetLineColor(6)
histtauratiovisptvspthad.SetStats(0)

histcorrvismassvsmass = ROOT.TH2D("corrvismassvsmass","di-#tau correlation between visible mass and mass",100,0,100,100,0,100)
histcorrvismassvsmass.GetXaxis().SetTitle("mass [GeV]")
histcorrvismassvsmass.GetYaxis().SetTitle("vis mass [GeV]")
histcorrvismassvsmass.SetStats(0)
histcorrvismassvsmasslep = ROOT.TH2D("corrvismassvsmasslep","di-#tau correlation between visible mass and mass leptonic decays",100,0,100,100,0,100)
histcorrvismassvsmasslep.GetXaxis().SetTitle("mass [GeV]")
histcorrvismassvsmasslep.GetYaxis().SetTitle("vis mass [GeV]")
histcorrvismassvsmasslep.SetStats(0)
histcorrvismassvsmasssemilep = ROOT.TH2D("corrvismassvsmasssemilep","di-#tau correlation between visible mass and mass semileptonic decays",100,0,100,100,0,100)
histcorrvismassvsmasssemilep.GetXaxis().SetTitle("mass [GeV]")
histcorrvismassvsmasssemilep.GetYaxis().SetTitle("vis mass [GeV]")
histcorrvismassvsmasssemilep.SetStats(0)
histcorrvismassvsmasshad = ROOT.TH2D("corrvismassvsmasshad","di-#tau correlation between visible mass and mass hadronic decays",100,0,100,100,0,100)
histcorrvismassvsmasshad.GetXaxis().SetTitle("mass [GeV]")
histcorrvismassvsmasshad.GetYaxis().SetTitle("vis mass [GeV]")
histcorrvismassvsmasshad.SetStats(0)



histvismass = ROOT.TH1D("vismass","visible di-#tau mass",100,0,100)
histvismass.GetXaxis().SetTitle("di-#tau visible mass [GeV]")
histvismass.GetYaxis().SetTitle("number of occurence")
histvismass.SetLineColor(2)
histvismass.SetStats(0)
histvismasslep = ROOT.TH1D("vismasslep","visible di-#tau mass",100,0,100)
histvismasslep.SetLineColor(3)
histvismasslep.SetStats(0)
histvismasssemilep = ROOT.TH1D("vismasssemilep","visible di-#tau mass",100,0,100)
histvismasssemilep.SetLineColor(4)
histvismasssemilep.SetStats(0)
histvismasshad = ROOT.TH1D("vismasshad","visible di-#tau mass",100,0,100)
histvismasshad.SetLineColor(6)
histvismasshad.SetStats(0)

histvismass10 = ROOT.TH1D("vismass10","visible di-#tau mass range 0-50 GeV",100,0,50)
histvismass10.GetXaxis().SetTitle("di-#tau visible mass [GeV]")
histvismass10.GetYaxis().SetTitle("number of occurence")
histvismass10.SetLineColor(2)
histvismass10.SetStats(0)
histvismasslep10 = ROOT.TH1D("vismasslep10","visible di-#tau mass 0-50 GeV",100,0,50)
histvismasslep10.SetLineColor(3)
histvismasslep10.SetStats(0)
histvismasssemilep10 = ROOT.TH1D("vismasssemilep10","visible di-#tau mass 0-50 GeV",100,0,50)
histvismasssemilep10.SetLineColor(4)
histvismasssemilep10.SetStats(0)
histvismasshad10 = ROOT.TH1D("vismasshad10","visible di-#tau mass 0-50 GeV",100,0,50)
histvismasshad10.SetLineColor(6)
histvismasshad10.SetStats(0)

histvismass90 = ROOT.TH1D("vismass90","visible di-#tau mass range 50-100GeV",100,50,100)
histvismass90.GetXaxis().SetTitle("di-#tau visible mass [GeV]")
histvismass90.GetYaxis().SetTitle("number of occurence")
histvismass90.SetLineColor(2)
histvismass90.SetStats(0)
histvismasslep90 = ROOT.TH1D("vismasslep90","visible di-#tau mass range 50-100 GeV",100,50,100)
histvismasslep90.SetLineColor(3)
histvismasslep90.SetStats(0)
histvismasssemilep90 = ROOT.TH1D("vismasssemilep90","visible di-#tau mass range 50-100 GeV",100,50,100)
histvismasssemilep90.SetLineColor(4)
histvismasssemilep90.SetStats(0)
histvismasshad90 = ROOT.TH1D("vismasshad90","visible di-#tau mass 50-100 GeV",100,50,100)
histvismasshad90.SetLineColor(6)
histvismasshad90.SetStats(0)

histmass = ROOT.TH1D("mass","di-#tau mass",100,0,100)
histmass.GetXaxis().SetTitle("di-#tau mass [GeV]")
histmass.GetYaxis().SetTitle("number of occurence")
histmass.SetLineColor(2)
histmass.SetStats(0)
histmasslep = ROOT.TH1D("masslep","di-#tau mass",100,0,100)
histmasslep.SetLineColor(3)
histmasslep.SetStats(0)
histmasssemilep = ROOT.TH1D("masssemilep","di-#tau mass",100,0,100)
histmasssemilep.SetLineColor(4)
histmasssemilep.SetStats(0)
histmasshad = ROOT.TH1D("masshad","di-#tau mass",100,0,100)
histmasshad.SetLineColor(6)
histmasshad.SetStats(0)

histmass10 = ROOT.TH1D("mass10","di-#tau mass range 0-50 GeV",100,0,50)
histmass10.GetXaxis().SetTitle("di-#tau mass [GeV]")
histmass10.GetYaxis().SetTitle("number of occurence")
histmass10.SetLineColor(2)
histmass10.SetStats(0)
histmasslep10 = ROOT.TH1D("masslep10","di-#tau mass 0-50 GeV",100,0,50)
histmasslep10.SetLineColor(3)
histmasslep10.SetStats(0)
histmasssemilep10 = ROOT.TH1D("masssemilep10","di-#tau mass 0-50 GeV",100,0,50)
histmasssemilep10.SetLineColor(4)
histmasssemilep10.SetStats(0)
histmasshad10 = ROOT.TH1D("masshad10","di-#tau mass 0-50 GeV",100,0,50)
histmasshad10.SetLineColor(6)
histmasshad10.SetStats(0)

histmass90 = ROOT.TH1D("mass90","di-#tau mass range 50-100GeV",100,50,100)
histmass90.GetXaxis().SetTitle("di-#tau mass [GeV]")
histmass90.GetYaxis().SetTitle("number of occurence")
histmass90.SetLineColor(2)
histmass90.SetStats(0)
histmasslep90 = ROOT.TH1D("masslep90","di-#tau mass range 50-100 GeV",100,50,100)
histmasslep90.SetLineColor(3)
histmasslep90.SetStats(0)
histmasssemilep90 = ROOT.TH1D("masssemilep90","di-#tau mass range 50-100 GeV",100,50,100)
histmasssemilep90.SetLineColor(4)
histmasssemilep90.SetStats(0)
histmasshad90 = ROOT.TH1D("masshad90","di-#tau mass 50-100 GeV",100,50,100)
histmasshad90.SetLineColor(6)
histmasshad90.SetStats(0)


inputNN = []
inputSVfit = []
ditaumass = []
ditauvismass = []
decaymode_count = 0
decaymode_count_lep = 0
decaymode_count_semilep = 0
decaymode_count_had = 0

for i in range(0,dataset_length):
    vistau1_pt = dataset_ditaumass[i,0]
    vistau1_eta = dataset_ditaumass[i,1]
    vistau1_phi = dataset_ditaumass[i,2]
    vistau1_mass = dataset_ditaumass[i,3]
    vistau1_att = dataset_ditaumass[i,4]
    vistau1_prongs = dataset_ditaumass[i,5]
    vistau1_pi0 = dataset_ditaumass[i,6]
    vistau1nu_pt = dataset_ditaumass[i,7]
    vistau1nu_eta = dataset_ditaumass[i,8]
    vistau1nu_phi = dataset_ditaumass[i,9]
    vistau1nu_mass = dataset_ditaumass[i,10]
    vistau2_pt = dataset_ditaumass[i,11]
    vistau2_eta = dataset_ditaumass[i,12]
    vistau2_phi = dataset_ditaumass[i,13]
    vistau2_mass = dataset_ditaumass[i,14]
    vistau2_att = dataset_ditaumass[i,15]
    vistau2_prongs = dataset_ditaumass[i,16]
    vistau2_pi0 = dataset_ditaumass[i,17]
    vistau2nu_pt = dataset_ditaumass[i,18]
    vistau2nu_eta = dataset_ditaumass[i,19]
    vistau2nu_phi = dataset_ditaumass[i,20]
    vistau2nu_mass = dataset_ditaumass[i,21]
    nu_pt = dataset_ditaumass[i,22]
    nu_eta = dataset_ditaumass[i,23]
    nu_phi = dataset_ditaumass[i,24]
    nu_mass = dataset_ditaumass[i,25]
    genMissingET_MET = dataset_ditaumass[i,26]
    genMissingET_Phi = dataset_ditaumass[i,27]
    MissingET_MET = dataset_ditaumass[i,28]
    MissingET_Phi = dataset_ditaumass[i,29]
    
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
    pt = v_mot.Pt()
    pt_nu = v_nu.Pt()
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    
    v_vistau1_nu = ROOT.TLorentzVector()
    v_vistau1_nu.SetPtEtaPhiM(vistau1nu_pt,vistau1nu_eta,vistau1nu_phi,vistau1nu_mass)
    v_vistau2_nu = ROOT.TLorentzVector()
    v_vistau2_nu.SetPtEtaPhiM(vistau2nu_pt,vistau2nu_eta,vistau2nu_phi,vistau2nu_mass)
    v_tau1 = v_vistau1+v_vistau1_nu
    v_tau2 = v_vistau2+v_vistau2_nu
    pt_vistau1 = v_vistau1.Pt()
    pt_tau1 = v_tau1.Pt()
    pt_vistau2 = v_vistau2.Pt()
    pt_tau2 = v_tau2.Pt()
    if pt_tau1 != 0:
        histtauratiovisptvspt.Fill(pt_vistau1/pt_tau1)
        if vistau1_att in (1,2):
            histtauratiovisptvsptlep.Fill(pt_vistau1/pt_tau1)
            #if pt_vistau1/pt_tau1 > 1:
            #    print "vis pt:",pt_vistau1,"   pt:",pt_tau1
        if vistau1_att == 3:
            histtauratiovisptvspthad.Fill(pt_vistau1/pt_tau1)
            #if pt_vistau1/pt_tau1 > 1:
            #    print "vis pt:",pt_vistau1,"  pt:",pt_tau1
    if pt_tau2 != 0:
        histtauratiovisptvspt.Fill(pt_vistau2/pt_tau2)
        if vistau2_att in (1,2):
            histtauratiovisptvsptlep.Fill(pt_vistau2/pt_tau2)
            #if pt_vistau2/pt_tau2 > 1 :
            #    print "vis pt:",pt_vistau2,"   pt:",pt_tau2
        if vistau2_att == 3:
            histtauratiovisptvspthad.Fill(pt_vistau2/pt_tau2)
            #if pt_vistau2/pt_tau2 > 1:
            #    print "vis pt:",pt_vistau2,"   pt:",pt_tau2
    
    if decaymode_count < fixed_dataset_length:
        inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
        ditaumass.append(ditaumass_value)
        ditauvismass.append(ditauvismass_value)
        #decaymode_count += 1
        if vistau1_pt == 0.0 or vistau2_pt == 0.0:
            continue
        else:
            decaymode_count += 1
            histcorrvismassvsmass.Fill(ditaumass_value,vismass)
            histratiovisptvspt.Fill(pt_vis/pt)
            histratiovismassvsmass.Fill(vismass/ditaumass_value)
            histvismass.Fill(vismass)
            histvismass10.Fill(vismass)
            histvismass90.Fill(vismass)
            histmass.Fill(ditaumass_value)
            histmass10.Fill(ditaumass_value)
            histmass90.Fill(ditaumass_value)
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
    else:
        break
    if vistau1_att in (1,2) and vistau2_att in (1,2):
        if decaymode_count_lep < fixed_dataset_length:
            inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_lep += 1
            histratiovisptvsptlep.Fill(pt_vis/pt)
            histratiovismassvsmasslep.Fill(vismass/ditaumass_value)
            histcorrvismassvsmasslep.Fill(ditaumass_value,vismass)
            histvismasslep.Fill(vismass)
            histvismasslep10.Fill(vismass)
            histvismasslep90.Fill(vismass)
            histmasslep.Fill(ditaumass_value)
            histmasslep10.Fill(ditaumass_value)
            histmasslep90.Fill(ditaumass_value)
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
        #else:
        #    break
    if (vistau1_att in (1,2) and vistau2_att == 3) or (vistau1_att == 3 and vistau2_att in (1,2)): 
        if decaymode_count_semilep < fixed_dataset_length:
            inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            #decaymode_count
            decaymode_count_semilep += 1
            histcorrvismassvsmasssemilep.Fill(ditaumass_value,vismass)
            histratiovisptvsptsemilep.Fill(pt_vis/pt)
            histratiovismassvsmasssemilep.Fill(vismass/ditaumass_value)
            histvismasssemilep.Fill(vismass)
            histvismasssemilep10.Fill(vismass)
            histvismasssemilep90.Fill(vismass)
            histmasssemilep.Fill(ditaumass_value)
            histmasssemilep10.Fill(ditaumass_value)
            histmasssemilep90.Fill(ditaumass_value) 
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
        
    if (vistau1_att == 3 and vistau2_att == 3):
        if decaymode_count_had < fixed_dataset_length:
            inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_had += 1
            histcorrvismassvsmasshad.Fill(ditaumass_value,vismass)
            histratiovisptvspthad.Fill(pt_vis/pt)
            histratiovismassvsmasshad.Fill(vismass/ditaumass_value)
            histvismasshad.Fill(vismass)
            histvismasshad10.Fill(vismass)
            histvismasshad90.Fill(vismass)
            histmasshad.Fill(ditaumass_value)
            histmasshad10.Fill(ditaumass_value)
            histmasshad90.Fill(ditaumass_value)
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV00,METCOV11,METCOV,ditaumass_value,ditauvismass_value])
#############################################################################################

output_name = "ditau_mass_svfit_control"
output_root_name = "%s.root" % (output_name)
rootfile = ROOT.TFile(output_root_name,"RECREATE")

#############################################################################################


canv5 = ROOT.TCanvas("ratio between vis pt and pt")
max_bin5 = max(histratiovisptvspt.GetMaximum(),histratiovisptvsptlep.GetMaximum(),histratiovisptvsptsemilep.GetMaximum(),histratiovisptvspthad.GetMaximum())
histratiovisptvspt.SetMaximum(max_bin5 * 1.08)
histratiovisptvspt.Draw()
histratiovisptvsptlep.Draw("SAME")
histratiovisptvsptsemilep.Draw("SAME")
histratiovisptvspthad.Draw("SAME")
leg5 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg5.AddEntry(histratiovisptvspt,"all decays","PL")
leg5.AddEntry(histratiovisptvsptlep,"leptonic decays","PL")
leg5.AddEntry(histratiovisptvsptsemilep,"semileptonic decays","PL")
leg5.AddEntry(histratiovisptvspthad,"hadronic decays","PL")
leg5.Draw()
canv5.Write()
output_hist_ratio_pt_name = "%s_ratio_visptvspt.png" %(output_name)
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_ratio_pt_name)



canv6 = ROOT.TCanvas("ratio between vis mass and mass")
max_bin6 = max(histratiovismassvsmass.GetMaximum(),histratiovismassvsmasslep.GetMaximum(),histratiovismassvsmasssemilep.GetMaximum(),histratiovismassvsmasshad.GetMaximum())
histratiovismassvsmass.SetMaximum(max_bin6 * 1.08)
histratiovismassvsmass.Draw()
histratiovismassvsmasslep.Draw("SAME")
histratiovismassvsmasssemilep.Draw("SAME")
histratiovismassvsmasshad.Draw("SAME")
leg6 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg6.AddEntry(histratiovismassvsmass,"all decays","PL")
leg6.AddEntry(histratiovismassvsmasslep,"leptonic decays","PL")
leg6.AddEntry(histratiovismassvsmasssemilep,"semileptonic decays","PL")
leg6.AddEntry(histratiovismassvsmasshad,"hadronic decays","PL")
leg6.Draw()
canv6.Write()
output_hist_ratio_mass_name = "%s_ratio_vismassvsmass.png" %(output_name)
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage(output_hist_ratio_mass_name)

canv7 = ROOT.TCanvas("vis mass")
max_bin7 = max(histvismass.GetMaximum(),histvismasslep.GetMaximum(),histvismasssemilep.GetMaximum(),histvismasshad.GetMaximum())
histvismass.SetMaximum(max_bin7 * 1.08)
histvismass.Draw()
histvismasslep.Draw("SAME")
histvismasssemilep.Draw("SAME")
histvismasshad.Draw("SAME")
leg7 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg7.AddEntry(histvismass,"all decays","PL")
leg7.AddEntry(histvismasslep,"leptonic decays","PL")
leg7.AddEntry(histvismasssemilep,"semileptonic decays","PL")
leg7.AddEntry(histvismasshad,"hadronic decays","PL")
leg7.Draw()
canv7.Write()
output_hist_vis_mass_name = "%s_vismass.png" %(output_name)
img7 = ROOT.TImage.Create()
img7.FromPad(canv7)
img7.WriteImage(output_hist_vis_mass_name)

canv8 = ROOT.TCanvas("vis mass range 0-50")
max_bin8 = max(histvismass10.GetMaximum(),histvismasslep10.GetMaximum(),histvismasssemilep10.GetMaximum(),histvismasshad10.GetMaximum())
histvismass10.SetMaximum(max_bin8 * 1.08)
histvismass10.Draw()
histvismasslep10.Draw("SAME")
histvismasssemilep10.Draw("SAME")
histvismasshad10.Draw("SAME")
leg8 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg8.AddEntry(histvismass10,"all decays","PL")
leg8.AddEntry(histvismasslep10,"leptonic decays","PL")
leg8.AddEntry(histvismasssemilep10,"semileptonic decays","PL")
leg8.AddEntry(histvismasshad10,"hadronic decays","PL")
leg8.Draw()
canv8.Write()
output_hist_vis_mass_10_name = "%s_vismass10.png" %(output_name)
img8 = ROOT.TImage.Create()
img8.FromPad(canv8)
img8.WriteImage(output_hist_vis_mass_10_name)

canv9 = ROOT.TCanvas("vis mass range 50-100")
max_bin9 = max(histvismass90.GetMaximum(),histvismasslep90.GetMaximum(),histvismasssemilep90.GetMaximum(),histvismasshad90.GetMaximum())
histvismass90.SetMaximum(max_bin9 * 1.08)
histvismass90.Draw()
histvismasslep90.Draw("SAME")
histvismasssemilep90.Draw("SAME")
histvismasshad90.Draw("SAME")
leg9 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg9.AddEntry(histvismass90,"all decays","PL")
leg9.AddEntry(histvismasslep90,"leptonic decays","PL")
leg9.AddEntry(histvismasssemilep90,"semileptonic decays","PL")
leg9.AddEntry(histvismasshad90,"hadronic decays","PL")
leg9.Draw()
canv9.Write()
output_hist_vis_mass_90_name = "%s_vismass90.png" %(output_name)
img9 = ROOT.TImage.Create()
img9.FromPad(canv9)
img9.WriteImage(output_hist_vis_mass_90_name)


canv10 = ROOT.TCanvas("mass")
max_bin10 = max(histmass.GetMaximum(),histmasslep.GetMaximum(),histmasssemilep.GetMaximum(),histmasshad.GetMaximum())
histmass.SetMaximum(max_bin10 * 1.08)
histmass.Draw()
histmasslep.Draw("SAME")
histmasssemilep.Draw("SAME")
histmasshad.Draw("SAME")
leg10 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg10.AddEntry(histmass,"all decays","PL")
leg10.AddEntry(histmasslep,"leptonic decays","PL")
leg10.AddEntry(histmasssemilep,"semileptonic decays","PL")
leg10.AddEntry(histmasshad,"hadronic decays","PL")
leg10.Draw()
canv10.Write()
output_hist_mass_name = "%s_mass.png" %(output_name)
img10 = ROOT.TImage.Create()
img10.FromPad(canv10)
img10.WriteImage(output_hist_mass_name)

canv11 = ROOT.TCanvas("mass range 0-50")
max_bin11 = max(histmass10.GetMaximum(),histmasslep10.GetMaximum(),histmasssemilep10.GetMaximum(),histmasshad10.GetMaximum())
histmass10.SetMaximum(max_bin11 * 1.08)
histmass10.Draw()
histmasslep10.Draw("SAME")
histmasssemilep10.Draw("SAME")
histmasshad10.Draw("SAME")
leg11 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg11.AddEntry(histmass10,"all decays","PL")
leg11.AddEntry(histmasslep10,"leptonic decays","PL")
leg11.AddEntry(histmasssemilep10,"semileptonic decays","PL")
leg11.AddEntry(histmasshad10,"hadronic decays","PL")
leg11.Draw()
canv11.Write()
output_hist_mass_10_name = "%s_mass10.png" %(output_name)
img11 = ROOT.TImage.Create()
img11.FromPad(canv11)
img11.WriteImage(output_hist_mass_10_name)

canv12 = ROOT.TCanvas("mass range 50-100")
max_bin12 = max(histmass90.GetMaximum(),histmasslep90.GetMaximum(),histmasssemilep90.GetMaximum(),histmasshad90.GetMaximum())
histmass90.SetMaximum(max_bin12 * 1.08)
histmass90.Draw()
histmasslep90.Draw("SAME")
histmasssemilep90.Draw("SAME")
histmasshad90.Draw("SAME")
leg12 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg12.AddEntry(histmass90,"all decays","PL")
leg12.AddEntry(histmasslep90,"leptonic decays","PL")
leg12.AddEntry(histmasssemilep90,"semileptonic decays","PL")
leg12.AddEntry(histmasshad90,"hadronic decays","PL")
leg12.Draw()
canv12.Write()
output_hist_mass_90_name = "%s_mass90.png" %(output_name)
img12 = ROOT.TImage.Create()
img12.FromPad(canv12)
img12.WriteImage(output_hist_mass_90_name)

canv13 = ROOT.TCanvas("ratio between vis pt and pt of taus")
max_bin13 = max(histtauratiovisptvspt.GetMaximum(),histtauratiovisptvsptlep.GetMaximum(),histtauratiovisptvspthad.GetMaximum())
histtauratiovisptvspt.SetMaximum(max_bin13 * 1.08)
histtauratiovisptvspt.Draw()
histtauratiovisptvsptlep.Draw("SAME")
histtauratiovisptvspthad.Draw("SAME")
leg13 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg13.AddEntry(histtauratiovisptvspt,"all decays","PL")
leg13.AddEntry(histtauratiovisptvsptlep,"leptonic decays","PL")
leg13.AddEntry(histtauratiovisptvspthad,"hadronic decays","PL")
leg13.Draw()
canv13.Write()
output_hist_tauratio_pt_name = "%s_tauratio_visptvspt.png" %(output_name)
img13 = ROOT.TImage.Create()
img13.FromPad(canv13)
img13.WriteImage(output_hist_tauratio_pt_name)


canv14 = ROOT.TCanvas("correlation between mass and vismass")
histcorrvismassvsmass.Draw()
histcorrvismassvsmass.Write()
canv14.Write()
output_hist_corr_mass_name = "%s_corr_mass.png" %(output_name)
img14 = ROOT.TImage.Create()
img14.FromPad(canv14)
img14.WriteImage(output_hist_corr_mass_name)

canv15 = ROOT.TCanvas("correlation between mass and vismass leptonic")
histcorrvismassvsmasslep.Draw()
histcorrvismassvsmasslep.Write()
canv15.Write()
output_hist_corr_masslep_name = "%s_corr_mass_lep.png" %(output_name)
img15 = ROOT.TImage.Create()
img15.FromPad(canv15)
img15.WriteImage(output_hist_corr_masslep_name)

canv16 = ROOT.TCanvas("correlation between mass and vismass semileptonic")
histcorrvismassvsmasssemilep.Draw()
histcorrvismassvsmasssemilep.Write()
canv15.Write()
output_hist_corr_masssemilep_name = "%s_corr_mass_semilep.png" %(output_name)
img16 = ROOT.TImage.Create()
img16.FromPad(canv16)
img16.WriteImage(output_hist_corr_masssemilep_name)

canv17 = ROOT.TCanvas("correlation between mass and vismass hadronic")
histcorrvismassvsmasshad.Draw()
histcorrvismassvsmasshad.Write()
canv17.Write()
output_hist_corr_masshad_name = "%s_corr_mass_had.png" %(output_name)
img17 = ROOT.TImage.Create()
img17.FromPad(canv17)
img17.WriteImage(output_hist_corr_masshad_name)

rootfile.Close()
