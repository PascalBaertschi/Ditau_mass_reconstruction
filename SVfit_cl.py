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
ROOT.gSystem.Load("libSVfitStandaloneAlgorithm")

dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfit_1e5.csv",delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values
dataset_length = len(dataset_ditaumass[:,0])
train_length = int(round(dataset_length*0.7))
test_length = int(round(dataset_length*0.3))

inputNN = []
inputSVfit = []
ditaumass_list = []
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
    ditaumass = v_mot.M()
    vismass = v_vismot.M()
    p_vis = v_vismot.P()
    pt_vis = v_vismot.Pt()
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    inputNN.append([v_vismot.Px(),v_vismot.Py(),v_vismot.Pz(),v_vismot.E(),v_nu.Px(),v_nu.Py(),p_vis,vismass])
    ditaumass_list.append(ditaumass)
    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,v_nu.Px(),v_nu.Py()])
inputNN = numpy.array(inputNN,numpy.float64)
inputSVfit = numpy.array(inputSVfit,numpy.float64)
inputNN[:,0:8] = preprocessing.scale(inputNN[:,0:8])

train_inputNN = inputNN[0:train_length,0:8]
train_ditaumass = inputNN[0:train_length,0]
test_inputNN = inputNN[train_length:dataset_length,0:8]
test_ditaumass = inputNN[train_length:dataset_length,0]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
inputSVfit = inputSVfit[train_length:dataset_length,:]

histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",100,0,100)
histditaumasssvfit.GetXaxis().SetTitle("")
histditaumasssvfit.GetXaxis().SetLabelSize(0)
histditaumasssvfit.GetYaxis().SetTitle("number of occurence")
histditaumasssvfit.GetYaxis().SetTitleOffset(1.2)
histditaumasssvfit.SetLineColor(4)
histditaumasssvfit.SetStats(0)
histditaumass = ROOT.TH1D("ditaumass","jhh",100,0,100)
histditaumass.SetLineColor(2)
histditaumass.SetStats(0)
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
histditaumasssvfitratio.SetStats(0)

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


def SVfit(event_start,event_stop,output_name):
    def reconstruct_mass(vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,measuredMETx,measuredMETy):
        """load compiled SVfit_standalone library and check the output"""
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
        algo = ROOT.SVfitStandaloneAlgorithm(measuredTauLeptons, measuredMETx, measuredMETy, covMET, verbosity)
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

    time_per_event = 0
    for j in range(event_start,event_stop):
        if inputSVfit[j,0] == 0 or inputSVfit[j,7] == 0:
            continue
        else:
            start = time.time()
            ditaumass_svfit = reconstruct_mass(inputSVfit[j,0],inputSVfit[j,1],inputSVfit[j,2],inputSVfit[j,3],inputSVfit[j,4],inputSVfit[j,5],inputSVfit[j,6],inputSVfit[j,7],inputSVfit[j,8],inputSVfit[j,9],inputSVfit[j,10],inputSVfit[j,11],inputSVfit[j,12],inputSVfit[j,13],inputSVfit[j,14],inputSVfit[j,15])
            histditaumasssvfit.Fill(ditaumass_svfit)
            histditaumass.Fill(ditaumass_list[j])
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


    canv2 = ROOT.TCanvas("di-tau mass using SVfit")
    pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
    pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
    pad1.SetMargin(0.09,0.02,0.02,0.1)
    pad2.SetMargin(0.09,0.02,0.3,0.02)
    pad1.Draw()
    pad2.Draw()
    pad1.cd()
    max_bin = max(histditaumasssvfit.GetMaximum(),histditaumass.GetMaximum())
    histditaumasssvfit.SetMaximum(max_bin*1.08)
    histditaumasssvfit.Draw()
    histditaumass.Draw("SAME")
    leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg2.AddEntry(histditaumasssvfit,"reconstructed mass using SVfit","PL")
    leg2.AddEntry(histditaumass,"actual mass","PL")
    leg2.Draw()
    pad2.cd()
    histditaumasssvfitratio.Draw("P")
    unit_line = ROOT.TLine(0.0,1.0,100.0,1.0)
    unit_line.Draw("SAME")
    output_hist_name = "%s.png" %(output_name)
    img2 = ROOT.TImage.Create()
    img2.FromPad(canv2)
    img2.WriteImage(output_hist_name)



####run SVfit here###
output_name = "ditau_mass_svfit_500"
output_file_name = "%s.txt" % (output_name)
output_file = open(output_file_name,'w')
sys.stdout = output_file
event_start = 0
event_stop = 500

SVfit(event_start,event_stop,output_name)

output_file.close()
