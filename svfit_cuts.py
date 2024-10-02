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




#list_name_normal = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_tight.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_tight_morelowmass.csv"
#list_name_normal = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_notsotight.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_notsotight_morelowmass.csv"
list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_skim_correct40.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_correct29_nocut.csv"
#dataframe_ditaumass_normal = pandas.read_csv(list_name_normal,delim_whitespace=False,header=None)
#dataframe_ditaumass_skim = pandas.read_csv(list_name_skim,delim_whitespace=False,header=None)
dataframe_ditaumass = pandas.read_csv(list_name_skim,delim_whitespace=False,header=None)
#dataframe_ditaumass = dataframe_ditaumass_normal.append(dataframe_ditaumass_skim)
dataframe_ditaumass_shuffled = dataframe_ditaumass.sample(frac=1,random_state =1337)
dataset_ditaumass = dataframe_ditaumass_shuffled.values
dataset_total_length = len(dataset_ditaumass[:,0])

##order of dataset:  (v_tau1.Pt(),v_tau1.Eta(),v_tau1.Phi,v_tau1.M(),v_vistau1.Pt(),v_vistau1.Eta(),v_vistau1.Phi(),v_vistau1.M(),vistau1_att,vistau1_prongs,vistau1_pi0,v_tau2.Pt(),v_tau2.Eta(),v_tau2.Phi(),v_tau2.M(),v_vistau2.Pt(),v_vistau2.Eta(),v_vistau2.Phi(),v_vistau2.M(),vistau2_att,vistau2_prongs,vistau2_pi0,v_nu_mot.Pt(),v_nu_mot.Eta(),v_nu_mot.Phi(),v_nu_mot.M(),genMissingET_MET,genMissingET_Phi,MissingET_MET,MissingET_Phi)
histCOV00 = ROOT.TH1D("COV00","histogram of COV00",100,-20,20)
histCOV11 = ROOT.TH1D("COV11","histogram of COV11",100,-20,20)

for i in range(0,dataset_total_length):
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

histditaumasstotal = ROOT.TH1D("ditaumasstotal","di-#tau mass",300,0,300)
histditaumasstotal.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumasstotal.GetYaxis().SetTitle("number of occurence")
histditaumasstotal.SetStats(0)

histditaupttotal = ROOT.TH1D("ditaupttotal","di-#tau pt",300,0,300)
histditaupttotal.GetXaxis().SetTitle("di-#tau pt [GeV]")
histditaupttotal.GetYaxis().SetTitle("number of occurence")
histditaupttotal.SetStats(0)

inputSVfit = []
ditaumass = []
ditauvismass = []
decaymode_count = 0
decaymode_count_fulllep = 0
decaymode_count_semilep = 0
decaymode_count_fullhad = 0
decaymode_count_all = 0
##### change number of input here ##########
fixed_dataset_length = 1910000
#fixed_dataset_length = 100000000000
fixed_test_length = 100000
############################################
fulllep_fixed_length = int(round(0.1226*fixed_dataset_length))
semilep_fixed_length = int(round(0.4553*fixed_dataset_length))
fullhad_fixed_length = int(round(0.422* fixed_dataset_length))

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
    genMETpx = genMissingET_MET*numpy.cos(genMissingET_Phi)
    genMETpy = genMissingET_MET*numpy.sin(genMissingET_Phi)
    METpx = MissingET_MET*numpy.cos(MissingET_Phi)
    METpy = MissingET_MET*numpy.sin(MissingET_Phi)
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
    ditaupt_value = v_mot.Pt()
    histditaumasstotal.Fill(ditaumass_value)
    histditaupttotal.Fill(ditaupt_value)
    vismass = v_vismot.M()
    p_vis = v_vismot.P()
    pt_vis = v_vismot.Pt()
    pt_nu = v_nu.Pt()
    pt = v_mot.Pt()
    nu_px = v_nu.Px()
    nu_py = v_nu.Py()
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    vistau1_decaymode = int(5*(vistau1_prongs-1)+vistau1_pi0)
    vistau2_decaymode = int(5*(vistau2_prongs-1)+vistau2_pi0)
    #print "nu px:",v_nu.Px(),"genMETpx:",genMETpx,"METpx:",METpx,"nu py:",v_nu.Py(),"genMETpy:",genMETpy,"METpy:",METpy
    #print "vistau1_pt:",vistau1_pt,"vistau1_eta:",vistau1_eta,"vistau1_phi:",vistau1_phi,"vistau1_mass:",vistau1_mass,"vistau1_att:",vistau1_att,"vistau1_prongs:",vistau1_prongs,"vistau1_pi0:",vistau1_pi0,"vistau2_pt:",vistau2_pt,"vistau2_eta:",vistau2_eta,"vistau2_phi:",vistau2_phi,"vistau2_mass:",vistau2_mass,"vistau2_att:",vistau2_att,"vistau2_prongs:",vistau2_prongs,"vistau2_pi0:",vistau2_pi0,"METpx:",METpx,"METpy:",METpy,"METCOV:",METCOV,"ditaumass_value:",ditaumass_value
    if decaymode_count < fixed_dataset_length and ditaumass_value > 80.0:
        #ditaumass.append(ditaumass_value)
        #ditauvismass.append(ditauvismass_value)
        #inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
        #decaymode_count += 1
        if vistau1_att in (1,2) and vistau2_att in (1,2) and decaymode_count_fulllep < fulllep_fixed_length:
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_fulllep += 1
            decaymode_count += 1
        elif vistau1_att in (1,2) and vistau2_att == 3 and decaymode_count_semilep < semilep_fixed_length:
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_semilep += 1
            decaymode_count += 1
        elif vistau1_att == 3 and vistau2_att in (1,2) and decaymode_count_semilep < semilep_fixed_length:
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_semilep += 1
            decaymode_count += 1
        elif vistau1_att == 3 and vistau2_att == 3 and decaymode_count_fullhad < fullhad_fixed_length:
            inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count_fullhad += 1 
            decaymode_count += 1

   
ditaumass = numpy.array(ditaumass,numpy.float64)
ditauvismass = numpy.array(ditauvismass,numpy.float64)
inputSVfit = numpy.array(inputSVfit,numpy.float64)
ditaumass = ditaumass[0:fixed_test_length]
ditauvismass = ditauvismass[0:fixed_test_length]
inputSVfit = inputSVfit[0:fixed_test_length,:]
dataset_length = len(inputSVfit[:,0])
#Defining how many Digits are shown for each axis labels
ROOT.TGaxis.SetMaxDigits(4)
#histogram of ditau mass using neural network and SVfit
histtitle = "reconstructed di-#tau mass using SVfit"
histditaumass = ROOT.TH1D("ditaumass",histtitle,70,0,350)
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleSize(0.045)
histditaumass.GetYaxis().SetTitleOffset(0.75)
histditaumass.GetYaxis().SetLabelSize(0.045)
histditaumass.SetLineColor(2)
histditaumass.SetLineWidth(3)
histditaumass.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","ditauvismass",70,0,350)
histditauvismass.SetLineColor(6)
histditauvismass.SetLineWidth(3)
histditauvismass.SetLineStyle(7)
histditauvismass.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",70,0,350)
histditaumasssvfit.SetLineColor(8)
histditaumasssvfit.SetLineWidth(3)
histditaumasssvfit.SetLineStyle(2)
histditaumasssvfit.SetStats(0)


####SVfit delta histogram
histditaumasssvfitdelta = ROOT.TH1D("ditaumasssvfitdelta","delta between di-#tau_{gen} mass and di-#tau mass using SVfit",100,-70,70)
histditaumasssvfitdelta.GetXaxis().SetTitle("#Deltam [GeV]")
histditaumasssvfitdelta.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitdelta.GetYaxis().SetTitleOffset(1.2)
histditaumasssvfitdelta.SetStats(0)
histditaumasssvfitcorr = ROOT.TH2D("ditaumasssvfitcorr","di-#tau_{gen} mass vs di-#tau mass",400,0,400,400,0,400)
histditaumasssvfitcorr.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumasssvfitcorr.GetYaxis().SetTitle("di-#tau mass [GeV]")
histditaumasssvfitcorr.SetStats(0)
histditaumasssvfitres = ROOT.TH1D("resolution","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres.GetXaxis().SetTitle("relative difference per event")
histditaumasssvfitres.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitres.GetYaxis().SetTitleOffset(1.4)
histditaumasssvfitres.SetLineWidth(3)
histditaumasssvfitcorrres = ROOT.TH2D("ditaumasssvfitcorrres","relative difference per event per di-#tau_{gen} mass",350,0,350,80,-1,1)
histditaumasssvfitcorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumasssvfitcorrres.GetYaxis().SetTitle("relative difference per event")
histditaumasssvfitcorrres.SetStats(0)

profditaumasssvfitcorrres = ROOT.TProfile("profditaumasssvfitcorrres","bias in the di-#tau mass reconstruction",350,0,350)
profditaumasssvfitcorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumasssvfitcorrres.GetYaxis().SetTitle("bias")
profditaumasssvfitcorrres.GetYaxis().SetTitleOffset(1.2)
profditaumasssvfitcorrres.SetStats(0)
profditaumasssvfitcorrres.SetMarkerStyle(7)

profditaumasssvfitcorrabsres = ROOT.TProfile("profditaumasssvfitcorrabsres","average absolute deviation of relative difference per event",350,0,350)
profditaumasssvfitcorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumasssvfitcorrabsres.GetYaxis().SetTitle("average absolute deviation")
profditaumasssvfitcorrabsres.GetYaxis().SetTitleOffset(1.2)
profditaumasssvfitcorrabsres.SetStats(0)
profditaumasssvfitcorrabsres.SetMarkerStyle(7)

#ratio histograms
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and gen ditaumass",70,0,350)
histditaumasssvfitratio.SetTitle("")
histditaumasssvfitratio.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
#histditaumasssvfitratio.GetXaxis().SetLabelSize(0.08)
#histditaumasssvfitratio.GetXaxis().SetTitleSize(0.08)
histditaumasssvfitratio.GetXaxis().SetLabelSize(0.115)
histditaumasssvfitratio.GetXaxis().SetTitleSize(0.115)
histditaumasssvfitratio.GetYaxis().SetTitle("ratio")
#histditaumasssvfitratio.GetYaxis().SetLabelSize(0.08)
#histditaumasssvfitratio.GetYaxis().SetTitleSize(0.08)
histditaumasssvfitratio.GetYaxis().SetLabelSize(0.115)
histditaumasssvfitratio.GetYaxis().SetTitleSize(0.115)
histditaumasssvfitratio.GetYaxis().SetTitleOffset(0.3)
histditaumasssvfitratio.GetYaxis().SetNdivisions(404)
histditaumasssvfitratio.GetYaxis().CenterTitle()
histditaumasssvfitratio.GetYaxis().SetRangeUser(0.0,2.0)
histditaumasssvfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(8)
histditaumasssvfitratio.SetStats(0)


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
    COVMET = inputSVfit[16]
    ditaumass = inputSVfit[17]
    ditauvismass = inputSVfit[18]
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
    #algo.addLogM(True, k)
    algo.addLogM(False)
    inputFileName_visPtResolution = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/SVfit_standalone/data/svFitVisMassAndPtResolutionPDF.root"
    ROOT.TH1.AddDirectory(False)
    inputFile_visPtResolution = ROOT.TFile(inputFileName_visPtResolution)
    algo.shiftVisPt(True, inputFile_visPtResolution)
    algo.integrateMarkovChain()
    mass = algo.getMCQuantitiesAdapter().getMass()  # full mass of tau lepton pair in units of GeV
    inputFile_visPtResolution.Close()
    return [mass,ditaumass,ditauvismass,vistau1_att,vistau1_decaymode,vistau2_att,vistau2_decaymode,k]

def fill_histditaumass():
    for g in ditaumass:
        histditaumass.Fill(g)

def fill_histditauvismass():
    for f in ditauvismass:
        histditauvismass.Fill(f)


###################################################################################
output_name = "ditau_mass_svfit_cuts_1e5_try9"
output_file_name = "%s.txt" % (output_name)
output_root_name = "%s.root" % (output_name)
rootfile = ROOT.TFile(output_root_name,"RECREATE")
output_file = open(output_file_name,'w')
sys.stdout = output_file

print "dataset length:",dataset_length

fill_histditaumass()
fill_histditauvismass()




#################             run SVfit          #########################
nprocesses = 10
start_svfit = time.time()
pool = multiprocessing.Pool(processes = nprocesses)
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
    res = (ditaumass_actual_loop-ditaumass_svfit_calc_loop)/ditaumass_actual_loop
    #print "actual mass:",ditaumass_actual_loop,"reconstructed mass:",ditaumass_svfit_calc_loop, "vistau1_att:",vistau1_att,"vistau1_DM:",vistau1_decaymode,"vistau2_att:",vistau2_att,"vistau2_DM:",vistau2_decaymode
    histditaumasssvfitdelta.Fill(ditaumass_actual_loop-ditaumass_svfit_calc_loop)
    histditaumasssvfitcorr.Fill(ditaumass_actual_loop,ditaumass_svfit_calc_loop)
    histditaumasssvfitres.Fill(res)
    histditaumasssvfitcorrres.Fill(ditaumass_actual_loop,res)
    profditaumasssvfitcorrres.Fill(ditaumass_actual_loop,res)
    profditaumasssvfitcorrabsres.Fill(ditaumass_actual_loop,abs(res))
for j in ditaumass_svfit_calc:
    histditaumasssvfit.Fill(j)  
failed_division = 0
for k in range(0,70):
    if histditaumass.GetBinContent(k) != 0:
        content_svfit = histditaumasssvfit.GetBinContent(k)
        content_actual = histditaumass.GetBinContent(k)
        ratio = content_svfit/content_actual
        error_svfit = numpy.sqrt(content_svfit)
        error_actual = numpy.sqrt(content_actual)
        error_ratio = ratio*numpy.sqrt((error_actual/content_actual)**2+(error_svfit/content_svfit)**2)
        histditaumasssvfitratio.SetBinError(k,error_ratio)
        histditaumasssvfitratio.SetBinContent(k,ratio)
    elif histditaumass.GetBinContent(k) == 0 and histditaumasssvfit.GetBinContent(k) == 0:
        histditaumasssvfitratio.SetBinContent(k,1.0)
    else:
        failed_division +=1


print "SVfit execution time:",(end_svfit-start_svfit)/3600 ,"h"
print "SVfit execution time per event:",((end_svfit-start_svfit)*nprocesses)/fixed_test_length,"s"
#histogram of di-tau mass using regression all decays
canv2 = ROOT.TCanvas("di-tau using SVfit")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.32,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.35,0.03)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumasssvfit.GetMaximum(),histditauvismass.GetMaximum())
histditaumass.SetMaximum(max_bin*1.3)
histditaumass.Draw()
histditaumasssvfit.Draw("SAME")
histditauvismass.Draw("SAME")
leg2 = ROOT.TLegend(0.1,0.68,0.35,0.9)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg2.AddEntry(histditauvismass,"di-#tau_{vis} mass","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
leg2.SetTextSize(0.05)
leg2.Draw()
pad2.cd()
histditaumasssvfitratio.Draw("P")
unit_line = ROOT.TLine(0.0,1.0,350.0,1.0)
unit_line.SetLineColor(2)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
histditaumass.Write()
histditaumasssvfit.Write()
histditauvismass.Write()
histditaumasssvfitratio.Write()
canv2.Write()
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)

canv3 = ROOT.TCanvas("ditaumass SVfit delta")
histditaumasssvfitdelta.Draw()
histditaumasssvfitdelta.Write()
output_hist_delta_name = "%s_delta.png" %(output_name)
canv3.Write()
img3 = ROOT.TImage.Create()
img3.FromPad(canv3)
img3.WriteImage(output_hist_delta_name)

canv4 = ROOT.TCanvas("ditaumass SVfit resolution")
histditaumasssvfitres.Draw()
histditaumasssvfitres.Write()
output_hist_res_name = "%s_res.png" %(output_name)
canv4.Write()
img4 = ROOT.TImage.Create()
img4.FromPad(canv4)
img4.WriteImage(output_hist_res_name)

canv5 = ROOT.TCanvas("ditaumass SVfit correlation")
histditaumasssvfitcorr.Draw()
histditaumasssvfitcorr.Write()
line = ROOT.TLine(0.0,0.0,400.0,400.0)
line.SetLineWidth(2)
line.Draw("SAME")
output_hist_corr_name = "%s_corr.png" %(output_name)
canv5.Write()
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_corr_name)

canv6 = ROOT.TCanvas("ditaumass SVfit correlation with resolution")
histditaumasssvfitcorrres.Draw()
histditaumasssvfitcorrres.Write()
output_hist_corrres_name = "%s_corrres.png" %(output_name)
canv6.Write()
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage(output_hist_corrres_name)

canv7 = ROOT.TCanvas("ditaumass profile resolution")
profditaumasssvfitcorrres.Draw()
profditaumasssvfitcorrres.Write()
output_profres_name = "%s_profres.png" %(output_name)
canv7.Write()
img7 = ROOT.TImage.Create()
img7.FromPad(canv7)
img7.WriteImage(output_profres_name)

canv8 = ROOT.TCanvas("ditaumass profile abs(resolution)")
profditaumasssvfitcorrabsres.Draw()
profditaumasssvfitcorrabsres.Write()
output_profabsres_name = "%s_profabsres.png" %(output_name)
canv8.Write()
img8 = ROOT.TImage.Create()
img8.FromPad(canv8)
img8.WriteImage(output_profabsres_name)
"""
canv5 = ROOT.TCanvas("ditaumass")
histditaumasstotal.Draw("C E0")
output_hist_ditaumasstotal = "%s_ditaumass_total.png" %(output_name)
canv5.Write()
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_ditaumasstotal)

canv6 = ROOT.TCanvas("ditaupt")
histditaupttotal.Draw("C E0")
output_hist_ditaupttotal = "%s_ditaupt_total.png" %(output_name)
canv6.Write()
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage(output_hist_ditaupttotal)
"""

output_file.close()
rootfile.Close()

