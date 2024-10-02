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
"""
list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_skim_correct40.csv"
#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_higgs_100GeV.csv"
#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_higgs_110GeV.csv"
#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_higgs_125GeV.csv"
#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_higgs_140GeV.csv"
#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_dy.csv"
dataframe_ditaumass = pandas.read_csv(list_name,delim_whitespace=False,header=None)
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
fixed_test_dataset_length = 100000
left_limit_hist = 0
right_limit_hist = 350
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
    if decaymode_count < fixed_dataset_length:
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
ditaumass = ditaumass[0:fixed_test_dataset_length]
ditauvismass = ditauvismass[0:fixed_test_dataset_length]
inputSVfit = inputSVfit[0:fixed_test_dataset_length,:]
"""



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



####################     get inputs     #########################
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small_fulllep.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small_fulllep.csv"
svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small_semilep.csv"
test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small_semilep.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small_fullhad.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small_fullhad.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_180GeV_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_180GeV_nostand_small.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_250GeV_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_250GeV_nostand_small.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_100GeV_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_100GeV_nostand_small.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_110GeV_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_110GeV_nostand_small.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_125GeV_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_125GeV_nostand_small.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_140GeV_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_140GeV_nostand_small.csv"
#svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_dy_nostand_small.csv"
#test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_dy_nostand_small.csv"

inputSVfit = numpy.array(pandas.read_csv(svfitinput_name, delim_whitespace=False,header=None))
ditaumass = pandas.read_csv(test_ditaumass_name, delim_whitespace=False,header=None).values
inputSVfit_length = len(ditaumass)
#####################################################################

output_name = "ditau_mass_svfit_small_wholerange_semilep_15"
output_file_name = "%s.txt" % (output_name)
output_file = open(output_file_name,'w')
sys.stdout = output_file
#################             run SVfit          #########################
nprocesses = 15
start_svfit = time.time()
pool = multiprocessing.Pool(processes = nprocesses)
ditaumass_svfit = pool.map(reconstruct_mass,inputSVfit)
ditaumass_svfit = numpy.array(ditaumass_svfit,numpy.float64)

end_svfit = time.time()

ditaumass_svfit_calc = ditaumass_svfit[:,0]
ditaumass_actual = ditaumass_svfit[:,1]
ditaumass_decaymode = ditaumass_svfit[:,7] 

print "SVfit execution time:",(end_svfit-start_svfit)/3600 ,"h"
print "SVfit execution time per event:",(end_svfit-start_svfit)/(inputSVfit_length/nprocesses),"s"


svfit_output_name = "%s.csv" %(output_name)
svfit_gen_name = "%s_gen.csv" %(output_name)
svfit_decaymode_name = "%s_decaymode.csv" %(output_name)
numpy.savetxt(svfit_output_name, ditaumass_svfit_calc, delimiter=",")
numpy.savetxt(svfit_gen_name, ditaumass_actual, delimiter=",")
numpy.savetxt(svfit_decaymode_name, ditaumass_decaymode, delimiter=",")
