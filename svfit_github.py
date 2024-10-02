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

#######################################################################################
########## choose the inputs for which SVfit should reconstruct the ditaumass #########
#options: "wholerange","wholerange_fulllep","wholerange_semilep","wholerange_fullhad","100GeV","110GeV","125GeV","140GeV","180GeV","250GeV","DY"
choose_input = "wholerange"

####################     get inputs     #########################
if choose_input == "wholerange":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small.csv"
if choose_input == "wholerange_fulllep":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small_fulllep.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small_fulllep.csv"
if choose_input == "wholerange_semilep":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small_semilep.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small_semilep.csv"
if choose_input == "wholerange_fullhad":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_nostand_small_fullhad.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_nostand_small_fullhad.csv"
if choose_input == "180GeV":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_180GeV_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_180GeV_nostand_small.csv"
if choose_input == "250GeV":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_250GeV_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_250GeV_nostand_small.csv"
if choose_input == "100GeV":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_100GeV_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_100GeV_nostand_small.csv"
if choose_input == "110GeV":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_110GeV_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_110GeV_nostand_small.csv"
if choose_input == "125GeV":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_125GeV_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_125GeV_nostand_small.csv"
if choose_input == "140GeV":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_140GeV_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_140GeV_nostand_small.csv"
if choose_input == "DY":
    svfitinput_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/svfitinput_test_dy_nostand_small.csv"
    test_ditaumass_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/nninput/nntarget_test_dy_nostand_small.csv"

inputSVfit = numpy.array(pandas.read_csv(svfitinput_name, delim_whitespace=False,header=None))
ditaumass = pandas.read_csv(test_ditaumass_name, delim_whitespace=False,header=None).values
inputSVfit_length = len(ditaumass)
#####################################################################

output_name = "ditau_mass_svfit_small_wholerange"
output_file_name = "%s.txt" % (output_name)
output_file = open(output_file_name,'w')
sys.stdout = output_file
#################             run SVfit          #########################
nprocesses = 10  #### choose number of cores  (at least 10 for the wholerange is recommended)
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
