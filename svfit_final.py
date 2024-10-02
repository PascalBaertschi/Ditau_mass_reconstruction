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



#################  choose size and decaymode of used dataset ######################

decaymode = 'all'
fixed_dataset_length = 5500000
#fixed_dataset_length = 10000

###################################################################################

#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_8e5_%s.csv" %(decaymode)
#list_name = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_2e4_over80.csv"
#dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_5.5e6.csv",delim_whitespace=False,header=None)
dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_withtaunu4Vec_2e6.csv",delim_whitespace=False,header=None)
#dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_recovistau_1e6.csv",delim_whitespace=False,header=None)
#dataframe_ditaumass = pandas.read_csv(list_name,delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values
dataset_total_length = len(dataset_ditaumass[:,0])

#test_dataset = [[-20.3299,89.2168,-2.91627,91.5532,3,1,0,0.634699,-22.0654,6.05439,22.9455,3,1,0,759.229,-49.2613,-49.2613,821.732,-101.342,438.959,-22.5347,216.027],[-64.465,18.4994,164.725,177.857,3,1,0,-23.8959,24.3142,56.816,66.2592,3,1,0,1429.9,-519.734,-519.734,1062.26,-210.569,155.81,530.884,79.6875],[-54.3293,170.847,3.62473,179.315,3,1,0,27.3296,154.704,2.24479,157.119,3,1,0,1305.3, -482.5,-482.5,1433.58,-14.5723, 498.67,24.9545,119.889],[-8.68873,-133.06,419.945,440.607,3,1,0,29.0069,-66.1812,346.259,353.719,3,1,0,818.401,215.996,215.996,1267.07,69.7151,-347.71,1479.26, 112.253],[118.444,143.009,82.0525,203.012,3,1,0,15.8387,37.739,48.8156,63.7177,3,1,0,723.542,531.828,531.828,1425.6,263.095,348.196,249.117,92.5621],[8.55113,181.485,108.813,211.779,3,1,0,-10.7205,25.2389,14.3622,30.9658,3,1,0,1050.63,669.109,669.109,2074.25,-0.0234645,359.16,207.649,46.3549],[-8.68873,-133.06,419.945,440.607,3,1,0,29.0069,-66.1812,346.259,353.719,3,1,0,818.401,215.996,215.996,1267.07,69.7151,-347.71,1479.26,112.253],[-96.1104,89.5302,-24.7207,133.656,3,1,0,9.07843,19.8314,-0.0988249,21.8111,2,0,0,1328.27,-645.756,-645.756,1064.41,-237.21, 263.21,-53.3486,125.529],[40.3531,-81.6577,30.1545,95.9462,3,1,0,32.2527,-70.7589,-32.1455,84.1451,1,0,0,2183.17,-484.032,-484.032,1394.5,150.611,-322.735,-42.276,128.396],[89.0827,-47.7335,196.924,221.348,3,1,1,57.999,-14.1073,50.9325,78.4667,2,0,0,820.407,-81.2062,-81.2062,848.967,349.242,-122.753,476.498,126.943],[50.9969,-31.3113,-2.34739,59.9011,3,3,0,203.782,-5.84073,-117.19,235.148,1,0,0,448.185,-5.85362,-5.85362,502.435,407.55,-50.2199,-191.084,113.067],[-171.524,25.6861,-46.1568,179.476,3,2,1,-7.91412,10.143,5.54836,14.0106,1,0,0,1587.02,-257.103,-257.103,1061.91,-282.313,122.841,-12.8014,166.26]]
"""
test_dataset = [[-73.5899,22.0477,100.678,126.646,3,1,1,-140.963,136.602,346.069,397.863,2,0,0,1422.79,-137.607,-137.607,2434.13,-176.413,111.533,110.881],[184.732,-131.764,80.2122,240.669,3,1,0,93.0326,-125.836,107.71,189.977,1,0,0,1799.82,108.987,108.987,705.034,121.949,-103.674,114.512],[-86.5832,-43.7965,36.6732,103.729,3,1,0,-24.639,2.72011,13.2525,28.1089,1,0,0,610.713,-7.0107,-7.0107,571.851,-282.623,4.3498,113.653],[19.9047,72.7913,-1.45401,75.4847,3,1,0,-29.4577,235.289,-113.304,262.805,1,0,0,525.773,-230.831,-230.831,1626.62,-27.6935,194.193,115.24],[122.7,94.1214,-180.869,237.966,3,1,0,29.1046,6.96307,-49.5298,57.877,3,1,0,720.9,31.9242,31.9242,747.849,134.566,113.697,53.1197],[-19.6892,136.309,126.733,187.163,3,1,0,13.1711,20.7383,20.8247,32.2228,3,1,0,365.285,-19.8308,-19.8308,612.731,38.6986,197.664,94.4009],[-53.5926,109.803,-271.405,297.642,3,1,0,6.50968,67.658,-195.306,206.796,3,1,0,550.405,-87.9811,-87.9811,811.605,-65.4599,189.051,104.864],[-52.8015,-189.741,174.138,262.899,3,1,0,10.984,-47.424,24.5841,54.536,3,1,0,1188.66,153.409,153.409,1485.41,14.8013,-186.552,122.171],[49.5342,-83.5754,-40.664,105.319,3,1,0,77.3466,-23.2817,-52.4598,96.317,3,1,0,987.755,-200.041,-200.041,1743.13,118.68,-212.289,132.879]]





test_dataset = numpy.array(test_dataset,numpy.float64)

inputSVfit_test_dataset = []
for i in range(len(test_dataset[:,0])):
    tau1_px = test_dataset[i,0]
    tau1_py = test_dataset[i,1]
    tau1_pz = test_dataset[i,2]
    tau1_E = test_dataset[i,3]
    tau1_att = test_dataset[i,4]
    tau1_prongs = test_dataset[i,5]
    tau1_pi0 = test_dataset[i,6]
    tau2_px = test_dataset[i,7]
    tau2_py = test_dataset[i,8]
    tau2_pz = test_dataset[i,9]
    tau2_E = test_dataset[i,10]
    tau2_att = test_dataset[i,11]
    tau2_prongs = test_dataset[i,12]
    tau2_pi0 = test_dataset[i,13]
    COV00 = test_dataset[i,14]
    COV01 = test_dataset[i,15]
    COV10 = test_dataset[i,16]
    COV11 = test_dataset[i,17]
    METx = test_dataset[i,18]
    METy = test_dataset[i,19]
    mass_value = test_dataset[i,20]
    v_tau1 = ROOT.TLorentzVector(tau1_px,tau1_py,tau1_pz,tau1_E)
    v_tau2 = ROOT.TLorentzVector(tau2_px,tau2_py,tau2_pz,tau2_E)
    tau1_pt = v_tau1.Pt()
    tau1_eta = v_tau1.Eta()
    tau1_phi = v_tau1.Phi()
    tau1_mass = v_tau1.M()
    tau2_pt = v_tau2.Pt()
    tau2_eta = v_tau2.Eta()
    tau2_phi = v_tau2.Phi()
    tau2_mass = v_tau2.M()
    inputSVfit_test_dataset.append([tau1_pt,tau1_eta,tau1_phi,tau1_mass,tau1_att,tau1_prongs,tau1_pi0,tau2_pt,tau2_eta,tau2_phi,tau2_mass,tau2_att,tau2_prongs,tau2_pi0,COV00,COV01,COV10,COV11,METx,METy,mass_value])


inputSVfit_test_dataset = numpy.array(inputSVfit_test_dataset,numpy.float64)

"""

histCOV00 = ROOT.TH1D("COV00","histogram of COV00",100,-20,20)
histCOV11 = ROOT.TH1D("COV11","histogram of COV11",100,-20,20)

for i in range(0,dataset_total_length):
    #genMissingET_MET = dataset_ditaumass[i,18]
    #genMissingET_Phi = dataset_ditaumass[i,19]
    #MissingET_MET = dataset_ditaumass[i,20]
    #MissingET_Phi = dataset_ditaumass[i,21]
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

histditaumasstotal = ROOT.TH1D("ditaumasstotal","di-#tau mass",150,0,300)
histditaumasstotal.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumasstotal.GetYaxis().SetTitle("number of occurence")
histditaumasstotal.SetStats(0)

histditaupttotal = ROOT.TH1D("ditaupttotal","di-#tau pt",150,0,300)
histditaupttotal.GetXaxis().SetTitle("di-#tau pt [GeV]")
histditaupttotal.GetYaxis().SetTitle("number of occurence")
histditaupttotal.SetStats(0)

inputSVfit = []
ditaumass = []
ditauvismass = []
decaymode_count = 0
decaymode_e_count = 0
decaymode_mu_count = 0
decaymode_had_count = 0
decaymode_fulllep_count = 0
decaymode_semilep_count = 0
decaymode_fullhad_count = 0
decaymode_all_count = 0
fulllep_fixed_length = int(round(0.1226*fixed_dataset_length))
semilep_fixed_length = int(round(0.4553*fixed_dataset_length))
fullhad_fixed_length = int(round(0.422* fixed_dataset_length))
 
for i in range(0,dataset_total_length):
    vistau1_pt = dataset_ditaumass[i,0]
    vistau1_eta = dataset_ditaumass[i,1]
    vistau1_phi = dataset_ditaumass[i,2]
    vistau1_mass = dataset_ditaumass[i,3]
    #vistau1_reco_pt = dataset_ditaumass[i,4]
    #vistau1_reco_eta = dataset_ditaumass[i,5]
    #vistau1_reco_phi = dataset_ditaumass[i,6]
    #vistau1_reco_mass = dataset_ditaumass[i,7]
    #vistau1_att = dataset_ditaumass[i,8]
    #vistau1_prongs = dataset_ditaumass[i,9]
    #vistau1_pi0 = dataset_ditaumass[i,10]
    #vistau2_pt = dataset_ditaumass[i,11]
    #vistau2_eta = dataset_ditaumass[i,12]
    #vistau2_phi = dataset_ditaumass[i,13]
    #vistau2_mass = dataset_ditaumass[i,14]
    #vistau2_reco_pt = dataset_ditaumass[i,15]
    #vistau2_reco_eta = dataset_ditaumass[i,16]
    #vistau2_reco_phi = dataset_ditaumass[i,17]
    #vistau2_reco_mass = dataset_ditaumass[i,18]
    #vistau2_att = dataset_ditaumass[i,19]
    #vistau2_prongs = dataset_ditaumass[i,20]
    #vistau2_pi0 = dataset_ditaumass[i,21]
    #nu_pt = dataset_ditaumass[i,22]
    #nu_eta = dataset_ditaumass[i,23]
    #nu_phi = dataset_ditaumass[i,24]
    #nu_mass = dataset_ditaumass[i,25]
    #genMissingET_MET = dataset_ditaumass[i,26]
    #genMissingET_Phi = dataset_ditaumass[i,27]
    #MissingET_MET = dataset_ditaumass[i,28]
    #MissingET_Phi = dataset_ditaumass[i,29]
    
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

    #vistau2_pt = dataset_ditaumass[i,7]
    #vistau2_eta = dataset_ditaumass[i,8]
    #vistau2_phi = dataset_ditaumass[i,9]
    #vistau2_mass = dataset_ditaumass[i,10]
    #vistau2_att = dataset_ditaumass[i,11]
    #vistau2_prongs = dataset_ditaumass[i,12]
    #vistau2_pi0 = dataset_ditaumass[i,13]
    #nu_pt = dataset_ditaumass[i,14]
    #nu_eta = dataset_ditaumass[i,15]
    #nu_phi = dataset_ditaumass[i,16]
    #nu_mass = dataset_ditaumass[i,17]
    #genMissingET_MET = dataset_ditaumass[i,18]
    #genMissingET_Phi = dataset_ditaumass[i,19]
    #MissingET_MET = dataset_ditaumass[i,20]
    #MissingET_Phi = dataset_ditaumass[i,21]
    genMETpx = genMissingET_MET*numpy.cos(genMissingET_Phi)
    genMETpy = genMissingET_MET*numpy.sin(genMissingET_Phi)
    METpx = MissingET_MET*numpy.cos(MissingET_Phi)
    METpy = MissingET_MET*numpy.sin(MissingET_Phi)
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
    #reco inputSVfit filling
    """
    if vistau1_reco_pt != 0 and vistau2_reco_pt !=0:
        decaymode_all_count += 1
        if vistau1_att in (1,2) and vistau2_att in (1,2):
            if decaymode_fulllep_count < fulllep_fixed_length:
                ditaumass.append(ditaumass_value)
                ditauvismass.append(ditauvismass_value)
                inputSVfit.append([vistau1_reco_pt,vistau1_reco_eta,vistau1_reco_phi,vistau1_reco_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_reco_pt,vistau2_reco_eta,vistau2_reco_phi,vistau2_reco_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                #inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                decaymode_fulllep_count += 1
        elif (vistau1_att == 3 and vistau2_att in (1,2)) or (vistau1_att in (1,2) and vistau2_att == 3):
            if decaymode_semilep_count < semilep_fixed_length:
                ditaumass.append(ditaumass_value)
                ditauvismass.append(ditauvismass_value)
                inputSVfit.append([vistau1_reco_pt,vistau1_reco_eta,vistau1_reco_phi,vistau1_reco_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_reco_pt,vistau2_reco_eta,vistau2_reco_phi,vistau2_reco_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                #inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                decaymode_semilep_count += 1
        elif vistau1_att == 3 and vistau2_att == 3:
            if decaymode_fullhad_count < fullhad_fixed_length:
                ditaumass.append(ditaumass_value)
                ditauvismass.append(ditauvismass_value)
                inputSVfit.append([vistau1_reco_pt,vistau1_reco_eta,vistau1_reco_phi,vistau1_reco_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_reco_pt,vistau2_reco_eta,vistau2_reco_phi,vistau2_reco_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                #inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                decaymode_fullhad_count += 1
    if decaymode_fulllep_count == fulllep_fixed_length and decaymode_semilep_count == semilep_fixed_length and decaymode_fullhad_count == fullhad_fixed_length:
        break
    """
    if decaymode_count < fixed_dataset_length:
        if tau1_pt > 20 and tau2_pt > 20 and tau1_eta < 2.3 and tau2_eta < 2.3 and MissingET_MET > 20:
            if vistau1_att == 3 and vistau2_att == 3:
                if vistau1_pt > 45 and vistau2_pt > 45 and abs(vistau1_eta) < 2.1 and abs(vistau2_eta) < 2.1: 
                    ditaumass.append(ditaumass_value)
                    ditauvismass.append(ditauvismass_value)
                    decaymode_count += 1
                    decaymode_fullhad_count += 1
                    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                else:
                    continue
            elif vistau1_att in (1,2) and vistau2_att in (1,2):
                if vistau1_pt > vistau2_pt:
                    if vistau1_pt > 20 and vistau2_pt > 10 and abs(vistau1_eta)<2.4 and abs(vistau2_eta) < 2.4:
                        ditaumass.append(ditaumass_value)
                        ditauvismass.append(ditauvismass_value)
                        decaymode_count += 1
                        decaymode_fulllep_count += 1
                        inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                    else:
                        continue
                elif vistau2_pt > vistau1_pt:
                    if vistau1_pt > 10 and vistau2_pt > 20 and abs(vistau1_eta)<2.4 and abs(vistau2_eta) < 2.4:
                        ditaumass.append(ditaumass_value)
                        ditauvismass.append(ditauvismass_value)
                        decaymode_count += 1
                        decaymode_fulllep_count += 1
                        inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                    else:
                        continue
            elif vistau1_att in (1,2) and vistau2_att == 3:
                if vistau1_pt > 20 and abs(vistau1_eta) < 2.1 and vistau2_pt > 30 and abs(vistau2_eta) < 2.3:
                    ditaumass.append(ditaumass_value)
                    ditauvismass.append(ditauvismass_value)
                    decaymode_count += 1
                    decaymode_semilep_count += 1
                    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                else:
                    continue
            elif vistau2_att in (1,2) and vistau1_att == 3:
                if vistau2_pt > 20 and abs(vistau2_eta) < 2.1 and vistau1_pt > 30 and abs(vistau1_eta) < 2.3:
                    ditaumass.append(ditaumass_value)
                    ditauvismass.append(ditauvismass_value)
                    decaymode_count += 1
                    decaymode_semilep_count += 1
                    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
                else:
                    continue
        else:
            continue
        """
        #if vistau1_reco_pt != 0 and vistau2_reco_pt !=0 and (vistau1_att == 3 or vistau2_att == 3):
        if vistau1_reco_pt != 0 and vistau2_reco_pt !=0:
            if vistau1_att == 1:
                decaymode_e_count += 1
            if vistau2_att == 1:
                decaymode_e_count += 1
            if vistau1_att == 2:
                decaymode_mu_count+= 1
            if vistau2_att == 2:
                decaymode_mu_count += 1
            if vistau1_att == 3:
                decaymode_had_count += 1
            if vistau2_att == 3:
                decaymode_had_count += 1
            ditaumass.append(ditaumass_value)
            ditauvismass.append(ditauvismass_value)
            decaymode_count += 1
            inputSVfit.append([vistau1_reco_pt,vistau1_reco_eta,vistau1_reco_phi,vistau1_reco_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_reco_pt,vistau2_reco_eta,vistau2_reco_phi,vistau2_reco_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
        #ditaumass.append(ditaumass_value)
        #ditauvismass.append(ditauvismass_value)
        #decaymode_count += 1
        #inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
        #if vistau1_att == 3 and vistau2_att == 3 and vistau1_decaymode == 11 and vistau2_decaymode == 11:
        #    ditaumass.append(ditaumass_value)
        #    ditauvismass.append(ditauvismass_value)
        #    decaymode_count += 1
        #    inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,METpx,METpy,METCOV,ditaumass_value,ditauvismass_value])
    """
    else:
        break
    
      
ditaumass = numpy.array(ditaumass,numpy.float64)
inputSVfit = numpy.array(inputSVfit,numpy.float64)      
dataset_length = len(inputSVfit[:,0])
train_length = int(round(dataset_length*0.7))
test_length = int(round(dataset_length*0.3))
dataset_svfit_length = len(inputSVfit[:,0])
train_svfit_length = dataset_svfit_length - test_length

test_ditaumass = ditaumass[:]
test_ditauvismass = ditauvismass[:]
#test_ditaumass = ditaumass[train_length:dataset_length]
#test_ditauvismass = ditauvismass[train_length:dataset_length]
#inputSVfit = inputSVfit[train_svfit_length:dataset_svfit_length,:]

#histogram of ditau mass using neural network and SVfit
histtitle = "reconstructed di-#tau mass using SVfit %s decays" % (decaymode)
histditaumass = ROOT.TH1D("ditaumass",histtitle,150,0,150)
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleOffset(1.2)
histditaumass.SetLineColor(2)
histditaumass.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","ditauvismass",150,0,150)
histditauvismass.SetLineColor(6)
histditauvismass.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",150,0,150)
histditaumasssvfit.SetLineColor(3)
histditaumasssvfit.SetStats(0)
histditauvismassreco = ROOT.TH1D("ditauvismassreco","di-#tau visible mass ",150,0,150)
histditauvismassreco.SetLineColor(1)
histditauvismassreco.GetXaxis().SetTitle("di-#tau vismass [GeV]")
histditauvismassreco.GetYaxis().SetTitle("number of occurence")
histditauvismassreco.SetStats(0)

####SVfit delta histogram
histditaumasssvfitdelta = ROOT.TH1D("ditaumasssvfitdelta","delta between reconstructed di-#tau mass using SVfit and actual mass",100,-10,50)
histditaumasssvfitdelta.GetXaxis().SetTitle("#Deltam [GeV]")
histditaumasssvfitdelta.GetYaxis().SetTitle("number of occurence")
histditaumasssvfitdelta.SetStats(0)
histditaumasssvfitres = ROOT.TH1D("svfitresolution","resolution of reconstructed di-#tau mass using SVfit",100,-5,5)
histditaumasssvfitres.GetXaxis().SetTitle("resolution")
histditaumasssvfitres.GetYaxis().SetTitle("number of occurence")


#ratio histograms
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and actual mass",150,0,150)
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
    #COV00 = inputSVfit[14]
    #COV01 = inputSVfit[15]
    #COV10 = inputSVfit[16]
    #COV11 = inputSVfit[17]
    #METx = inputSVfit[18]
    #METy = inputSVfit[19]
    #ditaumass = inputSVfit[20]
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
    #covMET = ROOT.TMatrixD(2, 2)
    #covMET[0][0] = COV00
    #covMET[1][0] = COV10
    #covMET[0][1] = COV01
    #covMET[1][1] = COV11
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
    #return [mass,ditaumass,1.0,vistau1_att,vistau1_decaymode,vistau2_att,vistau2_decaymode,k]

def fill_histditaumass():
    for g in test_ditaumass:
        histditaumass.Fill(g)

def fill_histditauvismass():
    for f in test_ditauvismass:
        histditauvismass.Fill(f)

def fill_histditauvismassreco():
    for t in range(len(ditauvismass)):
        vistau1_reco_pt = inputSVfit[t,0]
        vistau1_reco_eta = inputSVfit[t,1]
        vistau1_reco_phi = inputSVfit[t,2]
        vistau1_reco_mass = inputSVfit[t,3]
        vistau2_reco_pt = inputSVfit[t,7]
        vistau2_reco_eta = inputSVfit[t,8]
        vistau2_reco_phi = inputSVfit[t,9]
        vistau2_reco_mass = inputSVfit[t,10]
        v_vistau1_reco = ROOT.TLorentzVector()
        v_vistau2_reco = ROOT.TLorentzVector()
        v_vistau1_reco.SetPtEtaPhiM(vistau1_reco_pt,vistau1_reco_eta,vistau1_reco_phi,vistau1_reco_mass)
        v_vistau2_reco.SetPtEtaPhiM(vistau2_reco_pt,vistau2_reco_eta,vistau2_reco_phi,vistau2_reco_mass)
        v_reco = v_vistau1_reco+v_vistau2_reco
        ditauvismass_reco = v_reco.M()
        histditauvismassreco.Fill(ditauvismass_reco)

###################################################################################
#output_name = "ditau_mass_nnvssvfit_final_%s_decays" % (decaymode)
output_name = "ditau_mass_svfit_try121_reco"
#output_name = "ditau_mass_pt_small_dataset"

output_file_name = "%s.txt" % (output_name)
output_root_name = "%s.root" % (output_name)
#rootfile = ROOT.TFile(output_root_name,"RECREATE")
#output_file = open(output_file_name,'w')
#sys.stdout = output_file

print "dataset length:",dataset_length
print "train length:",train_length
print "test length:",test_length
print "SVfit length:",len(inputSVfit[:,0])

print "all decays:",decaymode_all_count
print "full leptonic:",decaymode_fulllep_count
print "semi leptonic:",decaymode_semilep_count
print "full hadronic:",decaymode_fullhad_count
print "full leptonic percentage:",(float(decaymode_fulllep_count)/float(fixed_dataset_length))*100,"%"
print "semileptonic percentage:",(float(decaymode_semilep_count)/float(fixed_dataset_length))*100,"%"
print "full hadronic percentage:",(float(decaymode_fullhad_count)/float(fixed_dataset_length))*100,"%"
"""
fill_histditaumass()
fill_histditauvismass()
#fill_histditauvismassreco()

histditaumass.Write()
histditauvismass.Write()



#################             run SVfit          #########################
start_svfit = time.time()
pool = multiprocessing.Pool(processes=20)
#ditaumass_svfit = pool.map(reconstruct_mass,inputSVfit_test_dataset)
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
    #print "actual mass:",ditaumass_actual_loop,"reconstructed mass:",ditaumass_svfit_calc_loop, "vistau1_att:",vistau1_att,"vistau1_DM:",vistau1_decaymode,"vistau2_att:",vistau2_att,"vistau2_DM:",vistau2_decaymode
    histditaumasssvfitdelta.Fill(ditaumass_actual_loop-ditaumass_svfit_calc_loop)
    histditaumasssvfitres.Fill((ditaumass_actual_loop-ditaumass_svfit_calc_loop)/ditaumass_svfit_calc_loop)
   
for j in ditaumass_svfit_calc:
    histditaumasssvfit.Fill(j)  
failed_division = 0
for k in range(0,150):
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

histditaumasssvfit.Write()
histditaumasssvfitratio.Write()
print "SVfit execution time:",(end_svfit-start_svfit)/3600 ,"h"

#histogram of di-tau mass using regression all decays

canv2 = ROOT.TCanvas("di-tau using SVfit")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.3,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.3,0.02)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumasssvfit.GetMaximum())
histditaumass.SetMaximum(max_bin*1.08)
histditaumass.Draw()
histditaumasssvfit.Draw("SAME")
histditauvismass.Draw("SAME")
leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg2.AddEntry(histditauvismass,"di-#tau_{vis} mass","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau{SVfit} mass","PL")
leg2.Draw()
pad2.cd()
histditaumasssvfitratio.Draw("P")
unit_line = ROOT.TLine(0.0,1.0,150.0,1.0)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
canv2.Write()
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)

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

canv5 = ROOT.TCanvas("ditauvismassreco")
histditauvismassreco.Draw()
histditauvismass.Draw("SAME")
leg5 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg5.AddEntry(histditauvismassreco,"reco vismass","PL")
leg5.AddEntry(histditauvismass,"gen vismass","PL")
leg5.Draw()
output_hist_ditauvismassreco = "%s_ditauvismassreco.png" %(output_name)
canv5.Write()
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage(output_hist_ditauvismassreco)

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


output_file.close()
rootfile.Close()

"""

