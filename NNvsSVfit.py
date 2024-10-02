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

dataframe_ditaumass = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_NNvsSVfitMET_2e6.csv",delim_whitespace=False,header=None)
dataset_ditaumass = dataframe_ditaumass.values


histCOV00 = ROOT.TH1D("COV00","histogram of COV00",100,-20,20)
#histCOV00.GetXaxis().SetTitle("recoMETpx-genMETpx")
#histCOV00.GetYaxis().SetTitle("number of occurence")
#histCOV00.SetStats(0)
histCOV11 = ROOT.TH1D("COV11","histogram of COV11",100,-20,20)
#histCOV11.GetXaxis().SetTitle("recoMETpy-genMETpy")
#histCOV11.GetYaxis().SetTitle("number of occurence")
#histCOV11.SetStats(0)

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
COV00_width = COV00_fit.GetParameter(2)
#COV00_width = COV00_fit.GetParameter(2)*2*numpy.sqrt(2*numpy.log(2))
histCOV11.Fit("gaus")
COV11_fit = histCOV11.GetFunction("gaus")
COV11_width = COV11_fit.GetParameter(2)
#COV11_width = COV11_fit.GetParameter(2)*2*numpy.sqrt(2*numpy.log(2))
COV_width_mean = (COV00_width+COV11_width)/2
COVMET = ROOT.TMath.Power(COV_width_mean,2.)


dataset_ditaumass = dataset_ditaumass[0:791167,:]
#dataset_ditaumass = dataset_ditaumass[0:100,:]
dataset_length = len(dataset_ditaumass[:,0])
train_length = int(round(dataset_length*0.7))
test_length = int(round(dataset_length*0.3))
leptonic = 0
semileptonic = 0
hadronic = 0

histditaumassloop = ROOT.TH1D("ditaumassloop","reconstructed di-#tau mass only for loop",100,0,100)
histMETcheck = ROOT.TH2D("METcheck","genMET vs recoMET",100,0,100,100,0,100)
histMETcheck.SetStats(0)
histMETcheck.GetXaxis().SetTitle("genMET [GeV]")
histMETcheck.GetYaxis().SetTitle("recoMET [GeV]")
inputNN = []
inputSVfit = []
ditaumass = []
inputSVfit_90GeV = []
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
    histMETcheck.Fill(genMissingET_MET,MissingET_MET)

    genMETpx = genMissingET_MET*numpy.cos(genMissingET_Phi)
    genMETpy = genMissingET_MET*numpy.sin(genMissingET_Phi)
    METpx = MissingET_MET*numpy.cos(MissingET_Phi)
    METpy = MissingET_MET*numpy.sin(MissingET_Phi)
    METCOV00 = ROOT.TMath.Power(METpx-genMETpx,2.)
    METCOV11 = ROOT.TMath.Power(METpy-genMETpy,2.)
    v_vistau1 = ROOT.TLorentzVector()
    v_vistau1.SetPtEtaPhiM(vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass)
    v_vistau2 = ROOT.TLorentzVector()
    v_vistau2.SetPtEtaPhiM(vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass)
    v_nu = ROOT.TLorentzVector()
    v_nu.SetPtEtaPhiM(nu_pt,nu_eta,nu_phi,nu_mass)
    nu_px = v_nu.Px()
    nu_py = v_nu.Py()
    v_mot = v_vistau1+v_vistau2+v_nu
    v_vismot = v_vistau1+v_vistau2
    ditaumass_value = v_mot.M()
    ditauvismass_value = v_vismot.M()
    vismass = v_vismot.M()
    p_vis = v_vismot.P()
    pt_vis = v_vismot.Pt()
    pt_nu = v_nu.Pt()
    pt = v_mot.Pt()
    mass_no_pz = v_vismot.E()**2-v_vismot.Pz()**2-pt_vis**2
    if vistau1_att in (1,2) and vistau2_att in (1,2):
        leptonic += 1
    if (vistau1_att in (1,2) and vistau2_att ==3) or (vistau1_att == 3 and vistau2_att in (1,2)):
        semileptonic += 1
    if vistau1_att == 3 and vistau2_att == 3:
        hadronic += 1
    """
    print "genMETpx:",genMETpx,"genMETpy:",genMETpy
    print "METpx:",METpx,"METpy:",METpy
    print "nu px:",nu_px,"nu py:",nu_py
    print ""
    """
    histditaumassloop.Fill(ditaumass_value)
    #inputNN.append([v_vismot.Pt(),v_vismot.Eta(),v_vismot.Phi(),v_vismot.M(),v_nu.Px(),v_nu.Py()])
    #inputNN.append([v_vismot.Px(),v_vismot.Py(),v_vismot.Pz(),v_vismot.E(),v_nu.Px(),v_nu.Py(),vismass,p_vis])
    #inputNN.append([v_vistau1.Px(),v_vistau1.Py(),v_vistau1.Pz(),v_vistau1.E(),v_vistau2.Px(),v_vistau2.Py(),v_vistau2.Pz(),v_vistau2.E(),v_nu.Px(),v_nu.Py(),p_vis,vismass,pt_nu,pt_vis,pt,mass_no_pz])
    inputNN.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),p_vis,vismass])
    #inputNN.append([v_vistau1.Px(),v_vistau1.Py(),v_vistau1.Pz(),v_vistau1.E(),vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,v_vistau2.Px(),v_vistau2.Py(),v_vistau2.Pz(),v_vistau2.E(),vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,v_nu.Px(),v_nu.Py(),vismass,p_vis,pt_vis,pt_nu,pt,mass_no_pz])
    ditaumass.append(ditaumass_value)
    if vistau1_pt == 0.0 or vistau2_pt == 0.0:
       continue
    else:
        if ditaumass_value > 80.0 and ditaumass_value < 100.0:
            inputSVfit_90GeV.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,v_nu.Px(),v_nu.Py(),METCOV00,METCOV11,COVMET,ditaumass_value,ditauvismass_value,nu_pt,nu_phi])
        inputSVfit.append([vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,v_nu.Px(),v_nu.Py(),METCOV00,METCOV11,COVMET,ditaumass_value,ditauvismass_value,nu_pt,nu_phi])
inputNN_normalized = []
ditaumass_normalized = []
inputNN = numpy.array(inputNN,numpy.float64)
ditaumass = numpy.array(ditaumass,numpy.float64)
inputSVfit = numpy.array(inputSVfit,numpy.float64)
inputSVfit_90GeV = numpy.array(inputSVfit_90GeV,numpy.float64)

train_inputNN = inputNN[0:train_length,:]
train_ditaumass = ditaumass[0:train_length]
test_inputNN = inputNN[train_length:dataset_length,:]
test_ditaumass = ditaumass[train_length:dataset_length]
overfit_inputNN = train_inputNN[0:len(test_ditaumass),:]
inputSVfit = inputSVfit[train_length:dataset_length,:]

print "number of leptonic decays:",leptonic
print "number of semileptonic decays:",semileptonic
print "number of hadronic decays:",hadronic

for i in range(len(ditaumass)):
    bin_number = histditaumassloop.FindBin(ditaumass[i])
    weight = histditaumassloop.GetBinContent(bin_number)
    inputNN_normalized.append([inputNN[i,0]/weight,inputNN[i,1]/weight,inputNN[i,2]/weight,inputNN[i,3]/weight,inputNN[i,4]/weight,inputNN[i,5]/weight,inputNN[i,6]/weight,inputNN[i,7]/weight,inputNN[i,8]/weight,inputNN[i,9]/weight,inputNN[i,10]/weight,inputNN[i,11]/weight])
    ditaumass_normalized.append(ditaumass[i]/weight)
inputNN_normalized = numpy.array(inputNN_normalized,numpy.float64)
ditaumass_normalized = numpy.array(ditaumass_normalized,numpy.float64)
inputNN[:,:] = preprocessing.scale(inputNN[:,:])
inputNN_normalized[:,:] = preprocessing.scale(inputNN_normalized[:,:])
train_inputNN_normalized = inputNN_normalized[0:train_length,:]
train_ditaumass_normalized = ditaumass_normalized[0:train_length]
test_inputNN_normalized = inputNN_normalized[train_length:dataset_length,:]
test_ditaumass_normalized = ditaumass_normalized[train_length:dataset_length]
overfit_inputNN_normalized = train_inputNN_normalized[0:len(test_ditaumass),:]


#histogram of ditau mass using neural network and SVfit
#histditaumass = ROOT.TH1D("ditaumass","reconstructed di-#tau mass using neural network and SVfit",100,0,100)
histditaumass = ROOT.TH1D("ditaumass","reconstructed di-#tau mass using neural network",100,0,100)
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleOffset(1.2)
histditaumass.SetLineColor(2)
histditaumass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassregall","reconstructed di-#tau mass using neural network",100,0,100)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",100,0,100)
histditaumasssvfit.SetLineColor(3)
histditaumasssvfit.SetStats(0)
histditaumassnnoverfit = ROOT.TH1D("ditaumassnnoverfit","dsfdsfg",100,0,100)
histditaumassnnoverfit.SetLineColor(3)
histditaumassnnoverfit.SetLineStyle(2)
histditaumassnnoverfit.SetStats(0)

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
histditaumasssvfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(3)
histditaumasssvfitratio.SetStats(0)
histditaumassnnoverfitratio = ROOT.TH1D("ditaumassregalloverfitratio","ratio between reconstruced and overfit mass",100,0,100)
histditaumassnnoverfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(4)
histditaumassnnoverfitratio.SetStats(0)


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
    print "SVFIT"
    def reconstruct_mass(vistau1_pt,vistau1_eta,vistau1_phi,vistau1_mass,vistau1_att,vistau1_prongs,vistau1_pi0,vistau2_pt,vistau2_eta,vistau2_phi,vistau2_mass,vistau2_att,vistau2_prongs,vistau2_pi0,measuredMETx,measuredMETy,METCOV00,METCOV11,COVMET,ditaumass,ditauvismass):
        # define MET covariance
        covMET = ROOT.TMatrixD(2, 2)
        covMET[0][0] = COVMET
        covMET[1][0] = 0.0
        covMET[0][1] = 0.0
        covMET[1][1] = COVMET
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
        #algo.addLogM(False)
        inputFileName_visPtResolution = "/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/SVfit_standalone/data/svFitVisMassAndPtResolutionPDF.root"
        ROOT.TH1.AddDirectory(False)
        inputFile_visPtResolution = ROOT.TFile(inputFileName_visPtResolution)
        algo.shiftVisPt(True, inputFile_visPtResolution)
        algo.integrateMarkovChain()
        mass = algo.getMCQuantitiesAdapter().getMass()  # full mass of tau lepton pair in units of GeV
        transverseMass = algo.getMCQuantitiesAdapter().getTransverseMass()  # transverse mass of tau lepton pair in units of GeV
        inputFile_visPtResolution.Close()
        return [mass,ditaumass,ditauvismass,vistau1_att,vistau1_decaymode,vistau2_att,vistau2_decaymode,k]

    time_per_event = 0
    for j in range(event_start,event_stop):
        start = time.time()
        ditaumass_svfit = reconstruct_mass(inputSVfit[j,0],inputSVfit[j,1],inputSVfit[j,2],inputSVfit[j,3],inputSVfit[j,4],inputSVfit[j,5],inputSVfit[j,6],inputSVfit[j,7],inputSVfit[j,8],inputSVfit[j,9],inputSVfit[j,10],inputSVfit[j,11],inputSVfit[j,12],inputSVfit[j,13],inputSVfit[j,14],inputSVfit[j,15],inputSVfit[j,16],inputSVfit[j,17],inputSVfit[j,18],inputSVfit[j,19],inputSVfit[j,20])
        
        print "Event:",j
        if inputSVfit[j,4] in (1,2) and inputSVfit[j,11] in (1,2):
            print "leptonic decay"
        elif (inputSVfit[j,4] in (1,2) and inputSVfit[j,11] == 3)or (inputSVfit[j,4] == 3 and inputSVfit[j,11] in (1,2)):
            print "semileptonic decay"
        elif inputSVfit[j,4]==3 and inputSVfit[j,11]==3:
            print "hadronic decay"
            
        #print "vistau1 4Vec (pt,eta,phi,mass): (",inputSVfit_90GeV[j,0],inputSVfit_90GeV[j,1],inputSVfit_90GeV[j,2],inputSVfit_90GeV[j,3],")"
        #print "vistau2 4Vec (pt,eta,phi,mass): (",inputSVfit_90GeV[j,7],inputSVfit_90GeV[j,8],inputSVfit_90GeV[j,9],inputSVfit_90GeV[j,10],")"
        #print "MET:",inputSVfit_90GeV[j,20], "MET_Phi:",inputSVfit_90GeV[j,21]
        #print "METCOV00:",inputSVfit_90GeV[j,16],"METCOV11:",inputSVfit_90GeV[j,17]
        #print "// Event_number =",j,";"
        #print "// float   metcov00=",inputSVfit[j,16],",metcov10=0.,metcov11=",inputSVfit[j,17],",metet=",inputSVfit[j,21],", metphi=",inputSVfit[j,22],";"
        #print "// TLorentzVector tautlv_,leptontlv_;"
        #print "// tautlv_.SetPtEtaPhiM(",inputSVfit[j,0],",",inputSVfit[j,1],",",inputSVfit[j,2],",",inputSVfit[j,3],");"
        #print "// leptontlv_.SetPtEtaPhiM(",inputSVfit[j,7],",",inputSVfit[j,8],",",inputSVfit[j,9],",",inputSVfit[j,10],");"
        #print ""
        print "decaymode vistau1:",ditaumass_svfit[4],"decaymode vistau2:",ditaumass_svfit[6]
        print "vistau1_att:",inputSVfit[j,4],"vistau2_att:",inputSVfit[j,11]
        print "1 = electron, 2 = muon, 3 = hadrons"
        print "actual mass:",inputSVfit[j,19], "reconstructed mass:",ditaumass_svfit[0]
        print ""
        print ""

        #histditaumasssvfit.Fill(ditaumass_svfit[0])
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


def neural_network(batch_size,epochs,learning_rate,decay_rate,droprate,output_name):
    print "NEURAL NETWORK"
    mass_model = Sequential()
    #mass_model.add(Dropout(droprate,input_shape=(12,)))
    mass_model.add(Dense(200,input_dim=12,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(droprate))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(droprate))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(droprate))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='softsign'))
    #mass_model.add(Dropout(droprate))
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
    mass_model.compile(loss='mean_squared_error',optimizer='adam')
    #mass_model.compile(loss='mean_squared_error',optimizer=adam)
    #history = mass_model.fit(inputNN_normalized,ditaumass_normalized,batch_size,epochs, validation_split = 0.3,verbose = 2)
    history = mass_model.fit(inputNN,ditaumass,batch_size,epochs, validation_split = 0.3,verbose = 2)
    #history = mass_model.fit(inputNN,ditaumass,batch_size,epochs, validation_split = 0.3,callbacks = lrate,verbose = 2)
    #mass_score = mass_model.evaluate(test_inputNN_normalized,test_ditaumass_normalized,batch_size,verbose=0)
    #ditaumass_nn = mass_model.predict(test_inputNN_normalized,batch_size,verbose=0)
    #ditaumass_nn_overfit = mass_model.predict(overfit_inputNN_normalized,batch_size,verbose=0)
    mass_score = mass_model.evaluate(test_inputNN,test_ditaumass,batch_size,verbose=0)
    ditaumass_nn = mass_model.predict(test_inputNN,batch_size,verbose=0)
    ditaumass_nn_overfit = mass_model.predict(overfit_inputNN,batch_size,verbose=0)

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
    
def svfit_fill_histditaumass(event_start,event_stop):
    for g in test_ditaumass[event_start:event_stop]:
        histditaumass.Fill(g)

def fill_histditaumass():
    for g in test_ditaumass:
        histditaumass.Fill(g)

def fill_histditaumass_normalized():
    for g in test_ditaumass_normalized:
        histditaumass.Fill(g)

####run NN/SVfit here###
#output_name = "ditau_mass_svfit_try59"
output_name = "ditau_mass_standardized100_try352"
#description_of_training = "standardized input data 10 Epochs try 2 step decay INPUT:vis tau1 (pt,eta,phi,mass)+vis tau2 (pt,eta,phi,mass)+METx+METy+p vis+vismass"
description_of_training = "standardized 1e6 input data 100 Epochs try 352 INPUT:vistau1 4Vec(pt,eta,phi,mass)+vistau2 4Vec(pt,eta,phi,mass)+METx+METy+p_vis+vismass"
output_file_name = "%s.txt" % (output_name)
#output_file = open(output_file_name,'w')
#sys.stdout = output_file

print "dataset length:",dataset_length
print "train length:",train_length
print "test length:",test_length


event_start = 0
event_stop = 20
batch_size = 60
epochs = 100
droprate = 0.5
learning_rate = 0.01
decay_rate = 0.00012
#decay_rate = 0.0


#fill_histditaumass()
#fill_histditaumass_normalized()
#neural_network(batch_size,epochs,learning_rate,decay_rate,droprate,output_name)

#svfit_fill_histditaumass(event_start,event_stop)
SVfit(event_start,event_stop,output_name)
"""
#histogram of di-tau mass
canv2 = ROOT.TCanvas("di-tau mass")
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
#histditaumassnnoverfit.Draw("SAME")
#histditaumasssvfit.Draw("SAME")
leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumass,"actual mass","PL")
leg2.AddEntry(histditaumassnn,"neural network","PL")
#leg2.AddEntry(histditaumassnnoverfit,"overfit-test","PL")
#leg2.AddEntry(histditaumasssvfit,"SVfit","PL")
leg2.Draw()
pad2.cd()
histditaumassnnratio.Draw("P")
#histditaumasssvfitratio.Draw("P SAME")
leg3 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg3.AddEntry(histditaumassnnratio,"neural network","P")
leg3.AddEntry(histditaumasssvfitratio,"SVfit","P")
#leg3.Draw()
unit_line = ROOT.TLine(0.0,1.0,100.0,1.0)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)
"""
"""
canv3 = ROOT.TCanvas("MET Check")
histMETcheck.Draw()
output_hist_METcheck_name = "%s_METcheck.png" %(output_name)
img3 = ROOT.TImage.Create()
img3.FromPad(canv3)
img3.WriteImage(output_hist_METcheck_name)
"""
#output_file.close()
