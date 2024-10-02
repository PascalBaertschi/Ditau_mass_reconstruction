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
fixed_dataset_length = 1910000
fixed_train_length = 560000
fixed_test_length = 100000
fulllep_fixed_length = int(round(0.1226*fixed_dataset_length))
semilep_fixed_length = int(round(0.4553*fixed_dataset_length))
fullhad_fixed_length = int(round(0.422* fixed_dataset_length))
onevalue_fixed_length = 15000
onemass_value = 250.0
lowerlimit = onemass_value-1.0
upperlimit = onemass_value+1.0
###################################################################################
#list_name_normal = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_notsotight.csv"
#list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_cuts_notsotight_morelowmass.csv"
list_name_skim = "/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/batch_output/reg_ditau_mass_skim_correct40.csv"
#dataframe_ditaumass_normal = pandas.read_csv(list_name_normal,delim_whitespace=False,header=None)
#dataframe_ditaumass_skim = pandas.read_csv(list_name_skim,delim_whitespace=False,header=None)
dataframe_ditaumass = pandas.read_csv(list_name_skim,delim_whitespace=False,header=None)
#dataframe_ditaumass = dataframe_ditaumass_normal.append(dataframe_ditaumass_skim)
dataframe_ditaumass_shuffled = dataframe_ditaumass.sample(frac=1,random_state =1337)
dataset_ditaumass = dataframe_ditaumass_shuffled.values
dataset_total_length = len(dataset_ditaumass[:,0])

inputNN = []
inputSVfit = []
ditaumass = []
ditauvismass = []
decaymode_count = 0
decaymode_count_fulllep = 0
decaymode_count_semilep = 0
decaymode_count_fullhad = 0

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


######### make flat mass distribution with less events ######################
min_bincontent = 4900
histtrainditaumasscalc = ROOT.TH1D("trainditaumasscalc","trainditaumasscalc",300,0,300)


train_inputNN_selected = []
train_ditaumass_selected = []
train_inputNN_notused = []
train_ditaumass_notused = []

for k,ditaumass_loopvalue in enumerate(train_ditaumass_forselection):
    bin_index = histtrainditaumasscalc.GetXaxis().FindBin(ditaumass_loopvalue)
    bin_content = histtrainditaumasscalc.GetBinContent(bin_index)
    if bin_content < min_bincontent:
        histtrainditaumasscalc.SetBinContent(bin_index,bin_content+1)
        train_inputNN_selected.append(train_inputNN_forselection[k,:])
        train_ditaumass_selected.append(ditaumass_loopvalue)
    else:
        train_inputNN_notused.append(train_inputNN_forselection[k,:])
        train_ditaumass_notused.append(ditaumass_loopvalue)

train_inputNN_selected = numpy.array(train_inputNN_selected,numpy.float64)
train_inputNN_notused = numpy.array(train_inputNN_notused,numpy.float64)
test_inputNN_onevalue = []
test_ditaumass_onevalue = []
onevalue_count = 0

for j,ditaumass_loopvalue in enumerate(train_ditaumass_notused):
    if lowerlimit < ditaumass_loopvalue and upperlimit > ditaumass_loopvalue and onevalue_count< onevalue_fixed_length:
        test_inputNN_onevalue.append(train_inputNN_notused[j,:])
        test_ditaumass_onevalue.append(ditaumass_loopvalue)
        onevalue_count += 1
for g,ditaumass_loopvalue in enumerate(test_ditaumass_selected):
    if lowerlimit < ditaumass_loopvalue and upperlimit > ditaumass_loopvalue and onevalue_count < onevalue_fixed_length:
        test_inputNN_onevalue.append(test_inputNN_selected[g,:])
        test_ditaumass_onevalue.append(ditaumass_loopvalue)
        onevalue_count += 1

test_inputNN_onevalue = numpy.array(test_inputNN_onevalue,numpy.float64)

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
#train_inputNN[:,:] = preprocessing.scale(train_inputNN[:,:])
#test_inputNN[:,:] = preprocessing.scale(test_inputNN[:,:])
#overfit_inputNN[:,:] = preprocessing.scale(overfit_inputNN[:,:])
#train_inputNN_selected[:,:] = preprocessing.scale(train_inputNN_selected[:,:])
#test_inputNN_selected[:,:] = preprocessing.scale(test_inputNN_selected[:,:])
#test_inputNN_onevalue[:,:] = preprocessing.scale(test_inputNN_onevalue[:,:])
#overfit_inputNN_selected[:,:] = preprocessing.scale(overfit_inputNN_selected[:,:])
#############    preparing the histograms       ###################

#histogram of ditau mass using neural network and SVfit
histtitle = "reconstructed di-#tau mass using neural network and SVfit"
histditaumass = ROOT.TH1D("ditaumass",histtitle,100,onemass_value-30,onemass_value+30)
histditaumass.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleOffset(1.2)
histditaumass.SetLineColor(2)
histditaumass.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","reconstructed di-#tau vismass using neural network",100,onemass_value-30,onemass_value+30)
histditauvismass.SetLineColor(6)
histditauvismass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassnn","reconstructed di-#tau mass using neural network",100,onemass_value-30,onemass_value+30)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",100,onemass_value-30,onemass_value+30)
histditaumasssvfit.SetLineColor(3)
histditaumasssvfit.SetStats(0)

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




def neural_network(batch_size,epochs,output_name):
    print "NEURAL NETWORK"
    mass_model = Sequential()
    mass_model.add(Dense(200,input_dim=17,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(200,kernel_initializer='random_uniform',activation='relu'))
    mass_model.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    mass_model.compile(loss='mean_squared_error',optimizer='adam')
    history = mass_model.fit(train_inputNN_selected,train_ditaumass_selected,batch_size,epochs,validation_data = (test_inputNN_onevalue,test_ditaumass_onevalue),verbose = 2)
    mass_score = mass_model.evaluate(test_inputNN_onevalue,test_ditaumass_onevalue,batch_size,verbose=0)
    ditaumass_nn = mass_model.predict(test_inputNN_onevalue,batch_size,verbose=0)

    mass_model.summary()
    print "mass_model(",batch_size,epochs,")"
    print "loss (MSE):",mass_score
    print description_of_training
    failed_division = 0
    #preparing the histograms
    for j in ditaumass_nn:
        histditaumassnn.Fill(j)
    for i,ditaumass_value in enumerate(test_ditaumass_onevalue):
        res = (ditaumass_value - ditaumass_nn[i])/ditaumass_value
        histditaumassnnres.Fill(res)
        histditaumassnnrescomp.Fill(res)

    histditaumassnn.Write()
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
    for i,ditaumass_value in enumerate(test_ditaumass_onevalue):
        histditaumass.Fill(ditaumass_value)

#####################  getting the SVfit histograms ####################
svfit = ROOT.TFile.Open("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/batch_cuts_output/ditau_mass_svfit_cuts_250GeV_15e3.root")
histsvfit = svfit.Get("ditaumasssvfit")
histsvfitres = svfit.Get("svfitresolution")

for i in range(100):
    binvalue = histsvfit.GetBinContent(i+1)
    histditaumasssvfit.SetBinContent(i+1,binvalue)
for j in range(80):
    resvalue = histsvfitres.GetBinContent(j+1)
    histditaumasssvfitres.SetBinContent(j+1,resvalue)


svfit.Close()
###################################################################################

output_name = "ditau_mass_nn_cuts_250GeV_15e3_try5"
description_of_training = "Standardized input data  INPUT: 1000 (classification of decaychannel) vis tau1 (pt,eta,phi,E,mass)+vis tau2 (pt,eta,phi,E,mass)+MET_ET+MET_Phi+collinear ditaumass"
output_file_name = "%s.txt" % (output_name)
output_root_name = "%s.root" % (output_name)
rootfile = ROOT.TFile(output_root_name,"RECREATE")
output_file = open(output_file_name,'w')
sys.stdout = output_file

print "dataset length:",len(ditaumass)
#print "train length:",len(train_ditaumass)
#print "test length:",len(test_ditaumass)
print "train length:",len(train_ditaumass_selected)
print "test length:",len(test_ditaumass_onevalue)

fill_histditaumass()


######################### run neural network  ######################

batch_size = 128
epochs = 200

start_nn = time.time()
neural_network(batch_size,epochs,output_name)
end_nn = time.time()

print "NN executon time:",(end_nn-start_nn)/3600,"h" 


#histogram of di-tau mass using regression all decays

canv2 = ROOT.TCanvas("di-tau mass using NN and SVfit")
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum())
histditaumass.SetMaximum(max_bin*1.08)
histditaumass.Draw("HIST")
#histditauvismass.Draw("HIST")
histditaumassnn.Draw("HIST SAME")
#histditaumassnnoverfit.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
#histditauvismass.Draw("SAME")
leg2 = ROOT.TLegend(0.1,0.68,0.4,0.9)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
#leg2.AddEntry(histditauvismass,"di-#tau_{vis} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau_{NN} mass","PL")
#leg2.AddEntry(histditaumassnnoverfit,"overfit-test","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
leg2.Draw()
output_hist_name = "%s.png" %(output_name)
histditaumass.Write()
histditaumassnn.Write()
histditauvismass.Write()
histditaumasssvfit.Write()
canv2.Write()
img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage(output_hist_name)

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

canv3 = ROOT.TCanvas("resolution comparison")
max_bin = max(histditaumassnnrescomp.GetMaximum(),histditaumasssvfitres.GetMaximum())
histditaumassnnrescomp.SetMaximum(max_bin*1.08)
histditaumassnnrescomp.Draw()
histditaumasssvfitres.Draw("SAMES")
nn_statbox.Draw("SAME")
svfit_statbox.Draw("SAME")
output_res_compare_name = "%s_rescompar.png" %(output_name)
histditaumassnnrescomp.Write()
histditaumasssvfitres.Write()
canv3.Write()
img3 = ROOT.TImage.Create()
img3.FromPad(canv3)
img3.WriteImage(output_res_compare_name)

canv4 = ROOT.TCanvas("resolution")
histditaumassnnres.Draw()
histditaumassnnres.Write()
output_hist_res_name = "%s_res.png" %(output_name)
img4 = ROOT.TImage.Create()
img4.FromPad(canv4)
img4.WriteImage(output_hist_res_name)

canv6 = ROOT.TCanvas("ditaumass NN inputs")
histtrainditaumasscheck.Draw()
histtrainditaumasscheck.Write()
output_input_name = "%s_NNinput.png" %(output_name)
canv6.Write()
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage(output_input_name)

output_file.close()
rootfile.Close()

