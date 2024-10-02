from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
import pandas
import ROOT
import numpy as np
import time
from rep.estimators import xgboost
import xgboost as xgb

start = time.time()

dataframe_housing = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset_housing = dataframe_housing.values

#load dataset hadronic decays
dataframe_mass_train = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train = dataframe_mass_train.values
dataframe_mass_test= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test = dataframe_mass_test.values
dataframe_mass= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass = dataframe_mass.values

dataframe_mass_train_total_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_total_had = dataframe_mass_train_total_had.values
dataframe_mass_test_total_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_total_had = dataframe_mass_test_total_had.values
dataframe_mass_total_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_total_had = dataframe_mass_total_had.values

dataframe_mass_train_inclnu_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_inclnu_had = dataframe_mass_train_inclnu_had.values
dataframe_mass_test_inclnu_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_inclnu_had = dataframe_mass_test_inclnu_had.values
dataframe_mass_inclnu_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_inclnu_had = dataframe_mass_inclnu_had.values


dataframe_vismass_train_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_vismass_had_1e6_new.csv",delim_whitespace=False,header=None)
dataset_vismass_train_had = dataframe_vismass_train_had.values
dataframe_vismass_test_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_vismass_had_1e6_new.csv",delim_whitespace=False,header=None)
dataset_vismass_test_had = dataframe_vismass_test_had.values


#load dataset all decays
dataframe_mass_train_all = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_all = dataframe_mass_train_all.values
dataframe_mass_test_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_all = dataframe_mass_test_all.values
dataframe_mass_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_all = dataframe_mass_all.values

dataframe_mass_train_inclnu = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_inclnu = dataframe_mass_train_inclnu.values
dataframe_mass_test_inclnu= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_inclnu = dataframe_mass_test_inclnu.values
dataframe_mass_inclnu= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_inclnu = dataframe_mass_inclnu.values

dataframe_mass_train_total = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_total = dataframe_mass_train_total.values
dataframe_mass_test_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_total = dataframe_mass_test_total.values
dataframe_mass_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_total = dataframe_mass_total.values

dataframe_train_theta = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_train_theta = dataframe_train_theta.values
dataframe_test_theta= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_test_theta = dataframe_test_theta.values
dataframe_theta= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_theta = dataframe_theta.values

dataframe_vismass_train_all = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_vismass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_vismass_train_all = dataframe_vismass_train_all.values
dataframe_vismass_test_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_vismass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_vismass_test_all = dataframe_vismass_test_all.values


# split into input and output variables
#hadronic decays
train_input = dataset_mass_train[:,0:7]
train_output = dataset_mass_train[:,7]
test_input = dataset_mass_test[:,0:7]
test_output = dataset_mass_test[:,7]
mass_input = dataset_mass[:,0:7]
mass_output = dataset_mass[:,7]

train_input_inclnu_had = dataset_mass_train_inclnu_had[:,0:8]
train_output_inclnu_had = dataset_mass_train_inclnu_had[:,8]
test_input_inclnu_had = dataset_mass_test_inclnu_had[:,0:8]
test_output_inclnu_had = dataset_mass_test_inclnu_had[:,8]
mass_input_inclnu_had = dataset_mass_inclnu_had[:,0:8]
mass_output_inclnu_had = dataset_mass_inclnu_had[:,8]

train_input_total_had = dataset_mass_train_total_had[:,0:4]
train_output_total_had = dataset_mass_train_total_had[:,4]
test_input_total_had = dataset_mass_test_total_had[:,0:4]
test_output_total_had = dataset_mass_test_total_had[:,4]
mass_input_total_had = dataset_mass_total_had[:,0:4]
mass_output_total_had = dataset_mass_total_had[:,4]

train_input_vismass_had = dataset_vismass_train_had[:,0:4]
train_output_vismass_had = dataset_vismass_train_had[:,4]
test_input_vismass_had = dataset_vismass_test_had[:,0:4]
test_output_vismass_had = dataset_vismass_test_had[:,4]


#all decays
train_input_all = dataset_mass_train_inclnu[:,0:6]
train_output_all = dataset_mass_train_inclnu[:,8]
test_input_all = dataset_mass_test_inclnu[:,0:6]
test_output_all = dataset_mass_test_inclnu[:,8]
mass_input_all = dataset_mass_inclnu[:,0:6]
mass_output_all=dataset_mass_inclnu[:,8]

train_input_inclnu = dataset_mass_train_inclnu[:,0:8]
train_output_inclnu = dataset_mass_train_inclnu[:,8]
test_input_inclnu = dataset_mass_test_inclnu[:,0:8]
test_output_inclnu = dataset_mass_test_inclnu[:,8]
mass_input_inclnu = dataset_mass_inclnu[:,0:8]
mass_output_inclnu = dataset_mass_inclnu[:,8]

#train_input_total = dataset_mass_train_total[:,0:4]
#train_output_total = dataset_mass_train_total[:,4]
#test_input_total = dataset_mass_test_total[:,0:4]
#test_output_total = dataset_mass_test_total[:,4]
mass_input_total = dataset_mass_total[:,0:4]
mass_output_total = dataset_mass_total[:,4]
train_input_total, test_input_total, train_output_total, test_output_total = train_test_split(mass_input_total,mass_output_total,test_size = 0.33, random_state = 42)
dtrain_total = xgb.DMatrix(train_input_total,train_output_total)
dtest_total = xgb.DMatrix(test_input_total,test_output_total)

train_length_total = len(train_output_total)
test_length_total = len(test_output_total)
overfit_total = train_input_total[0:len(test_output_total),:]

train_input_theta = dataset_train_theta[:,0:4]
train_output_theta = dataset_train_theta[:,4]
test_input_theta = dataset_test_theta[:,0:4]
test_output_theta = dataset_test_theta[:,4]
theta_input = dataset_theta[:,0:4]
theta_output = dataset_theta[:,4]

train_input_vismass_all = dataset_vismass_train_all[:,0:4]
train_output_vismass_all = dataset_vismass_train_all[:,4]
test_input_vismass_all = dataset_vismass_test_all[:,0:4]
test_output_vismass_all = dataset_vismass_test_all[:,4]

train_px = dataset_train_theta[:,0]
train_py = dataset_train_theta[:,1]
train_pz = dataset_train_theta[:,2]
train_E = dataset_train_theta[:,3]
train_theta = dataset_train_theta[:,4]
test_px = dataset_test_theta[:,0]
test_py = dataset_test_theta[:,1]
test_pz = dataset_test_theta[:,2]
test_E = dataset_test_theta[:,3]
test_theta = dataset_test_theta[:,4]


train_length = len(dataset_mass_train_total[:,0])
test_length = len(dataset_mass_test_total[:,0])
housing_length = len(dataset_housing[:,0])
train_length_housing = int(round(housing_length*0.7))

#histogram of ditau mass with regression only hadronic decays
histditaumassreg = ROOT.TH1D("ditaumassreg","di-#tau mass using regression hadronic decays",100,0,100)
histditaumassreg.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreg.GetYaxis().SetTitle("number of occurence")
histditaumassreg.SetLineColor(4)
histditaumassreg.SetStats(0)
histditaumassregout = ROOT.TH1D("ditaumassregout","di-#tau mass using regression hadronic decays",100,0,100)
histditaumassregout.SetLineColor(2)
histditaumassregout.SetStats(0)


#histogram of ditau mass with regression all particles
histditaumassregall = ROOT.TH1D("ditaumassregall","di-#tau mass using regression all decays",100,0,100)
histditaumassregall.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassregall.GetYaxis().SetTitle("number of occurence")
histditaumassregall.SetLineColor(4)
histditaumassregall.SetStats(0)
histditaumassregallout = ROOT.TH1D("ditaumassregallout","di-#tau mass using regression all decays out",100,0,100)
histditaumassregallout.SetLineColor(2)
histditaumassregallout.SetStats(0)


#histogram of ditau mass with regression all particles with all parameter (including pz of neutrino)
histditaumassreginclnu = ROOT.TH1D("ditaumassreginclnu","di-#tau mass using regression all decays all parameters",100,0,100)
histditaumassreginclnu.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreginclnu.GetYaxis().SetTitle("number of occurence")
histditaumassreginclnu.SetLineColor(2)
histditaumassreginclnu.SetStats(0)
histditaumassreginclnuout = ROOT.TH1D("ditaumassreginclnuout","di-#tau mass using regression",100,0,100)
histditaumassreginclnuout.SetLineColor(3)
histditaumassreginclnuout.SetStats(0)

#histogram of ditau mass with regression all particles with 4Vector (all decay-products)
histditaumassregtotal = ROOT.TH1D("ditaumassregtotal","di-#tau mass using regression all decays 4Vector all decay products",100,0,100)
histditaumassregtotal.GetXaxis().SetTitle("di-#tau mass")
histditaumassregtotal.GetYaxis().SetTitle("number of occurence")
histditaumassregtotal.SetLineColor(4)
histditaumassregtotal.SetStats(0)
histditaumassregtotalout = ROOT.TH1D("ditaumassregtotalout","ssss",100,0,100)
histditaumassregtotalout.SetLineColor(2)
histditaumassregtotalout.SetStats(0)
histditaumassregtotalcalc = ROOT.TH1D("ditaumassregtotalcalc","ssss",100,0,100)
histditaumassregtotalcalc.SetLineColor(3)
histditaumassregtotalcalc.SetLineStyle(2)
histditaumassregtotalcalc.SetStats(0)


#histogram of ditau theta with regression all particles
histditauthetareg = ROOT.TH1D("ditauthetareg","di-#tau theta using regression all decays",100,0,3.5)
histditauthetareg.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditauthetareg.GetYaxis().SetTitle("number of occurence")
histditauthetareg.SetLineColor(4)
histditauthetareg.SetStats(0)
histditauthetaregout = ROOT.TH1D("ditauthetaregout","di-#tau theta using regression all decays out",100,0,3.5)
histditauthetaregout.SetLineColor(2)
histditauthetaregout.SetStats(0)

#histogram of ditau vismass with regression all particles with 4Vector (all decay-products)
histditauvismassregall = ROOT.TH1D("ditauvismassregall","di-#tau vis-mass using regression all decays 4Vector all decay products",100,0,100)
histditauvismassregall.GetXaxis().SetTitle("di-#tau vis-mass")
histditauvismassregall.GetYaxis().SetTitle("number of occurence")
histditauvismassregall.SetLineColor(2)
histditauvismassregall.SetStats(0)
histditauvismassregallout = ROOT.TH1D("ditauvismassregallout","ssss",100,0,100)
histditauvismassregallout.SetLineColor(4)
histditauvismassregallout.SetStats(0)
histditauvismassregallcalc = ROOT.TH1D("ditauvismassregallcalc","tttt",100,0,100)
histditauvismassregallcalc.SetLineColor(3)
histditauvismassregallcalc.SetLineStyle(2)
histditauvismassregallcalc.SetStats(0)




#model_total = xgboost.XGBoostRegressor(n_estimators = 10000,subsample = 0.3, verbose = 1)
model_total = 
model_total.fit(train_input_total,train_output_total)
ditaumass_total = model_total.predict(test_input_total)
ditaumass_total_overfit = model_total.predict(train_input_total[0:len(test_output_total),:])
#ditaumass_total = runXGB(train_input_total,train_output_total,test_input_total)



"""
#regression using RandomForest hadronic decays
clf = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_split=2,random_state=0)
clf.fit(train_input,train_output)
scores = clf.score(test_input,test_output)
ditaumass = clf.predict(test_input)
print "score RandomForest hadronic:",scores

#regression using RandomForest all decays
clf_all = RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_split=2,random_state=0)
clf_all.fit(train_input_all,train_output_all)
scores_all = clf_all.score(test_input_all,test_output_all)
ditaumass_all = clf_all.predict(test_input_all)
print "score RandomForest:",scores_all
"""
"""
#regression using RandomForest inclnu
clf_inclnu = RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_split=2,random_state=0)
clf_inclnu.fit(train_input_inclnu,train_output_inclnu)
scores_inclnu = clf_inclnu.score(test_input_inclnu,test_output_inclnu)
ditaumass_inclnu = clf_inclnu.predict(test_input_inclnu)
print "score RandomForest inclnu:",scores_inclnu

#regression using RandomForest total
clf_total = RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_split=2,random_state=0)
clf_total.fit(train_input_total,train_output_total)
scores_total = clf_total.score(test_input_total,test_output_total)
ditaumass_total = clf_total.predict(test_input_total)
print "score RandomForest total:",scores_total
"""
"""
#regression using RandomForest for theta
clf_theta = RandomForestRegressor(n_estimators=20,max_depth=None,min_samples_split=2,random_state=0)
clf_theta.fit(train_input_theta,train_output_theta)
scores = clf_theta.score(test_input_theta,test_output_theta)
ditautheta = clf_theta.predict(test_input_theta)
print "score RandomForest for theta:",scores

#regression using RandomForest for vismass all
clf_vismass = RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_split=2,random_state=0)
clf_vismass.fit(train_input_vismass_all,train_output_vismass_all)
scores_vismass = clf_vismass.score(test_input_vismass_all,test_output_vismass_all)
ditauvismass_all = clf_vismass.predict(test_input_vismass_all)
print "score RandomForest for visible mass:",scores_vismass
"""

"""
for i in ditaumass:
    histditaumassreg.Fill(i)
for k in test_output:
    histditaumassregout.Fill(k)

for j in ditaumass_all:
    histditaumassregall.Fill(j)
for j in test_output_all:
    histditaumassregallout.Fill(j)

for g in ditaumass_inclnu:
    histditaumassreginclnu.Fill(g)
for d in test_output_inclnu:
    histditaumassreginclnuout.Fill(d)
"""
for u in ditaumass_total:
    histditaumassregtotal.Fill(u)
for g in test_output_total:
    histditaumassregtotalout.Fill(g)

for i in ditaumass_total_overfit:
    histditaumassregtotalcalc.Fill(i)

#for i in range(0,len(test_output_total)):
#    mass_calc = ROOT.TLorentzVector(test_input_total[i,0],test_input_total[i,1],test_input_total[i,2],test_input_total[i,3]).M()
#    histditaumassregtotalcalc.Fill(mass_calc)
"""
for o in ditautheta:
    histditauthetareg.Fill(o)
for s in test_output_theta:
    histditauthetaregout.Fill(s)

for p in ditauvismass_all:
    histditauvismassregall.Fill(p)
for s in test_output_vismass_all:
    histditauvismassregallout.Fill(s)

for i in range(0,len(dataset_vismass_test_all)):
    vismass_calc = ROOT.TLorentzVector(dataset_vismass_test_all[i,0],dataset_vismass_test_all[i,1],dataset_vismass_test_all[i,2],dataset_vismass_test_all[i,3]).M()
    histditauvismassregallcalc.Fill(vismass_calc)
"""
"""
#histogram of di-tau mass using regression hadronic decays
canv1 = ROOT.TCanvas("di-tau mass using regression hadronic")
max_bin1 = max(histditaumassreg.GetMaximum(),histditaumassregout.GetMaximum())
histditaumassreg.SetMaximum(max_bin1*1.08)
histditaumassreg.Draw()
histditaumassregout.Draw("SAME")

leg1 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg1.AddEntry(histditaumassreg,"reconstructed mass","PL")
leg1.AddEntry(histditaumassregout,"actual mass","PL")
leg1.Draw()


img1 = ROOT.TImage.Create()
img1.FromPad(canv1)
img1.WriteImage("reg_randomforest_ditau_mass_hadronic.png")

#histogram of di-tau mass using regression all decays
canv2 = ROOT.TCanvas("di-tau mass using regression all decays")
max_bin2 = max(histditaumassregall.GetMaximum(),histditaumassregallout.GetMaximum())
histditaumassregall.SetMaximum(max_bin2*1.08)
histditaumassregall.Draw()
histditaumassregallout.Draw("SAME")

leg2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg2.AddEntry(histditaumassregall,"reconstructed mass","PL")
leg2.AddEntry(histditaumassregallout,"actual mass","PL")
leg2.Draw()

img2 = ROOT.TImage.Create()
img2.FromPad(canv2)
img2.WriteImage("reg_randomforest_ditau_mass_all.png")


#histogram of di-tau theta using regression all decays
canv3 = ROOT.TCanvas("di-tau theta using regression all decays")
max_bin3 = max(histditauthetareg.GetMaximum(),histditauthetaregout.GetMaximum())
histditauthetareg.SetMaximum(max_bin3*1.08)
histditauthetareg.Draw()
histditauthetaregout.Draw("SAME")

leg3 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg3.AddEntry(histditauthetareg,"reconstructed mass","PL")
leg3.AddEntry(histditauthetaregout,"actual mass","PL")
leg3.Draw()

img3 = ROOT.TImage.Create()
img3.FromPad(canv3)
img3.WriteImage("reg_randomforest_ditau_theta.png")
"""
"""
#histogram of di-tau mass using regression all decays all parameters
canv4 = ROOT.TCanvas("di-tau mass using regression all decays all parameters")
max_bin4 = max(histditaumassreginclnu.GetMaximum(),histditaumassreginclnuout.GetMaximum())
histditaumassreginclnu.SetMaximum(max_bin4*1.08)
histditaumassreginclnu.Draw()
histditaumassreginclnuout.Draw("SAME")

leg4 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg4.AddEntry(histditaumassreginclnu,"reconstructed mass","PL")
leg4.AddEntry(histditaumassreginclnuout,"actual mass","PL")
leg4.Draw()


img4 = ROOT.TImage.Create()
img4.FromPad(canv4)
img4.WriteImage("reg_randomforest_ditau_mass_inclnu.png")

"""
#histogram of di-tau mass using regression all decays 4Vector all decay-products
canv5 = ROOT.TCanvas("di-tau mass using regression all decays 4Vector all decay products")
max_bin5 = max(histditaumassregtotal.GetMaximum(),histditaumassregtotalout.GetMaximum(),histditaumassregtotalcalc.GetMaximum())
histditaumassregtotal.SetMaximum(max_bin5*1.08)
histditaumassregtotal.Draw()
histditaumassregtotalout.Draw("SAME")
histditaumassregtotalcalc.Draw("SAME")

leg5 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg5.AddEntry(histditaumassregtotal,"reconstructed mass","PL")
leg5.AddEntry(histditaumassregtotalout,"actual mass","PL")
leg5.AddEntry(histditaumassregtotalcalc,"overfit mass","PL")
#leg5.AddEntry(histditaumassregtotalcalc,"calculated mass","PL")
leg5.Draw()

img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage("reg_xgboost_ditau_mass_total.png")
canv5.WaitPrimitive()

"""
#histogram of di-tau vismass using regression all decays 4Vector all decay-products
canv6 = ROOT.TCanvas("di-tau vismass regression all")
max_bin6 = max(histditauvismassregall.GetMaximum(),histditauvismassregallout.GetMaximum(),histditauvismassregallcalc.GetMaximum())
histditauvismassregall.SetMaximum(max_bin6*1.2)
histditauvismassregall.Draw()
histditauvismassregallout.Draw("SAME")
histditauvismassregallcalc.Draw("SAME")


leg6 = ROOT.TLegend(0.6,0.8,0.9,0.9)
leg6.AddEntry(histditauvismassregall,"reconstructed vis-mass","PL")
leg6.AddEntry(histditauvismassregallout,"actual vis-mass","PL")
leg6.AddEntry(histditauvismassregallcalc,"calculated vis-mass","PL")
leg6.Draw()

img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage("reg_randomforest_vismass_ditau_all.png")
"""

end = time.time()

print "execution time:",round((end-start)/60.0,3),"min"
