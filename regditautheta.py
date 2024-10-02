import numpy as np
from sklearn.svm import SVR
#from pylab import *
import pandas
import ROOT


dataframe_train_theta = pandas.read_csv("train_reg_ditau_theta.csv",delim_whitespace=False,header=None)
dataset_train_theta = dataframe_train_theta.values
dataframe_test_theta= pandas.read_csv("test_reg_ditau_theta.csv",delim_whitespace=False,header=None)
dataset_test_theta = dataframe_test_theta.values
dataframe_theta= pandas.read_csv("reg_ditau_theta.csv",delim_whitespace=False,header=None)
dataset_theta = dataframe_theta.values

train_input_theta = dataset_train_theta[:,0:4]
train_output_theta = dataset_train_theta[:,4]
test_input_theta = dataset_test_theta[:,0:4]
test_output_theta = dataset_test_theta[:,4]
theta_input = dataset_theta[:,0:4]
theta_output = dataset_theta[:,4]

#histogram of ditau theta with regression all particles with 4Vector (all decay-products)
histditauregtheta = ROOT.TH1D("ditauregtheta","di-#tau theta using regression all decays 4Vector all decay products",100,0,4)
histditauregtheta.GetXaxis().SetTitle("di-#tau theta")
histditauregtheta.GetYaxis().SetTitle("number of occurence")
histditauregtheta.SetLineColor(4)
histditauregtheta.SetStats(0)
histditauregthetaout = ROOT.TH1D("ditauregthetaout","ssss",100,0,4)
histditauregthetaout.SetLineColor(2)
histditauregthetaout.SetStats(0)

"""

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - np.random.rand(8))

"""

svr_rbf = SVR(kernel='rbf',C=1e3)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)


y_rbf = svr_rbf.fit(train_input_theta, train_output_theta).predict(test_input_theta)
#y_lin = svr_lin.fit(train_input_theta, train_output_theta).predict(train_input_theta)
#y_poly = svr_poly.fit(train_input_theta, train_output_theta).predict(test_input_theta)

for i in y_rbf:
    histditauregtheta.Fill(i)

for j in test_output_theta:
    histditauregthetaout.Fill(j)

#histogram of di-tau theta using regression all decays 4Vector all decay-products
canv5 = ROOT.TCanvas("di-tau theta using regression all decays 4Vector all decay products")
max_bin5 = max(histditauregtheta.GetMaximum(),histditauregthetaout.GetMaximum())
histditauregtheta.SetMaximum(max_bin5*1.08)
histditauregtheta.Draw()
histditauregthetaout.Draw("SAME")


leg5 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg5.AddEntry(histditauregtheta,"reconstructed theta","PL")
leg5.AddEntry(histditauregthetaout,"actual theta","PL")
leg5.Draw()

#canv5.Write()
img5 = ROOT.TImage.Create()
img5.FromPad(canv5)
img5.WriteImage("reg_ditau_theta.png")

canv5.WaitPrimitive()
"""
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

lw = 2
scatter(X, y, color='darkorange', label='data')
hold('on')
plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plot(X, y_lin, color='c', lw=lw, label='Linear model')
plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
xlabel('data')
ylabel('target')
title('Support Vector Regression')
legend()
show()

"""
