import ROOT
import numpy as np

Vec = ROOT.TLorentzVector(3.0,5.0,34.0,45.0)
pt = Vec.Pt()
pt_calc = np.sqrt(3**2+5**2)
print "pt from Vec:",pt
print "pt from sqrt:",pt_calc
