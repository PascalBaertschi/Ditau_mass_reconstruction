/* File: SVFit.i */
%module SVFit

%{
#define SWIG_FILE_WITH_INIT
#include "SVFit.h"
%}

TLorentzVector VHTausAnalysis::applySVFitHadronic( float cov00, float cov10, float cov11, float met, float met_phi, TLorentzVector lep1 , TLorentzVector lep2);
