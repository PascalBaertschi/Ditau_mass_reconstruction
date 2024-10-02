/* File: SVFit.cxx*/
#include "SVFit.h"

TLorentzVector VHTausAnalysis::applySVFitHadronic( float cov00, float cov10, float cov11, float met, float met_phi, TLorentzVector lep1 , TLorentzVector lep2){
  // std::cout<<"inside applySVFitHadronic "<<std::endl;

  
  TLorentzVector   lBoson4;
  lBoson4.SetPtEtaPhiE(0,0,0,0);
  
  TMatrixD covMETraw(2,2);
  covMETraw[0][0]=  cov00;
  covMETraw[1][0]=  cov10;
  covMETraw[0][1]=  cov10;
  covMETraw[1][1]=  cov11;
  float lcov00 =  cov00;
  float lcov10 =  cov10;
  float lcov01 =  cov10;
  float lcov11 =  cov11;
  
  TLorentzVector   nullo;
  nullo.SetPtEtaPhiE(0,0,0,0);
  if(lcov00*lcov11-lcov01*lcov01 == 0) {
    std::cout<<"covMat det null "<<std::endl;
    return nullo;
  }

 
  lBoson4.SetPtEtaPhiE(0,0,0,0);
  NSVfitStandalone::Vector measuredMET(met *TMath::Cos(met_phi), met *TMath::Sin(met_phi), 0); 
  // setup measure tau lepton vectors 
  NSVfitStandalone::LorentzVector l1(lep1.Px(), lep1.Py(), lep1.Pz(), TMath::Sqrt(lep1.M()*lep1.M()+lep1.Px()*lep1.Px()+lep1.Py()*lep1.Py()+lep1.Pz()*lep1.Pz()));
  NSVfitStandalone::LorentzVector l2(lep2.Px(), lep2.Py(), lep2.Pz(), TMath::Sqrt(lep2.M()*lep2.M()+lep2.Px()*lep2.Px()+lep2.Py()*lep2.Py()+lep2.Pz()*lep2.Pz()));
  std::vector<NSVfitStandalone::MeasuredTauLepton> measuredTauLeptons;
  measuredTauLeptons.push_back(NSVfitStandalone::MeasuredTauLepton(NSVfitStandalone::kHadDecay, l1));
  measuredTauLeptons.push_back(NSVfitStandalone::MeasuredTauLepton(NSVfitStandalone::kHadDecay, l2));
  // construct the class object from the minimal necesarry information
  NSVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMET, covMETraw, 0);
  // apply customized configurations if wanted (examples are given below)
  //algo.maxObjFunctionCalls(5000);
  algo.addLogM(false);
  //algo.metPower(0.5)
  //algo.fit();
  // integration by VEGAS (same as function algo.integrate() that has been in use when markov chain integration had not yet been implemented)
  //algo.integrateVEGAS();
  // integration by markov chain MC
  algo.integrateMarkovChain();

  if(algo.isValidSolution()){
    lBoson4.SetPtEtaPhiM( algo.pt(), algo.eta(), algo.phi(), algo.getMass());
  }
  else{
   std::cout << "sorry -- status of NLL is not valid [" << algo.isValidSolution() << "]" << std::endl;
   lBoson4.SetPtEtaPhiM( 0.,0,0,0);
  }
  
  measuredTauLeptons.clear(); 
  return   lBoson4 ;
}
