// $Id: CycleCreators.py 344 2012-12-13 13:10:53Z krasznaa $

// Local include(s):
#include "../include/VHTausAnalysis.h"

// External include(s):
#include "../GoodRunsLists/include/TGoodRunsListReader.h"

#include <TMath.h>


ClassImp( VHTausAnalysis );

// define cut names

const std::string VHTausAnalysis::kCutName[ VHTausAnalysis::kNumCuts ] = {
  "BeforeCuts",            // C0
  "JSON",                  // C1
  "Trigger",               // C2
  "MetFilters",            // C3
  "Met",                   // C4
  "Jet",                   // C5
  "Tau",                   // C6
  "TauIsolation",          // C7 
  "Lepton",                // C8
  "LeptonDR",               // C9
  "MassWindow",               // C10
  "Tau21",                 // C11
  "SubjetSingleTag",       // C12
  "SubjetDoubleTag"        // C13
};

VHTausAnalysis::VHTausAnalysis()
   : SCycleBase()
   , m_jetAK4( this )
   , m_jetAK8( this )
   , m_jetAK8Puppi( this )
   , m_genjetAK8( this )
   , m_genjetAK4( this )
   , m_eventInfo( this )
   , m_electron( this )
   , m_muon( this )
   , m_tau( this )
   , m_missingEt( this )
   , m_genParticle( this )
   , m_pileupReweightingTool( this )
   , m_bTaggingScaleTool( this )   
   , m_ScaleFactorTool(  this )
   , m_CorrectionTool( this )
   {

   m_logger << INFO << "Hello!" << SLogger::endmsg;
   SetLogName( GetName() );
   
   // read configuration details from XML file
   // (defaults agree with July 2010 acceptance challenge)
   DeclareProperty( "RecoTreeName",             m_recoTreeName             = "physics" );
   //   DeclareProperty( "OutputTreeName",           m_outputTreeName           = "analysis" );

   /// Declare trees and varibles in trees
 
   channels_.push_back("mutau");
   channels_.push_back("eletau");
   if (m_doTauHad) channels_.push_back("tautau");
   
   
   
   for(unsigned int ch=0;ch< channels_.size();ch++){
     m_outputTreeName_ch_.push_back("");
   }   
     
 

   DeclareProperty( "OutputTreeName_mutau" , m_outputTreeName_ch_[0]);
   DeclareProperty( "OutputTreeName_eletau" , m_outputTreeName_ch_[1]);
   if (m_doTauHad)DeclareProperty( "OutputTreeName_tautau" , m_outputTreeName_ch_[2]);
 

   DeclareProperty( "NtupleLevel",              m_ntupleLevel              = kTau );
   DeclareProperty( "JetAK4Name",               m_jetAK4Name               = "jetAK4" );
   DeclareProperty( "JetAK8Name",               m_jetAK8Name               = "jetAK8" );
   DeclareProperty( "ElectronName",             m_electronName             = "el" );
   DeclareProperty( "MuonName",                 m_muonName                 = "mu" );
   DeclareProperty( "TauName",                  m_tauName                  = "tau" );
   DeclareProperty( "MissingEtName",            m_missingEtName            = "MET" );
   DeclareProperty( "GenParticleName",          m_genParticleName          = "genParticle" );
   
   DeclareProperty( "IsData",                   m_isData                   = false );
   DeclareProperty( "IsSignal",                 m_isSignal                 = false );
   DeclareProperty( "ApplyMETFilters",          m_applyMETFilters          = true );
   DeclareProperty( "m_doNormSyst",             m_doNormSyst               = (m_isData ? false : true) );
   DeclareProperty( "m_doShapeSyst",            m_doShapeSyst              = (m_isData ? false :true) );
   DeclareProperty( "m_doPDFSyst",              m_doPDFSyst                = (m_isData ? false :true ));
   DeclareProperty( "m_doHistNotTree",          m_doHistNotTree            = (m_isData ? false :true ));

   DeclareProperty( "JetPtCut",                 m_jetPtCut           = 120. );
   DeclareProperty( "JetEtaCut",                m_jetEtaCut          = 2.4  );
   DeclareProperty( "AK4JetPtCut",              m_AK4jetPtCut           = 30. );
   DeclareProperty( "AK4JetEtaCut",             m_AK4jetEtaCut          = 3.  );
   DeclareProperty( "MjjCut",                   m_mjjCut             = 600  );
   DeclareProperty( "JetDeltaEtaCut",           m_jetDeltaEtaCut     = 1.3  );

   DeclareProperty( "Tau21HPCut",                m_tau21HPCut             = 0.45 );
   DeclareProperty( "Tau21LPCut",                m_tau21LPCut             = 0.75 );
   DeclareProperty( "MVLowSidebandCut",          m_mVLowSidebandCut       = 0 );
   DeclareProperty( "MWLowerCut",                m_mWLowerCut             = 65 );
   DeclareProperty( "MWUpperCut",                m_mWUpperCut             = 85 );
   DeclareProperty( "MZLowerCut",                m_mZLowerCut             = 85 );
   DeclareProperty( "MZUpperCut",                m_mZUpperCut             = 105 );
   DeclareProperty( "MHLowCut",                 m_mHLowerCut             = 105 );
   DeclareProperty( "MHUpperCut",                m_mHUpperCut             = 135 );
   
   DeclareProperty( "CSVMin",                m_csvMin             = 0.605 );
   DeclareProperty( "m_doTauHad",                m_doTauHad             = true );
   DeclareProperty( "m_doSVFit_new",             m_doSVFit_new    = false );

   DeclareProperty( "ElectronPtCut",                 m_electronPtCut              = 10. );
   DeclareProperty( "ElectronEtaCut",                m_electronEtaCut             = 2.5 );
   
   DeclareProperty( "MuonPtCut",                 m_muonPtCut              = 10. );
   DeclareProperty( "MuonEtaCut",                m_muonEtaCut             = 2.4 );

   DeclareProperty( "TauPtCut",                 m_tauPtCut           = 20 );
   DeclareProperty( "TauEtaCut",                m_tauEtaCut          = 2.3  );
   DeclareProperty( "LeptonPtCut",              m_leptonPtCut        = 10.   );
   DeclareProperty( "LeptonEtaCut",             m_leptonEtaCut       = 2.5  );
   DeclareProperty( "TauIsoCut",                m_tauIsoCut          = true ); 
   
   DeclareProperty( "METCut",                  m_metCut             = 150 );
   DeclareProperty( "FillHist",                m_FillHist             = false );
   
   DeclareProperty( "JSONName",                 m_jsonName             = std::string (std::getenv("SFRAME_DIR")) + "/../GoodRunsLists/JSON/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt");
   DeclareProperty( "PUPPIJEC",                  m_PUPPIJEC_           = std::string (std::getenv("SFRAME_DIR")) + "/../VHTausResonances/weightSD/puppiCorr.root" );
   DeclareProperty( "doGenRecoMatchVariables",                  m_doGenRecoMatchVariables           = true );
  
   
}

VHTausAnalysis::~VHTausAnalysis() {

  m_logger << INFO << "Tschoe!" << SLogger::endmsg;
  if (!m_isData) delete trandom;
}

void VHTausAnalysis::BeginCycle() throw( SError ) {

  m_logger << INFO << "Hello to you!" << SLogger::endmsg;
  
  mu_tau=0;
  ele_tau=0;
  tau_tau=0;

  //

  // Load the GRL:
  //
  if (m_isData) {
    m_logger << INFO << "Loading GoodRunsList from file: " << m_jsonName << SLogger::endmsg;
    Root::TGoodRunsListReader reader( TString( m_jsonName.c_str() ) );
    if( ! reader.Interpret() ) {
      m_logger << FATAL << "Couldn't read in the GRL!" << SLogger::endmsg;
      throw SError( ( "Couldn't read in file: " + m_jsonName ).c_str(), SError::SkipCycle );
    }
    m_grl = reader.GetMergedGoodRunsList();
    m_grl.Summary();
    m_grl.SetName( "MyGoodRunsList" );
  
    //
    // Add it as a configuration object, so that the worker nodes will receive it:
    //
    AddConfigObject( &m_grl );
		
    // m_logger << INFO << "Loading RunEventFilter from file: " << m_runEventFilterName << SLogger::endmsg;
    // m_runEventFilterReader.SetTextFile( TString( m_runEventFilterName.c_str() ) );
    // if( ! m_runEventFilterReader.Interpret() ) {
    //   m_logger << FATAL << "Couldn't read in the RunEventFilter file!" << SLogger::endmsg;
    //   throw SError( ( "Couldn't read in file: " + m_runEventFilterName ).c_str(), SError::SkipCycle );
    // }
		
  }
  
  m_triggerNames.clear();
	
  //Dijet triggers
  m_triggerNames.push_back("AK8PFJet360_TrimMass30") ;
  m_triggerNames.push_back("AK8PFHT700_TrimR0p1PT0p03Mass50") ;
  // // trignames.push_back("AK8DiPFJet280_200_TrimMass30_BTagSV0p45") ;
  m_triggerNames.push_back("PFHT650_WideJetMJJ950DEtaJJ1p5") ;
  m_triggerNames.push_back("PFHT650_WideJetMJJ900DEtaJJ1p5") ;
  m_triggerNames.push_back("PFHT800_v") ;
  m_triggerNames.push_back("PFHT650_v") ;
  m_triggerNames.push_back("PFHT900_v") ;

 

  m_triggerNames.push_back("HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v");
  m_triggerNames.push_back("HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v");
  m_triggerNames.push_back("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v");
  m_triggerNames.push_back("HLT_PFMETNoMu90_JetIdCleaned_PFMHTNoMu90_IDTight_v");
  m_triggerNames.push_back("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");
  m_triggerNames.push_back("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");
  m_triggerNames.push_back("HLT_PFMET110_PFMHT110_IDTight_v");
  m_triggerNames.push_back("HLT_PFMET120_PFMHT120_IDTight_v");
  m_triggerNames.push_back("HLT_PFMET170_NoiseCleaned_v");
  m_triggerNames.push_back("HLT_PFMET170_HBHECleaned_v");
  m_triggerNames.push_back("HLT_PFMET170_HBHE_BeamHaloCleaned_v");
   
  m_triggerNames.push_back("HLT_Ele27_eta2p1_WP75_Gsf_v") ;
  m_triggerNames.push_back("HLT_IsoMu20_v");
  m_triggerNames.push_back("HLT_IsoMu22_v");
  m_triggerNames.push_back("HLT_IsoMu22_eta2p1_v");
  m_triggerNames.push_back("HLT_IsoMu24_v");
  m_triggerNames.push_back("HLT_IsoMu27_v");
  m_triggerNames.push_back("HLT_Ele22_eta2p1_WPLoose_Gsf_v") ;
  m_triggerNames.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_v") ;
  m_triggerNames.push_back("HLT_Ele27_WPTight_Gsf_v") ;
  m_triggerNames.push_back("HLT_Ele115_CaloIdVT_GsfTrkIdT") ;
  m_triggerNames.push_back("HLT_IsoTkMu20_v"); 
  m_triggerNames.push_back("HLT_IsoTkMu22_eta2p1_v");
  m_triggerNames.push_back("HLT_IsoTkMu22_v");
  m_triggerNames.push_back("HLT_IsoTkMu24_v");
  m_triggerNames.push_back("HLT_IsoTkMu27_v");
  m_triggerNames.push_back("HLT_Mu50_v");
  m_triggerNames.push_back("HLT_TkMu50_v");
 


  // set names for various selections
  m_catNames.clear();
  m_catNames.push_back("mutau_NoWindow_NoTau21_SubjetPreTag");
  m_catNames.push_back("eletau_NoWindow_NoTau21_SubjetPreTag");
  m_catNames.push_back("tautau_NoWindow_NoTau21_SubjetPreTag");
 
 
  // mutau channel
  
  m_catNames.push_back("mutau_VWindow_HPTau21_SubjetPreTag");
  m_catNames.push_back("mutau_VWindow_LPTau21_SubjetPreTag");
  m_catNames.push_back("mutau_HWindow_NoTau21_1SubjetTag");
  m_catNames.push_back("mutau_HWindow_NoTau21_2SubjetTag");
 


  // eletau channel
 
  m_catNames.push_back("eletau_VWindow_HPTau21_SubjetPreTag");
  m_catNames.push_back("eletau_VWindow_LPTau21_SubjetPreTag");
  m_catNames.push_back("eletau_HWindow_NoTau21_1SubjetTag");
  m_catNames.push_back("eletau_HWindow_NoTau21_2SubjetTag");
 
 
 // tautau channel

  
  m_catNames.push_back("tautau_VWindow_HPTau21_SubjetPreTag");
  m_catNames.push_back("tautau_VWindow_LPTau21_SubjetPreTag");
  m_catNames.push_back("tautau_HWindow_NoTau21_1SubjetTag");
  m_catNames.push_back("tautau_HWindow_NoTau21_2SubjetTag");
 
 
  

   return;

}

void VHTausAnalysis::EndCycle() throw( SError ) {
   std::cout << "events in mu_tau " <<mu_tau <<std::endl;
   std::cout << "events in ele_tau " <<ele_tau <<std::endl;
   std::cout << "events in tau_tau " <<tau_tau <<std::endl;
   return;

}

void VHTausAnalysis::BeginInputData( const SInputData& id ) throw( SError ) {

  m_logger << INFO << "RecoTreeName:         " <<             m_recoTreeName << SLogger::endmsg;
  //m_logger << INFO << "OutputTreeName:       " <<             m_outputTreeName << SLogger::endmsg;
  m_logger << INFO << "NtupleLevel:          " <<             m_ntupleLevel << SLogger::endmsg;
  m_logger << INFO << "JetAK4Name:           " <<             m_jetAK4Name << SLogger::endmsg;
  m_logger << INFO << "JetAK8Name:           " <<             m_jetAK8Name << SLogger::endmsg;
  m_logger << INFO << "ElectronName:         " <<             m_electronName << SLogger::endmsg;
  m_logger << INFO << "MuonName:             " <<             m_muonName << SLogger::endmsg;
  m_logger << INFO << "TauName:             " <<             m_tauName << SLogger::endmsg;
  m_logger << INFO << "GenParticleName:      " <<             m_genParticleName << SLogger::endmsg;
  m_logger << INFO << "Doing hadronic channel:      " <<             m_doTauHad<< SLogger::endmsg;
  m_logger << INFO << "Doing Norm syst:      " <<             m_doNormSyst<< SLogger::endmsg;
  m_logger << INFO << "Doing Shape syst:      " <<             m_doShapeSyst<< SLogger::endmsg;
  m_logger << INFO << "Doing PDF syst:      " <<             m_doPDFSyst<< SLogger::endmsg;
  m_logger << INFO << "Doing Hist syst:      " <<             m_doHistNotTree<< SLogger::endmsg;
 
  m_logger << INFO << "IsData:           " <<                   (m_isData ? "TRUE" : "FALSE") << SLogger::endmsg;
  m_logger << INFO << "IsSignal:           " <<                 (m_isSignal ? "TRUE" : "FALSE") << SLogger::endmsg;
  m_logger << INFO << "ApplyMETFilters:           " <<          (m_applyMETFilters ? "TRUE" : "FALSE") << SLogger::endmsg;
  
  m_logger << INFO << "JetPtCut:           " <<                 m_jetPtCut << SLogger::endmsg;
  m_logger << INFO << "JetEtaCut:           " <<                m_jetEtaCut << SLogger::endmsg;
  m_logger << INFO << "MjjCut:           " <<                m_mjjCut << SLogger::endmsg;
  m_logger << INFO << "JetDeltaEtaCut:           " <<                m_jetDeltaEtaCut << SLogger::endmsg;

  m_logger << INFO << "Tau21HPCut:           " <<                m_tau21HPCut << SLogger::endmsg;
  m_logger << INFO << "Tau21LPCut:           " <<                m_tau21LPCut << SLogger::endmsg;
  
  m_logger << INFO << "MVLowSidebandCut:     " <<                m_mVLowSidebandCut << SLogger::endmsg;
  m_logger << INFO << "MWLowerCut:           " <<                m_mWLowerCut << SLogger::endmsg;
  m_logger << INFO << "MWUpperCut:           " <<                m_mWUpperCut << SLogger::endmsg;
  m_logger << INFO << "MZLowerCut:           " <<                m_mZLowerCut << SLogger::endmsg;
  m_logger << INFO << "MZUpperCut:           " <<                m_mZUpperCut << SLogger::endmsg;
  m_logger << INFO << "MHLowCut:             " <<                 m_mHLowerCut << SLogger::endmsg;
  m_logger << INFO << "MHUpperCut:           " <<                m_mHUpperCut << SLogger::endmsg;
  
  m_logger << INFO << "CSVMin:           " <<                m_csvMin << SLogger::endmsg;
  
  m_logger << INFO << "ElectronPtCut:           " <<                 m_electronPtCut << SLogger::endmsg;
  m_logger << INFO << "ElectronEtaCut:           " <<                m_electronEtaCut << SLogger::endmsg;
  
  m_logger << INFO << "MuonPtCut:           " <<                 m_muonPtCut << SLogger::endmsg;
  m_logger << INFO << "MuonEtaCut:           " <<                m_muonEtaCut << SLogger::endmsg;
  
  m_logger << INFO << "TauPtCut:           " <<                 m_tauPtCut << SLogger::endmsg;
  m_logger << INFO << "TauEtaCut:           " <<                m_tauEtaCut << SLogger::endmsg;
  m_logger << INFO << "TauIsoCut:           " <<                m_tauIsoCut << SLogger::endmsg;
  
  m_logger << INFO << "LeptonPtCut:           " <<                 m_leptonPtCut << SLogger::endmsg;
  m_logger << INFO << "LeptonEtaCut:           " <<                m_leptonEtaCut << SLogger::endmsg;
  
  m_logger << INFO << "METCut:           " <<                m_metCut << SLogger::endmsg;
  
  m_logger << INFO << "JSONName:           " <<                 m_jsonName << SLogger::endmsg;
  m_logger << INFO << "TrigSF_muonName:           " <<                 m_TrigSF_muonName << SLogger::endmsg;
  m_logger << INFO << "IDSF_muonName:           " <<                 m_IDSF_muonName << SLogger::endmsg;
  m_logger << INFO << "IsoSF_muonName:           " <<                 m_IsoSF_muonName << SLogger::endmsg;
  m_logger << INFO << "IDSF_eleName:           " <<                 m_IDSF_eleName << SLogger::endmsg;


  m_logger<< INFO << "Filling hists:         "<<                m_FillHist << SLogger::endmsg;
  
  if (!m_isData) m_pileupReweightingTool.BeginInputData( id );

  if (m_isData) {
    TObject* grl;
    if( ! ( grl = GetConfigObject( "MyGoodRunsList" ) ) ) {
      m_logger << FATAL << "Can't access the GRL!" << SLogger::endmsg;
      throw SError( "Can't access the GRL!", SError::SkipCycle );
    }
    m_grl = *( dynamic_cast< Root::TGoodRunsList* >( grl ) );
  }
  
 
  b_selection_bits.resize( m_catNames.size() );
  b_selection_lastcut.resize( m_catNames.size() );
  


 
  std::cout << " channels_.size() "<< channels_.size() << std::endl;
    
  for(unsigned int ch=0;ch< channels_.size();ch++){
    // m_outputTreeName_ch_.push_back("tree_"+ channels_[ch]);
    //camilla
    std::cout << " declaring variables for channel " <<  channels_[ch] << " ch " <<ch << std::endl; 
    DeclareVariable(b_weight_[channels_[ch]]              , "weight"                 ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_weightGen_[channels_[ch]]           , "weightGen"              ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_weightGen_corr_[channels_[ch]]      , "weightGen_corr"              ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_weightPU_[channels_[ch]]            , "weightPU"               ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_weightMET_[channels_[ch]]            , "weightMET"               ,  m_outputTreeName_ch_[ch].c_str());
   
    if (m_doNormSyst && !m_doHistNotTree){
      DeclareVariable(b_weightPU_up_[channels_[ch]]         , "weightPU_up"               ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightPU_down_[channels_[ch]]       , "weightPU_down"               ,  m_outputTreeName_ch_[ch].c_str());
    }
    DeclareVariable(b_weightBtag_[channels_[ch]]          , "weightBtag"             ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable(b_weightBtag_veto_[channels_[ch]]      , "weightBtag_veto"             ,  m_outputTreeName_ch_[ch].c_str());
    if (m_doNormSyst && !m_doHistNotTree){
      DeclareVariable(b_weightBtag_bc_up_[channels_[ch]]          , "weightBtag_bc_up"             ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable(b_weightBtag_veto_bc_up_[channels_[ch]]      , "weightBtag_veto_bc_up"             ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightBtag_bc_down_[channels_[ch]]          , "weightBtag_bc_down"             ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable(b_weightBtag_veto_bc_down_[channels_[ch]]      , "weightBtag_veto_bc_down"             ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightBtag_l_up_[channels_[ch]]          , "weightBtag_l_up"             ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable(b_weightBtag_veto_l_up_[channels_[ch]]      , "weightBtag_veto_l_up"             ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightBtag_l_down_[channels_[ch]]          , "weightBtag_l_down"             ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable(b_weightBtag_veto_l_down_[channels_[ch]]      , "weightBtag_veto_l_down"             ,  m_outputTreeName_ch_[ch].c_str());
    }

    DeclareVariable( b_runNumber_[channels_[ch]],           "runNumber"            ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_eventNumber_[channels_[ch]],         "eventNumber"          ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_lumiBlock_[channels_[ch]],           "lumiBlock"            ,  m_outputTreeName_ch_[ch].c_str());
  


    DeclareVariable( b_ak8jet0_pt_[channels_[ch]],           "ak8jet0_pt"         ,  m_outputTreeName_ch_[ch].c_str());

    if( m_doShapeSyst && !m_doHistNotTree){
    DeclareVariable( b_ak8jet0_pt_jesUp_[channels_[ch]],           "ak8jet0_pt_jesUp"         ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_pt_jesDown_[channels_[ch]],           "ak8jet0_pt_jesDown"         ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_pt_jerUp_[channels_[ch]],           "ak8jet0_pt_jerUp"         ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_pt_jerDown_[channels_[ch]],           "ak8jet0_pt_jerDown"         ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_pt_jer_[channels_[ch]],           "ak8jet0_pt_jer"         ,  m_outputTreeName_ch_[ch].c_str());
    }
    DeclareVariable( b_ak8jet0_phi_[channels_[ch]],          "ak8jet0_phi"         ,  m_outputTreeName_ch_[ch].c_str() );
    DeclareVariable( b_ak8jet0_eta_[channels_[ch]],          "ak8jet0_eta"          ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_e_[channels_[ch]],            "ak8jet0_e"            ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_tau21_[channels_[ch]],        "ak8jet0_tau21"        ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_ak8jet0_sd_tau21_[channels_[ch]],        "ak8jet0_sd_tau21"        ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_m_[channels_[ch]],            "ak8jet0_m"            ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_mpruned_[channels_[ch]],      "ak8jet0_mpruned"     ,  m_outputTreeName_ch_[ch].c_str() );   
    DeclareVariable( b_ak8jet0_msoftdrop_[channels_[ch]],      "ak8jet0_msoftdrop"     ,  m_outputTreeName_ch_[ch].c_str() ); 
    DeclareVariable( b_ak8jet0_mpruned_corr_[channels_[ch]],      "ak8jet0_mpruned_corr"     ,  m_outputTreeName_ch_[ch].c_str() );   
    DeclareVariable( b_ak8jet0_msoftdrop_corr_[channels_[ch]],      "ak8jet0_msoftdrop_corr"     ,  m_outputTreeName_ch_[ch].c_str() );
    DeclareVariable( b_ak8jet0_msoftdrop_corr_smear_[channels_[ch]],      "ak8jet0_msoftdrop_corr_smear"     ,  m_outputTreeName_ch_[ch].c_str() );

    if (m_doShapeSyst && !m_doHistNotTree){
      DeclareVariable( b_ak8jet0_msoftdrop_corr_smear_jmsUp_[channels_[ch]],      "ak8jet0_msoftdrop_corr_smear_jmsUp"     ,  m_outputTreeName_ch_[ch].c_str() );
      DeclareVariable( b_ak8jet0_msoftdrop_corr_smear_jmsDown_[channels_[ch]],      "ak8jet0_msoftdrop_corr_smear_jmsDown"     ,  m_outputTreeName_ch_[ch].c_str() ); 
      DeclareVariable( b_ak8jet0_msoftdrop_corr_smear_jmrUp_[channels_[ch]],      "ak8jet0_msoftdrop_corr_smear_jmrUp"     ,  m_outputTreeName_ch_[ch].c_str() );
      DeclareVariable( b_ak8jet0_msoftdrop_corr_smear_jmrDown_[channels_[ch]],      "ak8jet0_msoftdrop_corr_smear_jmrDown"     ,  m_outputTreeName_ch_[ch].c_str() );

    }
    DeclareVariable( b_ak8jet0_csv_[channels_[ch]],          "ak8jet0_csv"         ,  m_outputTreeName_ch_[ch].c_str() );
    DeclareVariable( b_ak8jet0_Hbbtag_[channels_[ch]],          "ak8jet0_Hbbtag"         ,  m_outputTreeName_ch_[ch].c_str() );
    DeclareVariable( b_ak8jet0_subjet01_dr_[channels_[ch]],   "ak8jet0_subjet01_dr"  ,  m_outputTreeName_ch_[ch].c_str() );
    DeclareVariable( b_ak8jet0_subjet01_deta_[channels_[ch]], "ak8jet0_subjet01_deta" ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_subjet01_dphi_[channels_[ch]], "ak8jet0_subjet01_dphi" ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_subjet0_pt_[channels_[ch]],   "ak8jet0_subjet0_pt"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_subjet1_pt_[channels_[ch]],   "ak8jet0_subjet1_pt"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_subjet0_csv_[channels_[ch]],  "ak8jet0_subjet0_csv"  ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_subjet1_csv_[channels_[ch]],  "ak8jet0_subjet1_csv"  ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_puppi_subjet0_csv_[channels_[ch]],  "ak8jet0_puppi_subjet0_csv"  ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_ak8jet0_puppi_subjet1_csv_[channels_[ch]],  "ak8jet0_puppi_subjet1_csv"  ,  m_outputTreeName_ch_[ch].c_str());
  
    DeclareVariable( b_tau_pt_[channels_[ch]],   "tau_pt"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_tau_eta_[channels_[ch]],   "tau_eta"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_tau_phi_[channels_[ch]],   "tau_phi"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_tau_Iso_[channels_[ch]],   "tau_iso"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_tau_IsoRaw_[channels_[ch]],   "tau_iso_raw"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_tau_againstEle_[channels_[ch]],   "tau_againstEle"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_tau_againstMu_[channels_[ch]],   "tau_againstMu"   ,  m_outputTreeName_ch_[ch].c_str());
  

    DeclareVariable( b_seclepton_pt_[channels_[ch]],   "lep2_pt"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_seclepton_eta_[channels_[ch]],   "lep2_eta"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_seclepton_phi_[channels_[ch]],   "lep2_phi"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_seclepton_iso_[channels_[ch]],   "lep2_iso"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_met_[channels_[ch]],   "met"   ,  m_outputTreeName_ch_[ch].c_str());

    if (m_doShapeSyst && !m_doHistNotTree){
      DeclareVariable( b_met_jesUp_[channels_[ch]],   "met_jesUp"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_met_jesDown_[channels_[ch]],   "met_jesDown"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_met_jerUp_[channels_[ch]],   "met_jerUp"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_met_jerDown_[channels_[ch]],   "met_jerDown"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_met_jer_[channels_[ch]],   "met_jer"   ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable( b_met_UncEnUp_[channels_[ch]],   "met_UncEnUp"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_met_UncEnDown_[channels_[ch]],   "met_UncEnDown"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_met_tesUp_[channels_[ch]],   "met_tesUp"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_met_tesDown_[channels_[ch]],   "met_tesDown"   ,  m_outputTreeName_ch_[ch].c_str());
   }
    DeclareVariable( b_met_phi_[channels_[ch]],   "met_phi"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_met_phi_ak4_[channels_[ch]],   "met_phi_ak4"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_vis_mass_ll_[channels_[ch]],   "vis_mass_ll"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_dR_ll_[channels_[ch]],   "dR_ll"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_H_Mass_SVFit_[channels_[ch]],   "H_Mass_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_H_Pt_SVFit_[channels_[ch]],   "H_Pt_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_H_Eta_SVFit_[channels_[ch]],   "H_Eta_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_H_Phi_SVFit_[channels_[ch]],   "H_Phi_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Mass_SVFit_[channels_[ch]],   "X_Mass_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Pt_SVFit_[channels_[ch]],   "X_Pt_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Eta_SVFit_[channels_[ch]],   "X_Eta_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Phi_SVFit_[channels_[ch]],   "X_Phi_SVFit"   ,  m_outputTreeName_ch_[ch].c_str());
 
    if  (m_doSVFit_new){
      DeclareVariable( b_H_Mass_SVFit_new_[channels_[ch]],   "H_Mass_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_H_Pt_SVFit_new_[channels_[ch]],   "H_Pt_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_H_Eta_SVFit_new_[channels_[ch]],   "H_Eta_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_H_Phi_SVFit_new_[channels_[ch]],   "H_Phi_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_X_Mass_SVFit_new_[channels_[ch]],   "X_Mass_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_X_Pt_SVFit_new_[channels_[ch]],   "X_Pt_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_X_Eta_SVFit_new_[channels_[ch]],   "X_Eta_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable( b_X_Phi_SVFit_new_[channels_[ch]],   "X_Phi_SVFit_new"   ,  m_outputTreeName_ch_[ch].c_str());
    }
    DeclareVariable( b_H_Mass_Collinear_[channels_[ch]],   "H_Mass_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());
    if (m_doShapeSyst && !m_doHistNotTree){
      DeclareVariable( b_H_Mass_Collinear_jesUp_[channels_[ch]],   "H_Mass_Collinear_jesUp"   ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable( b_X_Mass_Collinear_jesUp_[channels_[ch]],   "X_Mass_Collinear_jesUp"   ,  m_outputTreeName_ch_[ch].c_str()); 

      DeclareVariable( b_H_Mass_Collinear_jesDown_[channels_[ch]],   "H_Mass_Collinear_jesDown"   ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable( b_X_Mass_Collinear_jesDown_[channels_[ch]],   "X_Mass_Collinear_jesDown"   ,  m_outputTreeName_ch_[ch].c_str()); 

      DeclareVariable( b_H_Mass_Collinear_jerUp_[channels_[ch]],   "H_Mass_Collinear_jerUp"   ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable( b_X_Mass_Collinear_jerUp_[channels_[ch]],   "X_Mass_Collinear_jerUp"   ,  m_outputTreeName_ch_[ch].c_str()); 

      DeclareVariable( b_H_Mass_Collinear_jerDown_[channels_[ch]],   "H_Mass_Collinear_jerDown"   ,  m_outputTreeName_ch_[ch].c_str()); 
      DeclareVariable( b_X_Mass_Collinear_jerDown_[channels_[ch]],   "X_Mass_Collinear_jerDown"   ,  m_outputTreeName_ch_[ch].c_str()); 

     DeclareVariable( b_H_Mass_Collinear_jer_[channels_[ch]],   "H_Mass_Collinear_jer"   ,  m_outputTreeName_ch_[ch].c_str()); 
     DeclareVariable( b_X_Mass_Collinear_jer_[channels_[ch]],   "X_Mass_Collinear_jer"   ,  m_outputTreeName_ch_[ch].c_str()); 

     DeclareVariable( b_H_Mass_Collinear_UncEnUp_[channels_[ch]],   "H_Mass_Collinear_UncEnUp"   ,  m_outputTreeName_ch_[ch].c_str()); 
     DeclareVariable( b_X_Mass_Collinear_UncEnUp_[channels_[ch]],   "X_Mass_Collinear_UncEnUp"   ,  m_outputTreeName_ch_[ch].c_str()); 

     DeclareVariable( b_H_Mass_Collinear_UncEnDown_[channels_[ch]],   "H_Mass_Collinear_UncEnDown"   ,  m_outputTreeName_ch_[ch].c_str()); 
     DeclareVariable( b_X_Mass_Collinear_UncEnDown_[channels_[ch]],   "X_Mass_Collinear_UncEnDown"   ,  m_outputTreeName_ch_[ch].c_str()); 

     DeclareVariable( b_H_Mass_Collinear_tesUp_[channels_[ch]],   "H_Mass_Collinear_tesUp"   ,  m_outputTreeName_ch_[ch].c_str()); 
     DeclareVariable( b_X_Mass_Collinear_tesUp_[channels_[ch]],   "X_Mass_Collinear_tesUp"   ,  m_outputTreeName_ch_[ch].c_str()); 

     DeclareVariable( b_H_Mass_Collinear_tesDown_[channels_[ch]],   "H_Mass_Collinear_tesDown"   ,  m_outputTreeName_ch_[ch].c_str()); 
     DeclareVariable( b_X_Mass_Collinear_tesDown_[channels_[ch]],   "X_Mass_Collinear_tesDown"   ,  m_outputTreeName_ch_[ch].c_str()); 

   
    }

    DeclareVariable( b_H_Pt_Collinear_[channels_[ch]],   "H_Pt_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_H_Eta_Collinear_[channels_[ch]],   "H_Eta_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_H_Phi_Collinear_[channels_[ch]],   "H_Phi_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Mass_Collinear_[channels_[ch]],   "X_Mass_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Pt_Collinear_[channels_[ch]],   "X_Pt_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Eta_Collinear_[channels_[ch]],   "X_Eta_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_X_Phi_Collinear_[channels_[ch]],   "X_Phi_Collinear"   ,  m_outputTreeName_ch_[ch].c_str());

    // DeclareVariable( b_trig_HT800_[channels_[ch]], "trig_HT800"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_HT900_[channels_[ch]], "trig_HT900"   ,  m_outputTreeName_ch_[ch].c_str()); 
    // DeclareVariable( b_trig_HT650_[channels_[ch]], "trig_HT650"   ,  m_outputTreeName_ch_[ch].c_str()); 
    
    DeclareVariable( b_trig_METNoMu90_[channels_[ch]], "trig_METNoMu90"   ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_trig_METNoMu110_[channels_[ch]], "trig_METNoMu110"   ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_trig_METNoMu120_[channels_[ch]], "trig_METNoMu120"   ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_trig_METNoMu90_JetId_[channels_[ch]], "trig_METNoMu90_JetId"   ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_trig_METNoMu110_JetId_[channels_[ch]], "trig_METNoMu110_JetId"   ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_trig_METNoMu120_JetId_[channels_[ch]], "trig_METNoMu120_JetId"   ,  m_outputTreeName_ch_[ch].c_str());
    
    DeclareVariable( b_trig_MET170_[channels_[ch]], "trig_MET170"   ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_trig_MET110_[channels_[ch]], "trig_MET110"   ,  m_outputTreeName_ch_[ch].c_str()); 
    DeclareVariable( b_trig_MET120_[channels_[ch]], "trig_MET120"   ,  m_outputTreeName_ch_[ch].c_str()); 
  


    // DeclareVariable( b_trig_HT650_WideJetMJJ950DEtaJJ1p5_[channels_[ch]], "trig_HT650_WideJetMJJ950DEtaJJ1p5"   ,  m_outputTreeName_ch_[ch].c_str()); 
    // DeclareVariable( b_trig_HT650_WideJetMJJ900DEtaJJ1p5_[channels_[ch]], "trig_HT650_WideJetMJJ900DEtaJJ1p5"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_ak8Jet360_Trim30_[channels_[ch]], "trig_ak8Jet360_Trim30"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Ele27Tight_[channels_[ch]], "trig_Ele27Tight"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Ele27Loose_[channels_[ch]], "trig_Ele27Loose"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Ele22Loose_[channels_[ch]], "trig_Ele22Loose"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Ele115_[channels_[ch]], "trig_Ele115"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Mu20_[channels_[ch]], "trig_Mu20"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Mu22_[channels_[ch]], "trig_Mu22"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Mu22_eta2p1_[channels_[ch]], "trig_Mu22_eta2p1"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Mu24_[channels_[ch]], "trig_Mu24"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Mu27_[channels_[ch]], "trig_Mu27"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_trig_Mu50_[channels_[ch]], "trig_Mu50"   ,  m_outputTreeName_ch_[ch].c_str());
    
    DeclareVariable( b_N_AK4_[channels_[ch]], "N_AK4"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_HT_[channels_[ch]], "HT"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_N_taggedAK4_L_[channels_[ch]], "N_taggedAK4_L"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_N_taggedAK4_M_[channels_[ch]], "N_taggedAK4_M"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_N_taggedAK4_T_[channels_[ch]], "N_taggedAK4_T"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_isBoostedTau_[channels_[ch]], "isBoostedTau"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_PV_N_[channels_[ch]], "PV_N" ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_N_tight_muons_[channels_[ch]], "N_tight_muons" ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_N_tight_electrons_[channels_[ch]], "N_tight_electrons" ,  m_outputTreeName_ch_[ch].c_str());
    
    DeclareVariable( b_N_tight_leptons_[channels_[ch]], "N_tight_leptons" ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_N_loose_muons_[channels_[ch]], "N_loose_muons" ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable( b_N_loose_electrons_[channels_[ch]], "N_loose_electrons" ,  m_outputTreeName_ch_[ch].c_str());

    DeclareVariable( b_GenEvent_Htata_[channels_[ch]], "GenEvent_Htata"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_GenEvent_Ztata_[channels_[ch]], "GenEvent_Ztata"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_GenEvent_Gammatata_[channels_[ch]], "GenEvent_Gammatata"   ,  m_outputTreeName_ch_[ch].c_str()); 
    // DeclareVariable( b_ChannelInt_[channels_[ch]], "ChannelInt"   ,  m_outputTreeName_ch_[ch].c_str());
    //DeclareVariable(b_weightTrigSemi_[channels_[ch]], "weightTrig"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_weightLepID_[channels_[ch]], "weightLepID"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_weightTauID_[channels_[ch]], "weightTauID"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_weightLepIso_[channels_[ch]], "weightLepIso"   ,  m_outputTreeName_ch_[ch].c_str());

    if(m_doNormSyst && !m_doHistNotTree){
      //DeclareVariable(b_weightTrigSemi_up_[channels_[ch]], "weightTrig_up"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightLepID_up_[channels_[ch]], "weightLepID_up"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightLepIso_up_[channels_[ch]], "weightLepIso_up"   ,  m_outputTreeName_ch_[ch].c_str());
      //DeclareVariable(b_weightTrigSemi_down_[channels_[ch]], "weightTrig_down"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightLepID_down_[channels_[ch]], "weightLepID_down"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightLepIso_down_[channels_[ch]], "weightLepIso_down"   ,  m_outputTreeName_ch_[ch].c_str());
    }
    DeclareVariable(b_weightLumi_[channels_[ch]], "weightLumi"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable(b_weightTot_[channels_[ch]], "weightTot"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable(b_weightTot_woBtags_[channels_[ch]], "weightTot_woBtags"   ,  m_outputTreeName_ch_[ch].c_str());
    // DeclareVariable(b_weightTot_woBtagVeto_[channels_[ch]], "weightTot_woBtagVeto"   ,  m_outputTreeName_ch_[ch].c_str()); 
    // DeclareVariable(b_weightTot_woBtagSJ_[channels_[ch]], "weightTot_woBtagSJ"   ,  m_outputTreeName_ch_[ch].c_str());
    if (m_doPDFSyst  && !m_doHistNotTree ){
      DeclareVariable(b_weightFacUp_[channels_[ch]], "weightFacUp"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightFacDown_[channels_[ch]], "weightFacDown"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightRenUp_[channels_[ch]], "weightRenUp"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightRenDown_[channels_[ch]], "weightRenDown"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightFacRenUp_[channels_[ch]], "weightFacRenUp"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightFacRenDown_[channels_[ch]], "weightFacRenDown"   ,  m_outputTreeName_ch_[ch].c_str());
      DeclareVariable(b_weightPDF_rms_[channels_[ch]], "weightPDF_rms"   ,  m_outputTreeName_ch_[ch].c_str());



    }
  
    if (m_doGenRecoMatchVariables){
    DeclareVariable(b_dRG_tau_lep_[channels_[ch]], "dRG_tau_lep"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_dPtRG_tau_[channels_[ch]], "dPtRG_tau"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(  b_dRRG_tau_[channels_[ch]], "dRRG_tau"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_N_hadronicTaus_[channels_[ch]], "N_hadronicTaus"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_N_leptonicTaus_[channels_[ch]], "N_leptonicTaus"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_dPtRG_lepton_[channels_[ch]], "dPtRG_lepton"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(  b_dRRG_lepton_[channels_[ch]], "dRRG_lepton"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_dPtRG_Res_[channels_[ch]], "dPtRG_Res"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_dRRG_Res_[channels_[ch]], "dRRG_Res"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_dPtRG_Boson_qq_[channels_[ch]], "dPtRG_Boson_qq"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_dRRG_Boson_qq_[channels_[ch]], "dRRG_Boson_qq"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_dPtRG_Higgs_tata_[channels_[ch]], "dPtRG_Higgs_tata"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_dRRG_Higgs_tata_[channels_[ch]], "dRRG_Higgs_tata"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(  b_IdentityRG_tau_[channels_[ch]], "IdentityRG_tau"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable( b_IdentityRG_lepton_[channels_[ch]], "IdentityRG_lepton"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_IdentityRG_Boson_qq_[channels_[ch]], "IdentityRG_Boson"   ,  m_outputTreeName_ch_[ch].c_str());
    DeclareVariable(b_IdentityRG_Higgs_tata_[channels_[ch]], "IdentityRG_Higgs"   ,  m_outputTreeName_ch_[ch].c_str());
    }

}


  //
  // Declare the output histograms:
  //
  Book( TH1F( "Events", "Events;weight", 10, -.5, 9.5 ) );
  Book( TH1F( "SumEvents", "SumEvents;weight", 10, -.5, 9.5 ) );
  Book( TH1F( "METFilters", "METFilters", 20, 0.5, 20.5 ));
  Book( TH1F( "JetPt", "JetPt", 2000, 0, 2000));
  Book( TH1F( "Jet_pruned_massCorr", "Jet_pruned_massCorr", 2000, 0, 2000));
  Book( TH1F( "SumEt","SumEt", 2000, 0, 10000));
  Book( TH1F( "HT","HT", 2000, 0, 10000));
  Book( TH1F( "MET","MET", 2000, 0, 10000));
  Book( TH1F( "N_vertex","N_vertex", 2000, 0, 100));
  Book( TH1F( "N_ak4","N_ak4", 50 , -0.5 ,49.5));
  Book( TH1F( "N_ak8","N_ak8", 50 , -0.5 ,49.5));
  Book( TH1F( "PDF_rms","PDF_rms", 3 , -0.5 ,2.5));
  Book( TH1F( "weightFac","weightFac", 3 , -0.5 ,2.5));
  Book( TH1F( "weightRen","weightRen", 3 , -0.5 ,2.5));
  Book( TH1F( "weightFacRen","weightFacRen", 3 , -0.5 ,2.5));


  Book( TH1F( "Cuts_GenReco_mutau", "Cuts_GenReco_mutau", 40, -.5, 39.5 ) );
  Book( TH1F( "Cuts_GenReco_eletau", "Cuts_GenReco_eletau", 40, -.5, 39.5 ) );
  Book( TH1F( "Cuts_GenReco_tautau", "Cuts_GenReco_tautau", 40, -.5, 39.5 ) );



  for (unsigned int s=0;s<m_catNames.size();++s) {
    TString directory = m_catNames[s].c_str();

    //std::cout <<" booking hists in dir  "<< directory << std::endl;
    // cutflow
    if (m_catNames[s].find("NoWindow") != std::string::npos) { 
      Book( TH1F( "cutflow", "cutflow", 20, 0.5, 20.5 ), directory ); 
      
      Book( TH1F( "X_Mass_Collinear_PDF_rms", "X_Mass_Collinear_PDF_rms", 5000 , 0., 5000 ), directory );  
   
    
      Book( TH1F( "X_Mass_Collinear_weightFacUp", "X_Mass_Collinear_weightFacUp", 5000 , 0., 5000 ), directory );  
      Book( TH1F( "X_Mass_Collinear_weightFacDown", "X_Mass_Collinear_weightFacDown", 5000 , 0., 5000 ), directory );
    
      Book( TH1F( "X_Mass_Collinear_weightRenUp", "X_Mass_Collinear_weightRenUp", 5000 , 0., 5000 ), directory );  
      Book( TH1F( "X_Mass_Collinear_weightRenDown", "X_Mass_Collinear_weightRenDown", 5000 , 0., 5000 ), directory );
      Book( TH1F( "X_Mass_Collinear_weightFacRenUp", "X_Mass_Collinear_weightFacRenUp", 5000 , 0., 5000 ), directory );  
      Book( TH1F( "X_Mass_Collinear_weightFacRenDown", "X_Mass_Collinear_weightFacRenDown", 5000 , 0., 5000 ), directory );

      Book( TH1F( "PDF_rms", "PDF_rms", 3, -0.5, 2.5 ), directory );  
      Book( TH1F( "weightFac", "weightFac", 3, -0.5, 2.5 ), directory );  
      Book( TH1F( "weightRen", "weightRen", 3, -0.5, 2.5 ), directory );  
      Book( TH1F( "weightFacRen", "weightFacRen", 3, -0.5, 2.5 ), directory );  
    




      bookTriggerHistos(directory);
      if (m_FillHist)  bookHistograms(directory); 
    }
    
    Book( TH1F( "X_Mass_Collinear", "X_Mass_Collinear", 5000 , 0., 5000 ), directory );  
    Book( TH1F( "H_Mass_Collinear", "H_Mass_Collinear", 200 , 0., 200. ), directory );  
    Book( TH1F( "ak8jet0_msoftdrop_corr_smear", "ak8jet0_msoftdrop_corr_smear", 250 , 0., 250. ), directory );  


    Book( TH1F( "X_Mass_Collinear_jesUp", "X_Mass_Collinear_jesUp", 5000 , 0., 5000. ), directory );
    Book( TH1F( "X_Mass_Collinear_jesDown", "X_Mass_Collinear_jesDown", 5000 , 0., 5000. ), directory );
    Book( TH1F( "X_Mass_Collinear_jer", "X_Mass_Collinear_jer", 5000 , 0., 5000. ), directory );
    Book( TH1F( "X_Mass_Collinear_jerUp", "X_Mass_Collinear_jerUp", 5000 , 0., 5000. ), directory );
    Book( TH1F( "X_Mass_Collinear_jerDown", "X_Mass_Collinear_jerDown", 5000 , 0., 5000. ), directory );
    Book( TH1F( "X_Mass_Collinear_UncEnUp", "X_Mass_Collinear_UncEnUp", 5000 , 0., 5000. ), directory ); 
    Book( TH1F( "X_Mass_Collinear_UncEnDown", "X_Mass_Collinear_UncEnDown", 5000 , 0., 5000. ), directory );
    Book( TH1F( "X_Mass_Collinear_tesUp", "X_Mass_Collinear_tesUp", 5000 , 0., 5000 ), directory );  
    Book( TH1F( "X_Mass_Collinear_tesDown", "X_Mass_Collinear_tesDown", 5000 , 0., 5000 ), directory );  
    
    Book( TH1F( "H_Mass_Collinear_jesUp", "H_Mass_Collinear_jesUp", 250, 0, 250 ), directory );
    Book( TH1F( "H_Mass_Collinear_jesDown", "H_Mass_Collinear_jesDown", 250, 0, 250 ), directory );
    Book( TH1F( "H_Mass_Collinear_jer", "H_Mass_Collinear_jer", 250, 0, 250 ), directory );
    Book( TH1F( "H_Mass_Collinear_jerUp", "H_Mass_Collinear_jerUp", 250, 0, 250 ), directory );
    Book( TH1F( "H_Mass_Collinear_jerDown", "H_Mass_Collinear_jerDown", 250, 0, 250 ), directory );
    Book( TH1F( "H_Mass_Collinear_UncEnUp", "H_Mass_Collinear_UncEnUp", 250, 0, 250 ), directory ); 
    Book( TH1F( "H_Mass_Collinear_UncEnDown", "H_Mass_Collinear_UncEnDown", 250, 0, 250 ), directory );
    Book( TH1F( "H_Mass_Collinear_tesUp", "H_Mass_Collinear_tesUp", 250, 0, 250 ), directory );  
    Book( TH1F( "H_Mass_Collinear_tesDown", "H_Mass_Collinear_tesDown", 250, 0, 250 ), directory );  
  



   


    
    Book( TH1F( "TauSF_syst", "TauSF_syst", 3, -0.5, 2.5 ), directory );  
    Book( TH1F( "Btag_BC_syst", "Btag_BC_syst", 3, -0.5, 2.5 ), directory );  
    Book( TH1F( "Btag_LF_syst", "Btag_LF_syst", 3, -0.5, 2.5 ), directory );  
    Book( TH1F( "LepID_syst", "LepID_syst", 3, -0.5, 2.5 ), directory );  
    Book( TH1F( "Wtag_PtExtr_syst", "Wtag_PtExtr_syst", 3, -0.5, 2.5 ), directory );  
    
    Book( TH1F( "ak8jet0_msoftdrop_corr_smear_jmsUp", "ak8jet0_msoftdrop_corr_smear_jmsUp", 250 , 0., 250. ), directory );  
    Book( TH1F( "ak8jet0_msoftdrop_corr_smear_jmsDown", "ak8jet0_msoftdrop_corr_smear_jmsDown", 250 , 0., 250. ), directory );  
    Book( TH1F( "ak8jet0_msoftdrop_corr_smear_jmrUp", "ak8jet0_msoftdrop_corr_smear_jmrUp", 250 , 0., 250. ), directory );  
    Book( TH1F( "ak8jet0_msoftdrop_corr_smear_jmrDown", "ak8jet0_msoftdrop_corr_smear_jmrDown", 250 , 0., 250. ), directory );  
    
    
    
    //std::cout <<" done booking hists in dir  "<< directory << std::endl;
    
    
    
    
    //if (m_FillHist)  bookHistograms(directory); 
  
  }
  // std::cout <<" beginning btagging  scale tool " <<std::endl;
  // b-tagging tool initialisation
  
  m_bTaggingScaleTool.BeginInputData( id );
  m_ScaleFactorTool.BeginInputData( id );

  initPuppiSDWeight();
 
  m_CorrectionTool.BeginInputData( id );
  
  if (m_isSignal) {
    // b-tagging efficiencies
    initPuppiSD_ScaleAndRes_Weight();
    m_bTaggingScaleTool.bookHistograms();
  }
  
 
  std::cout <<" done booking "<< std::endl;
 
  return;

}




void VHTausAnalysis::EndInputData( const SInputData& ) throw( SError ) {

  //
  // Final analysis of cut flow
  //
  
  TString defaultCutflow = m_catNames[m_catNames.size()-m_catNames.size()]; //"VWindow_Tau21HP_SubjetDoubleTag";
  m_logger << INFO << "cut flow for " << defaultCutflow << SLogger::endmsg;
  m_logger << INFO << Form( "Cut\t%25.25s\tEvents\tRelEff\tAbsEff", "Name" ) << SLogger::endmsg;
  
  Double_t ntot = Hist( "cutflow", defaultCutflow )->GetBinContent( 1 );
  m_logger << INFO << Form( "\t%25.25s\t%6.0f", "start", ntot ) << SLogger::endmsg;
  for( Int_t ibin = 2; ibin <= kNumCuts; ++ibin ) {
    Int_t    icut    = ibin - 1;
    Double_t nevents = Hist( "cutflow", defaultCutflow )->GetBinContent( ibin );
    Double_t releff  = 100. * nevents / Hist( "cutflow", defaultCutflow )->GetBinContent( ibin-1 );
    Double_t abseff  = 100. * nevents / ntot;
    m_logger << INFO  << Form( "C%d\t%25.25s\t%6.0f\t%6.2f\t%6.2f", icut-1, kCutName[icut].c_str(), nevents, releff, abseff ) << SLogger::endmsg;
  }
  


  return;
}

void VHTausAnalysis::BeginInputFile( const SInputData& ) throw( SError ) {

  m_logger << INFO << "Connecting input variables" << SLogger::endmsg;
  if (m_isData) {
    m_jetAK4.ConnectVariables(       m_recoTreeName.c_str(), Ntuple::JetBasic|Ntuple::JetAnalysis, (m_jetAK4Name + "_").c_str() );
    m_jetAK8.ConnectVariables(       m_recoTreeName.c_str(), Ntuple::JetBasic|Ntuple::JetAnalysis|Ntuple::JetSubstructure|Ntuple::JetSoftdropSubjets, (m_jetAK8Name + "_").c_str() );
    m_jetAK8Puppi.ConnectVariables(  m_recoTreeName.c_str(),  Ntuple::PuppiJetBasic|Ntuple::PuppiJetAnalysis|Ntuple::PuppiJetSoftdropSubjets,"");
    m_eventInfo.ConnectVariables(    m_recoTreeName.c_str(), // Ntuple::EventInfoBasic|
				     Ntuple::EventInfoTrigger|Ntuple::EventInfoMETFilters|Ntuple::EventInfoMuonFilters, "" );
  }
  else {
    m_jetAK4.ConnectVariables(       m_recoTreeName.c_str(), Ntuple::JetBasic|Ntuple::JetAnalysis|Ntuple::JetTruth|Ntuple::JetJER, (m_jetAK4Name + "_").c_str() );
    m_jetAK8.ConnectVariables(       m_recoTreeName.c_str(), Ntuple::JetBasic|Ntuple::JetAnalysis|Ntuple::JetSubstructure|Ntuple::JetTruth|Ntuple::JetSoftdropSubjets|Ntuple::JetSoftdropSubjetsTruth|Ntuple::JetJER, (m_jetAK8Name + "_").c_str() );
    m_genjetAK8.ConnectVariables(       m_recoTreeName.c_str(), Ntuple::GenJetTruth,"");
    m_genjetAK4.ConnectVariables(       m_recoTreeName.c_str(), Ntuple::GenJetak4Truth,"");
    m_jetAK8Puppi.ConnectVariables(   m_recoTreeName.c_str(), Ntuple::PuppiJetBasic|Ntuple::PuppiJetAnalysis|Ntuple::PuppiJetSoftdropSubjets,"" );
    if (m_doPDFSyst){ 
      m_eventInfo.ConnectVariables(    m_recoTreeName.c_str(), Ntuple::EventInfoBasic|Ntuple::EventInfoTrigger|Ntuple::EventInfoMETFilters|Ntuple::EventInfoTruth|Ntuple::EventInfoPDF, "" );
    }else{ 
      m_eventInfo.ConnectVariables(    m_recoTreeName.c_str(), Ntuple::EventInfoBasic|Ntuple::EventInfoTrigger|Ntuple::EventInfoMETFilters|Ntuple::EventInfoTruth, "" );}

    m_genParticle.ConnectVariables(  m_recoTreeName.c_str(), Ntuple::GenParticleBasic, (m_genParticleName + "_").c_str() );
  }
  m_electron.ConnectVariables(     m_recoTreeName.c_str(), Ntuple::ElectronBasic|Ntuple::ElectronID|Ntuple::ElectronBoostedID|Ntuple::ElectronBoostedIsolation| Ntuple::ElectronSuperCluster, (m_electronName + "_").c_str() );
  m_muon.ConnectVariables(         m_recoTreeName.c_str(), Ntuple::MuonBasic|Ntuple::MuonID|Ntuple::MuonIsolation| Ntuple::MuonBoostedIsolation, (m_muonName + "_").c_str() );
  m_tau.ConnectVariables(         m_recoTreeName.c_str(), Ntuple::TauBasic|Ntuple::TauID|Ntuple::TauAdvancedID, (m_tauName + "_").c_str() );
  
  if( m_doShapeSyst){
    m_missingEt.ConnectVariables(    m_recoTreeName.c_str(), Ntuple::MissingEtBasic|Ntuple::MissingEtAnalysis|Ntuple::MissingEtAnalysisSyst , (m_missingEtName + "_").c_str() );
  }else{
    m_missingEt.ConnectVariables(    m_recoTreeName.c_str(), Ntuple::MissingEtBasic|Ntuple::MissingEtAnalysis , (m_missingEtName + "_").c_str() );
  }

  m_logger << INFO << "Connecting input variables completed" << SLogger::endmsg;

   return;

}

void VHTausAnalysis::ExecuteEvent( const SInputData&, Double_t ) throw( SError ) {

  m_logger << VERBOSE << "ExecuteEvent" << SLogger::endmsg;
  // std::cout << "ExecuteEvent" << std::endl;

  clearBranches();
  
  b_eventNumber = m_eventInfo.eventNumber;
  b_runNumber = m_eventInfo.runNumber;
  b_lumiBlock = m_eventInfo.lumiBlock;
  
  std::vector<TBits> selectionBits(m_catNames.size(), TBits(kNumCuts));
  for (unsigned int s=0;s<m_catNames.size();++s) {
    selectionBits[s].SetBitNumber( kBeforeCuts );
  }
  bool moveOn = true;
  
  // Cut 1: check for data if run/lumiblock in JSON
  if (m_isData) {
    if (isGoodEvent(m_eventInfo.runNumber, m_eventInfo.lumiBlock)) {
      for (unsigned int s=0;s<m_catNames.size();++s) {
        selectionBits[s].SetBitNumber( kJSON );
      }
    }
  }
  else {
    for (unsigned int s=0;s<m_catNames.size();++s) {
      selectionBits[s].SetBitNumber( kJSON );
    }
  }
  /// gen Weigth for MC(

  if (!m_isData) {
    b_weight= getEventWeight();
  }
  /// gen Lumi  Weigth for MC and data(data weight =1)
  TString File_at_the_moment = TString(this->GetHistInputFile()->GetName());	           
  b_weightLumi = m_LumiWeight.getLumiWeight( File_at_the_moment );
  //std::cout << "Weigth Lumi"<<b_weightLumi<<  std::endl; 
  ///genFilter to identify Z-tata events not cutting away
  if (!m_isData){
    genFilterZtautau();
  }
 // Rest of book-keeping
  Hist( "Events" )->Fill( 0., b_weightGen        ); // event with MC weight
  Hist( "Events" )->Fill( 2,  b_weightPU         ); // event pileup weight
  Hist( "Events" )->Fill( 9,  1                  ); // event without MC weight
  if ( GenEvent_Ztata_filter) Hist( "Events" )->Fill( 8,  1                  ); // event tata  without MC weight
  if ( GenEvent_Gammatata_filter) Hist( "Events" )->Fill( 7,  1                  ); // event tata  without MC weight
  if ( GenEvent_Htata_filter) Hist( "Events" )->Fill( 6,  1                  ); // event tata  without MC weight
  Hist( "SumEvents" )->Fill( 0., fabs(b_weightGen)       ); // event with MC weight
  Hist( "SumEvents" )->Fill( 2,  fabs(b_weightPU)       ); // event pileup weight
  Hist( "SumEvents" )->Fill( 9,  1                        ); // event without MC weight
  
  Hist( "PDF_rms" )->Fill(0. ,  1. ) ;
  Hist( "PDF_rms" )->Fill(1 ,   m_eventInfo.PDF_rms     );
  
  Hist( "weightFac" )->Fill(0. ,  1. ) ;
  Hist( "weightFac" )->Fill(1. , m_eventInfo.weightFacUp ) ;
  Hist( "weightFac" )->Fill(2. , m_eventInfo.weightFacDown ) ;

  Hist( "weightRen" )->Fill(0. ,  1. ) ;
  Hist( "weightRen" )->Fill(1. , m_eventInfo.weightRenUp ) ;
  Hist( "weightRen" )->Fill(2. , m_eventInfo.weightRenDown ) ;

  Hist( "weightFacRen" )->Fill(0. ,  1. ) ;
  Hist( "weightFacRen" )->Fill(1. , m_eventInfo.weightFacRenUp ) ;
  Hist( "weightFacRen" )->Fill(2. , m_eventInfo.weightFacRenDown ) ;



  // Cut 2: pass trigger
  //if (m_isData) checkTriggerTurnOn();

  if (passTrigger()) {
    m_logger << VERBOSE << "Trigger pass" << SLogger::endmsg;
    for (unsigned int s=0;s<m_catNames.size();++s) {
      selectionBits[s].SetBitNumber( kTrigger );
    }
    // m_logger << INFO << "pass: " << selectionBits[0].TestBitNumber( kTrigger ) << SLogger::endmsg;
  }
  else {
    m_logger <<  VERBOSE  << "Trigger fail:" << selectionBits[0].TestBitNumber( kTrigger ) << SLogger::endmsg;
   
    if (m_isSignal || !m_isData){
      for (unsigned int s=0;s<m_catNames.size();++s) {
	selectionBits[s].SetBitNumber( kTrigger );
      }
    }
   
  }
  
  // Cut 3: pass MET filters

  if (passMETFilters()) { 
      
    m_logger << VERBOSE << "passMETFilters" << SLogger::endmsg;
    for (unsigned int s=0;s<m_catNames.size();++s) {
      selectionBits[s].SetBitNumber( kMetFilters );
	
    }
    
 
  }

  TString File_at_the_moment_sig = TString(this->GetHistInputFile()->GetName());
  if (!m_isData && (File_at_the_moment_sig.Contains("Radion") || File_at_the_moment_sig.Contains("Wprime") || File_at_the_moment_sig.Contains("Zprime")|| File_at_the_moment_sig.Contains("BulkGrav")  )){
    PairingGenRecoLevel();
  }



  // Cut 4: select two fat jets
  std::vector<UZH::Jet> goodFatJets;


  bool already_doneshape_jet=false;
  for ( int i = 0; i < (m_jetAK8.N) ; ++i ) {
    UZH::Jet myjet( &m_jetAK8, i );
    if (myjet.pt() > m_jetPtCut) {
      if (fabs(myjet.eta()) < m_jetEtaCut) {

	 // std::cout<<" myjet.pt() " << myjet.pt() <<" myjet.eta() " << myjet.eta() << " myjet.IDTight() " << myjet.IDTight() << 
	 //   " myjet.pruned_massCorr() " << myjet.pruned_massCorr() <<std::endl;

        if (myjet.IDTight()) {
          goodFatJets.push_back(myjet);
	  
	  if(!m_isData){
	    bool isGenJetMatched= false;
	    float dR_genMatch=0.4;
	    float  dPt_genMatch = 3* myjet.pt() *myjet.jer_sigma_pt();
	    TLorentzVector GenJet_tlv;
	    GenJet_tlv.SetPtEtaPhiE(0,0,0,0);

	    for ( unsigned int j = 0; j < (*m_genjetAK8.genJetAK8_pt).size(); ++j ) {
	      UZH::GenJet mygenjet( &m_genjetAK8, j );
	       TLorentzVector genjet;
	       genjet.SetPtEtaPhiE(  mygenjet.genJetAK8_pt(),  mygenjet.genJetAK8_eta(),  mygenjet.genJetAK8_phi(),  mygenjet.genJetAK8_e() );
   
	      if (genjet.DeltaR( myjet.tlv() ) < dR_genMatch) {
		if (fabs(myjet.pt()-genjet.Pt()) < dPt_genMatch) {
		  isGenJetMatched = true;
		  dR_genMatch=genjet.DeltaR( myjet.tlv() );
		  GenJet_tlv.SetPtEtaPhiE(genjet.Pt(),genjet.Eta(),genjet.Phi(),genjet.E() );
		}      
	      }
	      
	    }
	    
	    if (m_doShapeSyst && !already_doneshape_jet ){
	      already_doneshape_jet=true;

	      //std::cout << "Initial pt" <<  myjet.tlv().Pt() <<std::endl;
	      goodJetsAK8_jesUp = m_CorrectionTool.GetCorrectedJet(myjet.tlv(), float(myjet.jec()), float(myjet.jecUp()), float(myjet.jecDown()), !m_isData, +1, 0, true);
	      goodJetsAK8_jesDown = m_CorrectionTool.GetCorrectedJet(myjet.tlv(), float(myjet.jec()), float(myjet.jecUp()), float(myjet.jecDown()), !m_isData, -1, 0, true);
	      trandom = new TRandom3();
	      float random=trandom->Gaus(0., float(myjet.jer_sigma_pt()));

	      goodJetsAK8_jerUp = m_CorrectionTool.GetCorrectedJetJER_v2(myjet.tlv(), GenJet_tlv, float(myjet.jer_sigma_pt()), float(myjet.jer_sf()), float(myjet.jer_sf_up()),float(myjet.jer_sf_down()), !m_isData,  +1,true, isGenJetMatched,random);
	      goodJetsAK8_jerDown = m_CorrectionTool.GetCorrectedJetJER_v2(myjet.tlv(), GenJet_tlv, float(myjet.jer_sigma_pt()), float(myjet.jer_sf()), float(myjet.jer_sf_up()),float(myjet.jer_sf_down()), !m_isData,  -1,true, isGenJetMatched,random);
	      goodJetsAK8_jer = m_CorrectionTool.GetCorrectedJetJER_v2(myjet.tlv(), GenJet_tlv, float(myjet.jer_sigma_pt()), float(myjet.jer_sf()), float(myjet.jer_sf_up()),float(myjet.jer_sf_down()), !m_isData,  0,true, isGenJetMatched,random);
	    }
	  }


          // fill output variables
          if (goodFatJets.size() == 1) {
            b_ak8jet0_pt = myjet.pt();
            b_ak8jet0_phi = myjet.phi();
            b_ak8jet0_eta = myjet.eta();
            b_ak8jet0_e = myjet.e();
            double tau21 = -1;
            if (myjet.tau1() != 0) {
              tau21 = myjet.tau2() / myjet.tau1();
            }
            b_ak8jet0_tau21 = tau21;
            b_ak8jet0_m = myjet.m();
            b_ak8jet0_mpruned = myjet.pruned_massCorr();
            b_ak8jet0_csv = myjet.csv();
            if (myjet.subjet_softdrop_N() >= 2) {
              double deta = fabs(myjet.subjet_softdrop_eta()[0] - myjet.subjet_softdrop_eta()[1]);
              double dphi = fabs(myjet.subjet_softdrop_phi()[0] - myjet.subjet_softdrop_phi()[1]);
              double dr = sqrt(deta*deta + dphi*dphi);
              b_ak8jet0_subjet01_dr = dr;
              b_ak8jet0_subjet01_deta = deta;
              b_ak8jet0_subjet01_dphi = dphi;
              b_ak8jet0_subjet0_pt = myjet.subjet_softdrop_pt()[0];
              b_ak8jet0_subjet1_pt = myjet.subjet_softdrop_pt()[1];
              b_ak8jet0_subjet0_csv = myjet.subjet_softdrop_csv()[0];
              b_ak8jet0_subjet1_csv = myjet.subjet_softdrop_csv()[1];
            }
          }
          else if (goodFatJets.size() == 2) {
            b_ak8jet1_pt = myjet.pt();
            b_ak8jet1_phi = myjet.phi();
            b_ak8jet1_eta = myjet.eta();
            b_ak8jet1_e = myjet.e();
            double tau21 = -1;
            if (myjet.tau1() != 0) {
              tau21 = myjet.tau2() / myjet.tau1();
            }
            b_ak8jet1_tau21 = tau21;
            b_ak8jet1_m = myjet.m();
            b_ak8jet1_mpruned = myjet.pruned_massCorr();
            b_ak8jet1_csv = myjet.csv();
            if (myjet.subjet_softdrop_N() >= 2) {
              double deta = fabs(myjet.subjet_softdrop_eta()[0] - myjet.subjet_softdrop_eta()[1]);
              double dphi = fabs(myjet.subjet_softdrop_phi()[0] - myjet.subjet_softdrop_phi()[1]);
              double dr = sqrt(deta*deta + dphi*dphi);
              b_ak8jet1_subjet01_dr = dr;
              b_ak8jet1_subjet01_deta = deta;
              b_ak8jet1_subjet01_dphi = dphi;
              b_ak8jet1_subjet0_pt = myjet.subjet_softdrop_pt()[0];
              b_ak8jet1_subjet1_pt = myjet.subjet_softdrop_pt()[1];
              b_ak8jet1_subjet0_csv = myjet.subjet_softdrop_csv()[0];
              b_ak8jet1_subjet1_csv = myjet.subjet_softdrop_csv()[1];
            }
          }
        }
      }
    }
  }


  if (goodFatJets.size() >= 1) {
    for (unsigned int s=0;s<m_catNames.size();++s) {
      selectionBits[s].SetBitNumber( kJet );
      if (m_catNames[s].find("NoWindow") != std::string::npos) {
	selectionBits[s].SetBitNumber( kMassWindow );
      }
    }
  }
  else {
    // can only continue with at least 1 selected fat jet
    moveOn = false;
  }
  
  m_logger << VERBOSE << "kJet" << SLogger::endmsg;
  
  int goodFatJet1Index = -1;

  
  bool foundJet=false;
  UZH::Jet Jet;
  m_logger << VERBOSE << "kFatJetsDeltaEta" << SLogger::endmsg;
  

  
  if(goodFatJets.size() >0)	 {
    Jet = goodFatJets[0];
    
    if (m_isSignal) {
      std::vector<UZH::Jet> selectedJets;
      selectedJets.push_back(Jet);
      m_bTaggingScaleTool.fillEfficiencies(selectedJets);
      m_bTaggingScaleTool.fillSoftdropSubjetEfficiencies(selectedJets);
    }
    // std::cout << " goodFatJets pt " << goodFatJets[0].pt()<< " ;goodFatJets[0].pruned_massCorr() " <<  goodFatJets[0].pruned_massCorr() << " ; eta " << goodFatJets[0].eta() <<" ; phi "<<  goodFatJets[0].phi() << std::endl;
    foundJet=true;
    for (unsigned int s=0;s<m_catNames.size();++s) {
      if (m_catNames[s].find("NoWindow") != std::string::npos) { 
	selectionBits[s].SetBitNumber( kJet );//kMet
      }
    }

  }
  else { 
    throw SError( SError::SkipEvent );  
    return ;
  }
  //} ///close loop goodFatJet
  
  
    
    m_logger << VERBOSE << "kMassWindow" << SLogger::endmsg;

    // m_logger << VERBOSE << Jet << SLogger::endmsg;    
    
    // if (moveOn==false) return;
    //commented 

    // Cut 9: pass tau21 for V-candidate jet
    //found at least 1 jet Vlow, v, H or HHigh
      
    if (Jet.tau1() != 0) {
      double tau21 = Jet.tau2() / Jet.tau1();
      m_logger << VERBOSE << tau21 << SLogger::endmsg;
      if (tau21 < m_tau21HPCut) {
	for (unsigned int s=0;s<m_catNames.size();++s) {
	  if ((m_catNames[s].find("Tau21HP") != std::string::npos)) {
	    m_logger << VERBOSE << m_catNames[s] << SLogger::endmsg;
	    selectionBits[s].SetBitNumber( kTau21 );
	  }
	}
      }
      else if ((tau21 >= m_tau21HPCut) && (tau21 < m_tau21LPCut)) {
	for (unsigned int s=0;s<m_catNames.size();++s) {
	  if ((m_catNames[s].find("Tau21LP") != std::string::npos)) {
	    m_logger << VERBOSE << m_catNames[s] << SLogger::endmsg;
	    selectionBits[s].SetBitNumber( kTau21 );
	  }
	}
      }
    }
    for (unsigned int s=0;s<m_catNames.size();++s) {
      if (m_catNames[s].find("NoTau21") != std::string::npos) {
	selectionBits[s].SetBitNumber( kTau21 );
      }
    }
    
    m_logger << VERBOSE << "kTau21" << SLogger::endmsg;
    
    // Cut 10: require at least one of the subjets from the Higgs jet to be b-tagged
    // Cut 11: require two subjets from the Higgs jet to be b-tagged
    // count number of b-tagged subjets
    int nTaggedSubjets = 0;
    for (int i = 0; i < Jet.subjet_softdrop_N(); ++i) {
      if (m_bTaggingScaleTool.isTagged(Jet.subjet_softdrop_csv()[i])) {
	++nTaggedSubjets;
      }
    }
    for (unsigned int s=0;s<m_catNames.size();++s) {
      if (m_catNames[s].find("NoTag") != std::string::npos) {
	if (nTaggedSubjets == 0) {
	  selectionBits[s].SetBitNumber( kSubjetSingleTag );
	  selectionBits[s].SetBitNumber( kSubjetDoubleTag );
	}
      }
      else if (m_catNames[s].find("PreTag") != std::string::npos) {
	selectionBits[s].SetBitNumber( kSubjetSingleTag );
	selectionBits[s].SetBitNumber( kSubjetDoubleTag );
      }
      else if (m_catNames[s].find("SingleTagExcl") != std::string::npos) {
	if (nTaggedSubjets == 1) {
	  selectionBits[s].SetBitNumber( kSubjetSingleTag );
	  selectionBits[s].SetBitNumber( kSubjetDoubleTag );
	}
      }
      else if (m_catNames[s].find("SingleTagIncl") != std::string::npos) {
	if (nTaggedSubjets >= 1) {
	  selectionBits[s].SetBitNumber( kSubjetSingleTag );
	  selectionBits[s].SetBitNumber( kSubjetDoubleTag );
	}
      }
      else if (nTaggedSubjets >= 1) {
	// these will all be DoubleTag
	selectionBits[s].SetBitNumber( kSubjetSingleTag );
	if (nTaggedSubjets >= 2) {
	  selectionBits[s].SetBitNumber( kSubjetDoubleTag );
	}
      }
    }
    TLorentzVector diJet = Jet.tlv() // + higgsJet.tlv()
      ;
   
    // selection done
    m_logger << VERBOSE << "selection done" << SLogger::endmsg;
    if (!m_isData) {
      b_weight= getEventWeight();
      std::vector<UZH::Jet> selectedJets;
      selectedJets.push_back(Jet);
    
      // b_weightBtag = m_bTaggingScaleTool.getScaleFactor(selectedJets); // event b-tag SF weight
      b_weightBtag = m_bTaggingScaleTool.getSoftdropSubjetScaleFactor(selectedJets); // event b-tag SF weight
      if (m_doNormSyst ){
	int sigma_bc = 1 , sigma_udsg = 1;
	b_weightBtag_bc_up = m_bTaggingScaleTool.getSoftdropSubjetScaleFactor(selectedJets,  sigma_bc, 0);
	b_weightBtag_l_up = m_bTaggingScaleTool.getSoftdropSubjetScaleFactor(selectedJets,  0 ,sigma_udsg);
	sigma_bc = -1 ; sigma_udsg = -1;
	b_weightBtag_bc_down = m_bTaggingScaleTool.getSoftdropSubjetScaleFactor(selectedJets,  sigma_bc, 0);
	b_weightBtag_l_down = m_bTaggingScaleTool.getSoftdropSubjetScaleFactor(selectedJets,  0 ,sigma_udsg);
      }


      //std::cout <<" Subjet Btag SF weight "<< b_weightBtag << std::endl;
      b_weight *= b_weightBtag;
    }
  
     
  
  
    ///} // 1 fat jets
  
  // Rest of book-keeping
  // Hist( "Events" )->Fill( 0., b_weightGen        ); // event with MC weight
    Hist( "Events" )->Fill( 1,  b_weight           ); // event total weight
    // Hist( "Events" )->Fill( 2,  b_weightPU         ); // event pileup weight
    Hist( "Events" )->Fill( 3,  b_weightBtag ); // event b-tag SF weight
    // // Hist( "Events" )->Fill( 3,  b_weight_elScale   ); // event electron SF weight
    // // Hist( "Events" )->Fill( 4,  b_weight_muScale   ); // event muon SF weight
    //  Hist( "Events" )->Fill( 9, b_weightGen                  ); // event without MC weight
    // Hist( "SumEvents" )->Fill( 0., fabs(b_weightGen)       ); // event with MC weight
    Hist( "SumEvents" )->Fill( 1,  fabs(b_weight)          ); // event total weight
    // Hist( "SumEvents" )->Fill( 2,  fabs(b_weightPU)       ); // event pileup weight
    Hist( "SumEvents" )->Fill( 3,  fabs(b_weightBtag)); // event b-tag SF weight
    Hist( "SumEvents" )->Fill( 4,  fabs(b_weightBtag_veto)); // event b-tag SF weight
  // // Hist( "SumEvents" )->Fill( 3,  fabs(b_weight_elScale)  ); // event electron SF weight
  // // Hist( "SumEvents" )->Fill( 4,  fabs(b_weight_muScale)  ); // event muon SF weight
  // // Hist( "SumEvents" )->Fill( 6,  fabs(b_weight_jvfScale));  // event JVF SF weight
  // Hist( "SumEvents" )->Fill( 9,  1                       ); // event without MC weight
  
 
  
  bool foundTau=false;
  std:: vector<bool> isBoosted_taus; 
  std::vector<UZH::Tau> goodTaus;
  for ( int i = 0; i <   (m_tau.N)
	  ; ++i ) {
    UZH::Tau mytau( &m_tau, i );
    if (mytau.pt() > m_tauPtCut && mytau.TauType() ==2 ){
    
      if (fabs(mytau.eta()) < m_tauEtaCut){

	//there is at least a  tau
	for (unsigned int s=0;s<m_catNames.size();++s) {
	  if  (goodTaus.size()<1){
	    if (m_catNames[s].find("mutau") != std::string::npos || m_catNames[s].find("eletau") != std::string::npos|| m_catNames[s].find("tautau") != std::string::npos ) { 
   
	      selectionBits[s].SetBitNumber( kTau );

	    }
	  }
	}
      
	// std::cout<<" mytau.pt() " << mytau.pt() <<" mytau.eta() " << mytau.eta() <<" mytau.TauType() " <<  mytau.TauType() <<std::endl;

//if it's isolated i save it
	if( mytau.byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5  // && ( (goodFatJets.size() >0 && mytau.tlv().DeltaR(goodFatJets[0].tlv())>0.8 ) || goodFatJets.size() <1)
	    // && mytau.againstElectronVLooseMVA6>0.5
	    ){
	  // std::cout<<"Boosted  mytau.pt() " << mytau.pt() <<" mytau.eta() " << mytau.eta() <<" mytau.phi() " << mytau.phi() <<" mytau.TauType() " <<  mytau.TauType() <<std::endl;
	  goodTaus.push_back(mytau);
	  isBoosted_taus.push_back(true);
	  foundTau=true;
	}
      }
    }
  }
  
  for ( int i = 0; i <   (m_tau.N); ++i ) {
    UZH::Tau mytau( &m_tau, i );
    if (mytau.pt() > m_tauPtCut && mytau.TauType()==1 ){
      
      if (fabs(mytau.eta()) < m_tauEtaCut){
	//there is at least a  tau
	for (unsigned int s=0;s<m_catNames.size();++s) {
	  if  (goodTaus.size()<1){
	    if (m_catNames[s].find("mutau") != std::string::npos || m_catNames[s].find("eletau") != std::string::npos|| m_catNames[s].find("tautau") != std::string::npos ) { 
   
	      selectionBits[s].SetBitNumber( kTau );

	    }
	  }
	}
	//	std::cout<<"Standard mytau.pt() " << mytau.pt() <<" mytau.eta() " << mytau.eta() <<" mytau.TauType() " <<  mytau.TauType() <<std::endl;
     
	//if it's isolated i save it
	if( mytau.byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5   // &&( (goodFatJets.size() >0 && mytau.tlv().DeltaR(goodFatJets[0].tlv())>0.8 ) || goodFatJets.size() <1)
	    // && mytau.againstElectronVLooseMVA5()>0.5
	    ){
	  //std::cout<<"Standard  mytau.pt() " << mytau.pt() <<" mytau.eta() " << mytau.eta() <<" mytau.phi() " << mytau.phi() <<" mytau.TauType() " <<  mytau.TauType() <<std::endl;

	  goodTaus.push_back(mytau);
	  isBoosted_taus.push_back(false);
	  foundTau=true;
	}
      }
    }
  }
  

 
 
 
  for (unsigned int s=0;s<m_catNames.size();++s) {
    if (m_catNames[s].find("mutau") != std::string::npos || m_catNames[s].find("eletau") != std::string::npos|| m_catNames[s].find("tautau") != std::string::npos ) { 
      if  (goodTaus.size()>0){
	selectionBits[s].SetBitNumber( kTauIsolation );
      }
    }
    
  }

  bool foundMet=false;
 // Cut : select met
  UZH::MissingEt goodMet( &m_missingEt, 0 );
  if (goodMet.et() > m_metCut) {
    // std::cout<<"MET " << goodMet.et() <<std::endl;
    foundMet=true;
  }
  for (unsigned int s=0;s<m_catNames.size();++s) {
    if (m_catNames[s].find("mutau") != std::string::npos || m_catNames[s].find("eletau") != std::string::npos|| m_catNames[s].find("tautau") != std::string::npos) { 
      if (foundMet) selectionBits[s].SetBitNumber( kMet );
    }
  }
  
  

  
  
  N_tight_muons=0;
  N_tight_electrons=0;
  N_loose_muons=0;
  N_loose_electrons=0;

  // if (!foundMet) return;
  // if (!foundTau) return;
  bool foundSecondLepton=false;
  bool foundElectron=false;
  std::vector<UZH::Electron> goodElectrons;
  std::vector<UZH::Electron> goodElectrons_iso;
 
  for ( int i = 0; i <   (m_electron.N)
  	  ; ++i ) {
    UZH::Electron myelectron( &m_electron, i );
    if (myelectron.pt() > m_electronPtCut) {
      if (fabs(myelectron.eta()) < m_electronEtaCut) {
	if ( 
	    myelectron.isVetoElectronWithoutIPandIsolation()==1 
	    
  	    ){
	  if(( fabs(myelectron.eta()) <=  1.479 &&  (myelectron.SemileptonicPFIso()/myelectron.pt()) < 0.175) ||
	     ( fabs(myelectron.eta()) >   1.479  &&  (myelectron.SemileptonicPFIso()/myelectron.pt()) < 0.159 ) ){

	    goodElectrons_iso.push_back(myelectron);
	  }
	}
	 // if(( fabs(myelectron.eta()) <=  1.479 &&  (myelectron.SemileptonicPFIso()/myelectron.pt()) < 0.0893) ||
	//      ( fabs(myelectron.eta()) >   1.479  &&  (myelectron.SemileptonicPFIso()/myelectron.pt()) < 0.121 ) ){
	//     goodElectrons_iso.push_back(myelectron);
	//   }
	// }
	
  	if ( // myelectron.nonTrigMVAID()==1 
	    // myelectron.isTightElectronBoosted()==1
	    //myelectron.isLooseElectronBoosted()==1 
	    myelectron.isVetoElectronWithoutIPandIsolation()==1 
	    // && ((goodFatJets.size() >0 && myelectron.tlv().DeltaR(goodFatJets[0].tlv())>0.8) ||goodFatJets.size() <1)
	    // || myelectron.isMediumElectronBoosted()==1 
  	    ){
  	  //	if (myelectron.isLooseElectron()==1 || myelectron.isMediumElectron()==1   ){
  	  // std::cout<<" myelectron.pt() " << myelectron.pt() <<" myelectron.eta() " << myelectron.eta() <<" myelectron.isLooseElectronBoosted() "<<  myelectron.isLooseElectronBoosted() <<" myelectron.isMediumElectronBoosted()"<<myelectron.isMediumElectronBoosted() <<" iso " << myelectron.SemileptonicPFIso()/myelectron.pt() << std::endl;
  	  // if(( fabs(myelectron.eta()) <=  1.479 &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.0893) ||
  	  //    ( fabs(myelectron.eta()) >   1.479  &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.121 ) ){
	  
  	  // if(( fabs(myelectron.eta()) <=  1.479 &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.0354) ||
  	  //    ( fabs(myelectron.eta()) >   1.479  &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.0646 ) ){

  	  // if(( fabs(myelectron.eta()) <=  1.479 &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.0766) ||
  	  //    ( fabs(myelectron.eta()) >   1.479  &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.0678 ) ){
	 
	  // std::cout << " electron energy before " << myelectron.e() <<std::endl;
	  //double energy = m_CorrectionTool.GetCorrectedEnergy(myelectron.e(), m_isSignal, m_eventInfo.runNumber, (fabs(myelectron.eta()) < 1.4442), 1., myelectron.superCluster_eta(), myelectron.et());
	  // myelectron.e(energy);
	  //std::cout << " electron energy after " << myelectron.e() <<std::endl;
 
	  goodElectrons.push_back(myelectron);
	  
	  foundElectron=true;
	  // }
  	}
      }
    }
  }

 


  
  // for ( int i = 0; i <   (m_electron.N)
  // 	  ; ++i ) {
  //   UZH::Electron myelectron( &m_electron, i );
  //   if (myelectron.pt() > m_electronPtCut) {
  //     if (fabs(myelectron.eta()) < m_electronEtaCut) {
	
  // 	if (myelectron.isLooseElectron()==1   ){
	  
  // 	  goodElectrons.push_back(myelectron);
		 
  // 	  foundElectron=true;
	    
  // 	}
  //     }
  //   }
  // }


  N_tight_electrons=goodElectrons_iso.size();
  N_loose_electrons=goodElectrons.size();

  UZH::Electron goodElectron;
  UZH::Tau goodTau_eletau;
  if (foundElectron ){
      for (unsigned int s=0;s<m_catNames.size();++s) {	
	if (m_catNames[s].find("eletau") != std::string::npos) { 
	  selectionBits[s].SetBitNumber( kLepton );
	}
      }
  }

  bool  useLepPairPt =true; 
  if (!useLepPairPt){
    if (foundElectron && foundTau){ 

      for (unsigned int s=0;s<m_catNames.size();++s) {

	TLorentzVector ele_tlv = goodElectrons[0].tlv();
	TLorentzVector tau_tlv = goodTaus[0].tlv();
	if ( ele_tlv.DeltaR(tau_tlv) >0.05 // && ele_tlv.DeltaR(tau_tlv) < 1.5
	     &&
	     m_catNames[s].find("eletau") != std::string::npos) {
	  selectionBits[s].SetBitNumber( kLeptonDR );
	}
      }
   
    goodElectron = goodElectrons[0];
    goodTau_eletau=goodTaus[0];
    isBoostedTau_eletau= isBoosted_taus[0];
    }
    
  }else{

    float ZPtMin = -99.;
    if (foundElectron && foundTau){ 
      for(unsigned int el=0; el<goodElectrons.size(); el++){
	for(int ta = (int)  goodTaus.size() - 1 ; ta>=0; --ta){

	  TLorentzVector ele_tlv = goodElectrons[el].tlv();
	  TLorentzVector tau_tlv = goodTaus[ta].tlv();
	  if(ele_tlv.DeltaR(tau_tlv)>0.05){  

	    // std::cout <<"Checking ele tau DR "<<  ele_tlv.DeltaR(tau_tlv) <<std::endl;
	    
	    // if( ((goodElectrons[el].tlv()).DeltaR(goodTaus[ta].tlv()) <= 0.3  &&
	    // 	(	( fabs(goodElectrons[el].eta()) <=  1.479 &&  ((goodElectrons[el].SemileptonicPFIso() -  goodTaus[ta].pt())/goodElectrons[el].pt())  < 0.0893 ) ||
	    // 		( fabs(goodElectrons[el].eta()) >   1.479 &&  ((goodElectrons[el].SemileptonicPFIso() -  goodTaus[ta].pt())/goodElectrons[el].pt())   < 0.121 ) ))
	    // 	||
	    // 	((goodElectrons[el].tlv()).DeltaR(goodTaus[ta].tlv())> 0.3  && 
	    // 	 (      ( fabs(goodElectrons[el].eta()) <=  1.479 && goodElectrons[el].SemileptonicPFIso()/goodElectrons[el].pt()  < 0.0893  ) ||
	    // 		( fabs(goodElectrons[el].eta()) >   1.479 && goodElectrons[el].SemileptonicPFIso()/goodElectrons[el].pt()  < 0.121 ) )   ))  {


	    if( ((goodElectrons[el].tlv()).DeltaR(goodTaus[ta].tlv()) <= 0.3  &&
	    	 (	( fabs(goodElectrons[el].eta()) <=  1.479 &&  ((goodElectrons[el].SemileptonicPFIso() -  goodTaus[ta].pt())/goodElectrons[el].pt())  < 0.175 ) ||
	    		( fabs(goodElectrons[el].eta()) >   1.479 &&  ((goodElectrons[el].SemileptonicPFIso() -  goodTaus[ta].pt())/goodElectrons[el].pt())   < 0.159 ) ))
	    	||
	    	((goodElectrons[el].tlv()).DeltaR(goodTaus[ta].tlv())> 0.3  && 
	    	 (      ( fabs(goodElectrons[el].eta()) <=  1.479 && goodElectrons[el].SemileptonicPFIso()/goodElectrons[el].pt()  < 0.175  ) ||
	    		( fabs(goodElectrons[el].eta()) >   1.479 && goodElectrons[el].SemileptonicPFIso()/goodElectrons[el].pt()  < 0.159 ) )   ))  {
	    // if(      ( fabs(goodElectrons[el].eta()) <=  1.479 && goodElectrons[el].SemileptonicCorrPFIso()/goodElectrons[el].pt()  < 0.175  ) ||
	    // 	     ( fabs(goodElectrons[el].eta()) >   1.479 && goodElectrons[el].SemileptonicCorrPFIso()/goodElectrons[el].pt()  < 0.159 ) )  {
	      
	      
	      
	      for (unsigned int s=0;s<m_catNames.size();++s) {	
		if (m_catNames[s].find("eletau") != std::string::npos) { 
		  selectionBits[s].SetBitNumber( kLeptonDR );
		}
	      }
	      if((ele_tlv+tau_tlv).Pt()>ZPtMin){
		goodElectron = goodElectrons[el];
		if( ((goodElectrons[el].tlv()).DeltaR(goodTaus[ta].tlv()) <= 0.3  &&
		     (	( fabs(goodElectrons[el].eta()) <=  1.479 &&  ((goodElectrons[el].SemileptonicPFIso() -  goodTaus[ta].pt())/goodElectrons[el].pt())  < 0.175 ) ||
	    		( fabs(goodElectrons[el].eta()) >   1.479 &&  ((goodElectrons[el].SemileptonicPFIso() -  goodTaus[ta].pt())/goodElectrons[el].pt())   < 0.159 ) ))){
		  goodElectron.SemileptonicPFIso(goodElectrons[el].SemileptonicPFIso() -  goodTaus[ta].pt());
		}
		// if( ((goodElectrons[el].tlv()).DeltaR(goodTaus[ta].tlv()) <= 0.3  &&
		//      (	( fabs(goodElectrons[el].eta()) <=  1.479 &&  ((goodElectrons[el].SemileptonicCorrPFIso())/goodElectrons[el].pt())  < 0.175 ) ||
	    	// 	( fabs(goodElectrons[el].eta()) >   1.479 &&  ((goodElectrons[el].SemileptonicCorrPFIso())/goodElectrons[el].pt())   < 0.159 ) ))){
		//   goodElectron.SemileptonicPFIso(goodElectrons[el].SemileptonicCorrPFIso() );
		// }
	

		goodTau_eletau = goodTaus[ta];
		isBoostedTau_eletau= isBoosted_taus[ta];
		ZPtMin = (ele_tlv+tau_tlv).Pt();
	      } 
	    }
	  }
	}
      }
    }
  }



  bool foundMuon=false;
  std::vector<UZH::Muon> goodMuons;
  std::vector<UZH::Muon> goodMuons_iso;
  
  for ( int i = 0; i <   (m_muon.N)
  	  ; ++i ) {
    UZH::Muon mymuon( &m_muon, i );
    if (mymuon.pt() > m_muonPtCut) {
      if (fabs(mymuon.eta()) < m_muonEtaCut) { 
  	if (mymuon.isLooseMuon()==1
	    // && ((goodFatJets.size() >0 && mymuon.tlv().DeltaR(goodFatJets[0].tlv())>0.8) ||goodFatJets.size() <1)
	    
	    ){
  	  //   std::cout<<" mymuon.SemileptonicPFIso() "<<  mymuon.SemileptonicPFIso() <<std::endl;
  	  // std::cout<<" mymuon.pt() " << mymuon.pt() <<" mymuon.eta() " << mymuon.eta() <<" mymuon.isLooseMuon() "<<  mymuon.isLooseMuon() << " iso "  <<mymuon.SemileptonicPFIso() /mymuon.pt() <<std::endl;
	  


	  // if (mymuon.SemileptonicCorrPFIso() /mymuon.pt() <0.25){
  	    goodMuons.push_back(mymuon);
	    
  	    foundMuon=true;
	    // }
	    
	    if (mymuon.SemileptonicPFIso() /mymuon.pt() <0.25){
	      goodMuons_iso.push_back(mymuon);
	      
	      
	    }


  	}
      }
    }
  }


  // for ( int i = 0; i <   (m_muon.N)
  // 	  ; ++i ) {
  //   UZH::Muon mymuon( &m_muon, i );
  //   if (mymuon.pt() > m_muonPtCut) {
  //     if (fabs(mymuon.eta()) < m_muonEtaCut) { 
  // 	if (mymuon.isLooseMuon()==1){//  mymuon.isLooseMuon()
  // 	  // std::cout<<" mymuon.SemileptonicPFIso() "<<  mymuon.SemileptonicPFIso() <<std::endl;
  // 	  // std::cout<<" mymuon.pt() " << mymuon.pt() <<" mymuon.eta() " << mymuon.eta() <<" mymuon.isLooseMuon() "<<  mymuon.isLooseMuon() <<" mymuon.isTightMuon() "<< mymuon.isTightMuon() << " iso "  <<mymuon.SemileptonicPFIso() /mymuon.pt() <<std::endl;
  // 	  // std::cout <<	  "Muon, pt " <<mymuon.pt() << " eta " <<  mymuon.eta() << " SF " << m_ScaleFactorTool.get_ScaleFactor(mymuon.pt(),fabs(mymuon.eta())) << " Error " << m_ScaleFactorTool.get_ScaleFactorError(mymuon.pt(),fabs(mymuon.eta()))<<std::endl;

  // 	  if (mymuon.SemileptonicPFIso() /mymuon.pt() <0.25){
  // 	    goodMuons.push_back(mymuon);
	    
  // 	    foundMuon=true;
  // 	  }
  // 	}
  //     }
  //   }
  // }

  if (foundMuon ){
    for (unsigned int s=0;s<m_catNames.size();++s) {	
      if (m_catNames[s].find("mutau") != std::string::npos) { 
	selectionBits[s].SetBitNumber( kLepton );
      }
    }
  }
  N_tight_muons=goodMuons_iso.size();
  N_loose_muons=goodMuons.size();

  UZH::Muon goodMuon;
  UZH::Tau goodTau_mutau;

  if (!useLepPairPt){
    
    if (foundMuon && foundTau){ 
      TLorentzVector muon_tlv = goodMuons[0].tlv();
      TLorentzVector tau_tlv = goodTaus[0].tlv();
      goodMuon= goodMuons[0];
      goodTau_mutau=goodTaus[0]; 
      isBoostedTau_mutau= isBoosted_taus[0];
      if( muon_tlv.DeltaR(tau_tlv) >0.05  ){// && muon_tlv.DeltaR(tau_tlv) <1.5
	// std::cout << "Checking DeltaR goodMuons[0].tlv().pt() " << goodMuons[0].tlv().Pt()<< " &&  goodTaus[0].tlv().Pt()" << goodTaus[0].tlv().Pt() <<std::endl;
	std::cout <<"Checking mu tau DR "<<  muon_tlv.DeltaR(tau_tlv) <<std::endl;
	
	if( ((goodMuons[0].tlv()).DeltaR( goodTaus[0].tlv() )<0.4  && ( goodMuons[0].SemileptonicPFIso()-  goodTaus[0].pt() )/ goodMuons[0].pt()  <0.25 )
	    ||((goodMuons[0].tlv()).DeltaR( goodTaus[0].tlv() )> 0.4  && ( goodMuons[0].SemileptonicPFIso() )/goodMuons[0].pt()  <0.25  ))  {
	  
	  for (unsigned int s=0;s<m_catNames.size();++s) {	
	    if (m_catNames[s].find("mutau") != std::string::npos) { 
	      selectionBits[s].SetBitNumber( kLeptonDR );
	    }
	  }
	}
      }
    }
  }else{
    float ZPtMin = -99.;
    if (foundMuon && foundTau){ 
      for(unsigned int mu=0; mu<goodMuons.size(); mu++){
	for(int ta = (int)  goodTaus.size() - 1 ; ta>=0; --ta){

	
	  TLorentzVector muo_tlv = goodMuons[mu].tlv();
	  TLorentzVector tau_tlv = goodTaus[ta].tlv();
	  if(muo_tlv.DeltaR(tau_tlv)>0.05// 1 && muo_tlv.DeltaR(tau_tlv)<0.8
	     ){

	    //std::cout <<"Checking mu tau DR "<<  muo_tlv.DeltaR(tau_tlv) <<std::endl;
	    if( ((goodMuons[mu].tlv()).DeltaR( goodTaus[ta].tlv() )<0.4  && ( goodMuons[mu].SemileptonicPFIso()-  goodTaus[ta].pt() )/ goodMuons[mu].pt()  <0.25 )
	    	||((goodMuons[mu].tlv()).DeltaR( goodTaus[ta].tlv() )> 0.4  && ( goodMuons[mu].SemileptonicPFIso() )/ goodMuons[mu].pt()  <0.25  ))  {

	    // if( ((goodMuons[mu].tlv()).DeltaR( goodTaus[ta].tlv() )<0.4  && ( goodMuons[mu].SemileptonicCorrPFIso() )/ goodMuons[mu].pt()  <0.25 )
	    // 	||((goodMuons[mu].tlv()).DeltaR( goodTaus[ta].tlv() )> 0.4  && ( goodMuons[mu].SemileptonicPFIso() )/ goodMuons[mu].pt()  <0.25  ))  {



	      if((muo_tlv+tau_tlv).Pt()>ZPtMin){ 
	    
		goodMuon = goodMuons[mu];
		if ((goodMuons[mu].tlv()).DeltaR( goodTaus[ta].tlv() )<0.4  && ( goodMuons[mu].SemileptonicPFIso()-  goodTaus[ta].pt() )/ goodMuons[mu].pt()  <0.25 ) {
		  goodMuon.SemileptonicPFIso( goodMuons[mu].SemileptonicPFIso()-  goodTaus[ta].pt());
		  }
		// if ((goodMuons[mu].tlv()).DeltaR( goodTaus[ta].tlv() )<0.4  && ( goodMuons[mu].SemileptonicCorrPFIso() )/ goodMuons[mu].pt()  <0.25 ) {
		//   goodMuon.SemileptonicPFIso( goodMuons[mu].SemileptonicCorrPFIso());
		// }



		goodTau_mutau = goodTaus[ta];
		isBoostedTau_mutau= isBoosted_taus[ta];
		ZPtMin = (muo_tlv+tau_tlv).Pt();
	    
		for (unsigned int s=0;s<m_catNames.size();++s) {	
		  if (m_catNames[s].find("mutau") != std::string::npos) { 
		    selectionBits[s].SetBitNumber( kLeptonDR);
		  }
		}
	      }
	    }
	  }
	}
      }
    }

  }


 




  UZH::Tau goodTau1_tautau;
  UZH::Tau goodTau2_tautau;


  if (m_doTauHad){
    // std::cout<<"trying to look for a pair"<< std::endl;
    float ZPtMin = -99.;
    if (foundTau && goodTaus.size()>1 ){ 
      for (unsigned int s=0;s<m_catNames.size();++s) {
	if ( m_catNames[s].find("tautau") != std::string::npos) {
	  selectionBits[s].SetBitNumber( kLepton   );
	}
      }
      for(unsigned int tau1=0; tau1<goodTaus.size(); tau1++){
	for(unsigned int tau2 =0 ; tau2<goodTaus.size();tau2++){
	  if (tau1==tau2 ) continue;
	
	  TLorentzVector tau1o_tlv = goodTaus[tau1].tlv();
	  TLorentzVector tau_tlv = goodTaus[tau2].tlv();
	  if(tau1o_tlv.DeltaR(tau_tlv)>0.05 //&& tau1o_tlv.DeltaR(tau_tlv)<1.5
	     ){

	    
	    
	    //std::cout <<"Checking mu tau DR :is isolated "<<  muo_tlv.DeltaR(tau_tlv) <<std::endl;
	    if((tau1o_tlv+tau_tlv).Pt()>ZPtMin){ 
	      
	      if (goodTaus[tau1].pt()> goodTaus[tau2].pt() ){
		
		goodTau1_tautau= goodTaus[tau1];
		goodTau2_tautau = goodTaus[tau2];
	      }
	      else { {
		
		  goodTau1_tautau= goodTaus[tau2];
		  goodTau2_tautau = goodTaus[tau1];
		}

	        ZPtMin = (tau1o_tlv+tau_tlv).Pt();
	      
		for (unsigned int s=0;s<m_catNames.size();++s) {	
		  if (m_catNames[s].find("tautau") != std::string::npos) { 
		    selectionBits[s].SetBitNumber( kLeptonDR);
		    // std::cout << " found a  tautau pair" <<std::endl;
		    
		  }
		}
	      }
	  
	    }
	  }
	}
      }
    }
  }
 

  bool foundAK4Jets=false;
  HT=0.;
  int goodAK4Index=0;
  taggedAK4_L=0;
  taggedAK4_M=0;
  taggedAK4_T=0;
  // std::cout << "before ak4 jets: number of jets "<< (m_jetAK4.N)  <<std::endl;

  // UZH::MissingEt goodMet_tlv;
  //goodMet_tlv.et(goodMet.et());
  //goodMet_tlv.phi(goodMet.phi());
  
  TLorentzVector goodMet_tlv;
  goodMet_tlv.SetPtEtaPhiE(goodMet.et(),0, goodMet.phi(),goodMet.et());
  met_phi_ak4=99;
  std::vector<UZH::Jet> goodJetsAK4;
  




 
  if ( m_doShapeSyst){
    // std::cout << "MET Initial " << goodMet.et() <<std::endl;
    goodMet_jesUp.SetPtEtaPhiE(goodMet.et() * goodMet.JetEnUp(),0.,goodMet.phi(),goodMet.et() * goodMet.JetEnUp());
    goodMet_jesDown.SetPtEtaPhiE(goodMet.et() * goodMet.JetEnDown(),0.,goodMet.phi(),goodMet.et() * goodMet.JetEnDown());
    goodMet_jer.SetPtEtaPhiE(goodMet.et(),0.,goodMet.phi(), goodMet.et());
    goodMet_jerUp.SetPtEtaPhiE(goodMet.et() * goodMet.JetResUp(),0.,goodMet.phi(),goodMet.et() * goodMet.JetResUp());
    goodMet_jerDown.SetPtEtaPhiE(goodMet.et() * goodMet.JetResDown(),0.,goodMet.phi(),goodMet.et() * goodMet.JetResDown());
    goodMet_UncEnUp.SetPtEtaPhiE(goodMet.et() * goodMet.UnclusteredEnUp(),0.,goodMet.phi(),goodMet.et() * goodMet.UnclusteredEnUp());
    goodMet_UncEnDown.SetPtEtaPhiE(goodMet.et() *goodMet.UnclusteredEnDown(),0.,goodMet.phi(),goodMet.et() *goodMet.UnclusteredEnDown());

  }

  for ( int i = 0; i < (m_jetAK4.N); ++i ) {
    UZH::Jet myjetak4( &m_jetAK4, i );
    if (fabs(myjetak4.eta()) < m_AK4jetEtaCut) {
      //std::cout << "number of JEts " << i <<std::endl;
      if (myjetak4.pt() > m_AK4jetPtCut) {
	HT = HT + myjetak4.tlv().Et();
	if (m_doShapeSyst && false // && myjetak4.IDTight()
	    ){
	  
	  if(!m_isData){
	    bool isGenJetMatched= false;
	    float dR_genMatch=0.2;
	    float  dPt_genMatch = 3* myjetak4.pt() *myjetak4.jer_sigma_pt();
	    TLorentzVector GenJet_tlv;
	    GenJet_tlv.SetPtEtaPhiE(0,0,0,0);

	    for ( unsigned int j = 0; j < (*m_genjetAK4.genJetAK4_pt).size(); ++j ) {
	      UZH::GenJetak4 mygenjetak4( &m_genjetAK4, j );
	       TLorentzVector genjet;
	       genjet.SetPtEtaPhiE(  mygenjetak4.genJetAK4_pt(),  mygenjetak4.genJetAK4_eta(),  mygenjetak4.genJetAK4_phi(),  mygenjetak4.genJetAK4_e() );
   
	      if (genjet.DeltaR( myjetak4.tlv() ) < dR_genMatch) {
		if (fabs(myjetak4.pt()-genjet.Pt()) < dPt_genMatch) {
		  isGenJetMatched = true;
		  dR_genMatch=genjet.DeltaR( myjetak4.tlv() );
		  GenJet_tlv.SetPtEtaPhiE(genjet.Pt(),genjet.Eta(),genjet.Phi(),genjet.E() );
		}      
	      }
	      
	    }
	  

	  //std::cout <<" i " << i <<  " jet pt "<< myjetak4.tlv().Pt() << std::endl;
	  TLorentzVector jet_tlv=myjetak4.tlv();
	  goodJetsAK4_jesUp = m_CorrectionTool.GetCorrectedJet(jet_tlv, float(myjetak4.jec()), float(myjetak4.jecUp()), float(myjetak4.jecDown()), !m_isData, +1, 0,false);
	  // std::cout << "-MET "<< goodMet_jesUp.Et()<< std::endl;
	  goodMet_jesUp += myjetak4.tlv() ;
	  goodMet_jesUp  -= goodJetsAK4_jesUp ;
	  //std::cout << "jet pt "<< myjetak4.tlv().Pt() << " with jecup "<< goodJetsAK4_jesUp.Pt()  << std::endl;
	  // std::cout << "MET "<< goodMet_jesUp.Et()<< " eta "<<  goodMet_jesUp.Eta()<<  std::endl;
	 
	  goodJetsAK4_jesDown = m_CorrectionTool.GetCorrectedJet(jet_tlv, float(myjetak4.jec()), float(myjetak4.jecUp()), float(myjetak4.jecDown()), !m_isData, -1, 0,false);
	  goodMet_jesDown += myjetak4.tlv() - goodJetsAK4_jesDown ;
	  trandom = new TRandom3();
	  float random=trandom->Gaus(0., float(myjetak4.jer_sigma_pt()));
	  //goodJetsAK4_jerUp = m_CorrectionTool.GetCorrectedJetJER(jet_tlv, float(myjetak4.jec()), float(myjetak4.jecUp()), float(myjetak4.jecDown()), !m_isData, 0, +1,false);
	  goodJetsAK4_jerUp = m_CorrectionTool.GetCorrectedJetJER_v2(myjetak4.tlv(), GenJet_tlv, float(myjetak4.jer_sigma_pt()), float(myjetak4.jer_sf()), float(myjetak4.jer_sf_up()),float(myjetak4.jer_sf_down()), !m_isData,  +1,true, isGenJetMatched,random);

	  goodMet_jerUp += myjetak4.tlv() - goodJetsAK4_jerUp ;
	
	  
	  goodJetsAK4_jerDown = m_CorrectionTool.GetCorrectedJetJER_v2(myjetak4.tlv(),GenJet_tlv, float(myjetak4.jer_sigma_pt()),float(myjetak4.jer_sf()), float(myjetak4.jer_sf_up()),float(myjetak4.jer_sf_down()), !m_isData, -1,true, isGenJetMatched,random);
	  goodMet_jerDown += myjetak4.tlv() - goodJetsAK4_jerDown ;
	
	  goodJetsAK4_jer = m_CorrectionTool.GetCorrectedJetJER_v2(myjetak4.tlv(), GenJet_tlv, float(myjetak4.jer_sigma_pt()), float(myjetak4.jer_sf()), float(myjetak4.jer_sf_up()),float(myjetak4.jer_sf_down()), !m_isData,  0,true, isGenJetMatched,random);
	
	 
	  goodMet_jer += myjetak4.tlv() - goodJetsAK4_jer ;
	  }
	  
	 
	 
 	}

     
	// m_logger << INFO << myjetak4.pt() << SLogger::endmsg;

	// Book( TH1F( "Jet_pt", "Jet p_{T} [GeV]", 100, 0.0, 1000 ), "AK4_low" )->Fill( myjetak4.pt() );
	// Book( TH1F( "Jet_eta", "Jet #eta", 60, -3,3 ), "AK4_low" )->Fill( myjetak4.eta() );
	// Book( TH1F( "Jet_m", "Jet m [GeV]", 50, 0, 250 ), "AK4_low" )->Fill( myjetak4.m() );
	// Book( TH1F( "Jet_phi", "Jet #phi", 50, -TMath::Pi(), TMath::Pi() ), "AK4_low" )->Fill( myjetak4.phi() );
	// Book( TH1F( "Jet_e", "Jet e [GeV]", 100, 0.0, 1000 ), "AK4_low" )->Fill( myjetak4.e() );
    
	// Book( TH1F( "Jet_muf", "Jet muf", 50, 0.0, 1.0 ), "AK4_low" )->Fill( myjetak4.muf() );
	// Book( TH1F( "Jet_phf", "Jet phf", 50, 0.0, 1.0 ), "AK4_low" )->Fill( myjetak4.phf() );
	// Book( TH1F( "Jet_emf", "Jet emf", 50, 0.0, 1.0 ), "AK4_low" )->Fill( myjetak4.emf() );
	// Book( TH1F( "Jet_nhf", "Jet nhf", 50, 0.0, 1.0 ), "AK4_low" )->Fill( myjetak4.nhf() );
	// Book( TH1F( "Jet_chf", "Jet chf", 50, 0.0, 1.0 ), "AK4_low" )->Fill( myjetak4.chf() );
	// Book( TH1F( "Jet_area", "Jet area", 40, 0.0, 4.0 ), "AK4_low" )->Fill( myjetak4.area() );
	// Book( TH1F( "Jet_cm", "Jet cm", 50, 0.0, 50.0 ), "AK4_low" )->Fill( myjetak4.cm() );
	// Book( TH1F( "Jet_nm", "Jet nm", 50, 0.0, 50.0 ), "AK4_low" )->Fill( myjetak4.nm() );
      }
      if (fabs(myjetak4.eta()) <2.4 &&fabs(myjetak4.pt()>20) ){// priam era 40 e eta<3
	if (myjetak4.IDTight()){
	  bool notmatchedwithoters= true;
	  if( goodFatJets.size()>0 && myjetak4.tlv().DeltaR( Jet.tlv())  < 1.2 ) notmatchedwithoters=false;
	 
	  if ( selectionBits[1].TestBitNumber( kLepton )  && selectionBits[1].TestBitNumber( kLeptonDR ) 
	       && ( myjetak4.tlv().DeltaR(goodElectron.tlv() )<0.4 || myjetak4.tlv().DeltaR(goodTau_eletau.tlv())  <0.4))notmatchedwithoters=false;
	  if ( selectionBits[0].TestBitNumber( kLepton )  && selectionBits[0].TestBitNumber( kLeptonDR ) 
	       && ( myjetak4.tlv().DeltaR(goodMuon.tlv() )  <0.4 || myjetak4.tlv().DeltaR(goodTau_mutau.tlv())  <0.4))notmatchedwithoters=false;
	  if ( selectionBits[2].TestBitNumber( kLepton )  && selectionBits[2].TestBitNumber( kLeptonDR ) 
	       && ( myjetak4.tlv().DeltaR(goodTau2_tautau.tlv() )<0.4 || myjetak4.tlv().DeltaR(goodTau1_tautau.tlv())  <0.4))notmatchedwithoters=false;
	
	
	  if (notmatchedwithoters ){
	    goodJetsAK4.push_back(myjetak4);



	    
	    // if ( myjetak4.csv() >  0.60 ) taggedAK4_L++;
	    // if ( myjetak4.csv() >  0.89 ) taggedAK4_M++;
	    // if ( myjetak4.csv() >  0.97 ) taggedAK4_T++;
	    if ( myjetak4.csv() >  0.5426  ) taggedAK4_L++;
	    if ( myjetak4.csv() >  0.8484 ) taggedAK4_M++;
	    if ( myjetak4.csv() >  0.9535 ) taggedAK4_T++;

	    
	    if( met_phi_ak4> myjetak4.tlv().DeltaPhi(goodMet_tlv)){
	      met_phi_ak4=  myjetak4.tlv().DeltaPhi(goodMet_tlv);
	    }
	   
	  }

	}
      }
  
    }
  }
  N_AK4=goodJetsAK4.size();
  if (goodJetsAK4.size()>0){

     if (m_isSignal) {
         m_bTaggingScaleTool.fillEfficiencies_veto(goodJetsAK4);
     }    
     
    if (!m_isData) {
      b_weightBtag_veto = m_bTaggingScaleTool.getScaleFactor_veto(goodJetsAK4); // event b-tag SF weight
      if (m_doNormSyst ){
	int sigma_bc = 1 , sigma_udsg = 1;
	b_weightBtag_veto_bc_up = m_bTaggingScaleTool.getScaleFactor_veto(goodJetsAK4,  sigma_bc, 0);
	b_weightBtag_veto_l_up = m_bTaggingScaleTool.getScaleFactor_veto(goodJetsAK4,  0 ,sigma_udsg);
	sigma_bc = -1 ; sigma_udsg = -1;
	b_weightBtag_veto_bc_down = m_bTaggingScaleTool.getScaleFactor_veto(goodJetsAK4,  sigma_bc, 0);
	b_weightBtag_veto_l_down = m_bTaggingScaleTool.getScaleFactor_veto(goodJetsAK4,  0 ,sigma_udsg);
      }
      b_weight *= b_weightBtag_veto;
    }
  }

  // std::cout << "after ak4 jets, HT " <<HT<< std::endl;
  
  //Jet for trigger turn on on the first cat, anyways just checking triggers and met filters doesn't change
  if   (selectionBits[0].TestBitNumber(  kTrigger )  && selectionBits[0].TestBitNumber( kMetFilters )) {
  
    //if (!foundJet) return; it already returns before if doesn't find a jet AK8
    if(Jet.pt()<20 ) std::cout << "Jet pt " <<Jet.pt()<< " ;Jet.pruned_massCorr() " << Jet.pruned_massCorr() << " ;Jet eta " <<Jet.eta() <<" ;Jet phi "<< Jet.phi() << std::endl;
    Hist( "JetPt" )->Fill(Jet.pt());
    Hist( "Jet_pruned_massCorr")->Fill(Jet.pruned_massCorr());
    Hist( "SumEt")->Fill(goodMet.sumEt());
    Hist( "MET")->Fill(goodMet.et());
    Hist( "HT")->Fill(HT);
    Hist( "N_ak4")->Fill(goodJetsAK4.size());
    Hist( "N_ak8")->Fill(goodFatJets.size());

    if (!m_isData) {
      for( unsigned int v = 0; v < (m_eventInfo.actualIntPerXing)->size(); ++v ){
      
	if ( (*m_eventInfo.bunchCrossing)[v] == 0 ) {
	Hist( "N_vertex")->Fill((*m_eventInfo.actualIntPerXing)[v] );
	}
    
      }
    }
  }
  // std::cout << "after hists for trigger efficiency "<< std::endl;


  
  for (unsigned int s=0;s<m_catNames.size();++s) {
    // m_logger << INFO << selectionBits[s].TestBitNumber( kTrigger ) << SLogger::endmsg;
    if (m_catNames[s].find("NoWindow") != std::string::npos) {  
      
      fillCutflow("cutflow", m_catNames[s], selectionBits[s], 1  );
    }
  }
  
  bool doHistograms = false;
  // need vectorJet and higgsJet to be defined
  for (unsigned int s=0;s<m_catNames.size();++s) {
    if (selectionBits[s].TestBitNumber( kMassWindow )) {
      doHistograms = true;
    }
  }
  
  m_logger << VERBOSE << "before doHistograms" << SLogger::endmsg;
  
  if (doHistograms) {
    // calculate a few variables before filling histograms
    double vJet_tau21 = -1;
    double vJet_tau31 = -1;
    double vJet_tau32 = -1;
    if (Jet.tau1() != 0) {
      vJet_tau21 = Jet.tau2()/Jet.tau1();
      vJet_tau31 = Jet.tau3()/Jet.tau1();
    }
    if (Jet.tau2() != 0) {
      vJet_tau32 = Jet.tau3()/Jet.tau2();
    }
    int vJet_nTaggedSubjets = 0;
    double vJet_subjet0_csv = -99;
    double vJet_subjet1_csv = -99;
    double ak8jet0_subjet01_deta=-99;
    double ak8jet0_subjet01_dphi=-99;
    double ak8jet0_subjet01_dr = -99;
    
    if (Jet.subjet_softdrop_N() >= 2) {
      ak8jet0_subjet01_deta = fabs(Jet.subjet_softdrop_eta()[0] - Jet.subjet_softdrop_eta()[1]);
      ak8jet0_subjet01_dphi = fabs(Jet.subjet_softdrop_phi()[0] - Jet.subjet_softdrop_phi()[1]);
      ak8jet0_subjet01_dr = sqrt(ak8jet0_subjet01_deta*ak8jet0_subjet01_deta + ak8jet0_subjet01_dphi*ak8jet0_subjet01_dphi);
    }
    for (int i = 0; i < Jet.subjet_softdrop_N(); ++i) {
      switch(i) {
      case 0:
	vJet_subjet0_csv = Jet.subjet_softdrop_csv()[i];
	break;
      case 1:
	vJet_subjet1_csv = Jet.subjet_softdrop_csv()[i];
	break;
      }
      
      if ((m_bTaggingScaleTool.isTagged(Jet.subjet_softdrop_csv()[i]))) {
        ++vJet_nTaggedSubjets;
      }
    }
  
    // fill cut info for ntuple
    std::vector<bool> passed_all(m_catNames.size(), true);
    for (unsigned int s=0;s<m_catNames.size();++s) {
       // if (m_catNames[s].find("NoWindow") != std::string::npos){ In noWindow the tau21, mass b-tagg are set to true 
       // 	std:: cout <<m_catNames[s]<< std::endl;
	for( UInt_t icut = 0; icut <    kNumCuts; ++icut ) {
	  //std:: cout << kCutName[ icut]  << " " << selectionBits[s].TestBitNumber( icut ) << std::endl;
	  if( selectionBits[s].TestBitNumber( icut ) != kTRUE ){
	    passed_all[s] = false;
	  }
	  else{


	    // if (icut) b_selection_bits[s]|=1<<icut; 
	    // if (icut-1==(unsigned)b_selection_lastcut[s])
	    //   b_selection_lastcut[s]++;
	  }
	}//cut loop
       // }else passed_all[s] = false;
    }//category loop
    

    
    double deta = Jet.eta();
    double dphi = Jet.phi();
    double dr = sqrt(deta*deta + dphi*dphi);
    TLorentzVector diJet = Jet.tlv()   ;
    
    m_logger << VERBOSE << "category loopfillHistograms" << SLogger::endmsg;
    
    for (unsigned int s=0;s<m_catNames.size();++s) {
      if (passed_all[s]) {
        m_logger << VERBOSE << m_catNames[s] << SLogger::endmsg;
	if (m_catNames[s].find("NoWindow") != std::string::npos) {

	  bool triggerStudy_=false;
	  if (triggerStudy_){
	    bool channel_alreadyassigned=false;

	    if (m_catNames[s].find("mutau")  != std::string::npos && selectionBits[s].TestBitNumber(kTau )&& selectionBits[s].TestBitNumber(kLepton ) && !channel_alreadyassigned ) 	{
	      // std::cout<<" checking out offline selection for mutau; mu iso = "<< goodMuons[0].SemileptonicPFIso() << std::endl;
	      checkTrigger(m_catNames[s],"mutau");
	    
	      checkTriggerOfflineSelection(m_catNames[s], Jet,goodTaus[0], goodMuons[0].tlv() , goodMuons[0].isTightMuon(), goodMuons[0].SemileptonicPFIso(),
					   goodMet, "mutau", HT);
	      channel_alreadyassigned=true;
	    }
	    if (m_catNames[s].find("eletau")  != std::string::npos && selectionBits[s].TestBitNumber(kTau )&& selectionBits[s].TestBitNumber(kLepton )  && !channel_alreadyassigned ) {
	      // std::cout<<" checking out offline selection for eletau; ele iso = " << goodElectrons[0].SemileptonicPFIso() <<std::endl;	
	      checkTrigger(m_catNames[s],"eletau");  
	      checkTriggerOfflineSelection(m_catNames[s], Jet,goodTaus[0], goodElectrons[0].tlv() , goodElectrons[0].isTightElectron(), goodElectrons[0].SemileptonicPFIso(),
					   goodMet, "eletau", HT);
	      channel_alreadyassigned=true; 
	    }
	    if (m_catNames[s].find("tautau")  != std::string::npos && selectionBits[s].TestBitNumber(kTau )&& selectionBits[s].TestBitNumber(kLepton ) && !channel_alreadyassigned ) { 
	      //std::cout<<" checking out offline selection for tautau"<< std::endl;
	      checkTrigger(m_catNames[s],"tautau");
	      checkTriggerOfflineSelection(m_catNames[s], Jet,goodTaus[0], goodTaus[1].tlv() , 0,0, goodMet, "tautau", HT);
	      channel_alreadyassigned=true;
	    }
	  }
	}
	if (m_FillHist) {
	  fillHistograms(m_catNames[s], Jet, Jet, diJet, vJet_tau21, vJet_tau31, vJet_tau32, vJet_nTaggedSubjets, vJet_subjet0_csv, vJet_subjet1_csv, vJet_tau21, vJet_tau31, vJet_tau32, vJet_nTaggedSubjets, vJet_subjet0_csv, vJet_subjet1_csv, deta, dphi, dr );
	}
      }
    }
    
    m_logger << VERBOSE << "after hist" << SLogger::endmsg;
    
  
    bool channel_alreadyassigned=false; 
    bool event_written=false;
  

    bool priority_tomutau=false;
    // std:: cout << "writing event" << std::endl;
    // if( m_catNames[1].find("eletau_") != std::string::npos &&  m_catNames[0].find("mutau_") !=std::string::npos ) std:: cout << "cutname ok "<<std::endl;
    //std:: cout << "  passed_all[0]  " <<  passed_all[0] <<" passed_all[1] " << passed_all[1]<<" passed_all[2] " << passed_all[2]<<std::endl;

    
    if (  m_catNames[0].find("mutau_")  != std::string::npos &&  passed_all[0] && m_catNames[1].find("eletau_")  != std::string::npos && passed_all[1]) {
      if( ( goodTau_mutau.tlv()+ goodMuon.tlv()).Pt() >= ( goodTau_eletau.tlv()+ goodElectron.tlv()).Pt()) priority_tomutau=true;

    }
  
    if ( ( passed_all[0]  && ! passed_all[1] )

	  ||  
	 (  passed_all[0]  && passed_all[1]  && priority_tomutau )

	 ) {
      
      FillBranches( "mutau", Jet, vJet_tau21, vJet_tau31, vJet_tau32, vJet_nTaggedSubjets, vJet_subjet0_csv,vJet_subjet1_csv,  ak8jet0_subjet01_dr,   ak8jet0_subjet01_deta, ak8jet0_subjet01_dphi,  b_weight,  b_weightGen,  b_weightPU,  b_weightBtag, b_weightBtag_veto,   b_runNumber, b_eventNumber, b_lumiBlock,  goodTau_mutau, goodMuon.tlv() ,goodMuon.SemileptonicPFIso(), goodMet, 0,0,0);
      channel_alreadyassigned=true; 
      mu_tau++;
      event_written=true;
      //return;
    }
    else if ( 
	     
	     (!passed_all[0]  && passed_all[1])
	     
	     ||  
	     (  passed_all[0]  &&  passed_all[1]  && !priority_tomutau )
	     

	     
	     
	      ) {
      FillBranches( "eletau", Jet, vJet_tau21, vJet_tau31, vJet_tau32, vJet_nTaggedSubjets, vJet_subjet0_csv, vJet_subjet1_csv,  ak8jet0_subjet01_dr,   ak8jet0_subjet01_deta, ak8jet0_subjet01_dphi,  b_weight,  b_weightGen,  b_weightPU,  b_weightBtag, b_weightBtag_veto,    b_runNumber, b_eventNumber, b_lumiBlock,  goodTau_eletau, goodElectron.tlv(),goodElectron.SemileptonicPFIso(), goodMet, 0, goodElectron.superCluster_eta() ,0);
      channel_alreadyassigned=true; 
      event_written=true;
      ele_tau++;
    }
    else if( m_doTauHad &&  passed_all [2]) {
      // std::cout << " crashing while wirting" <<" tau iso and pt"<< goodTau2_tautau.byIsolationMVArun2v1PWnewDMwLTraw() << " "<< goodTau2_tautau.tlv().Pt() <<std::endl;
      FillBranches( "tautau", Jet, vJet_tau21, vJet_tau31, vJet_tau32, vJet_nTaggedSubjets, vJet_subjet0_csv, vJet_subjet1_csv,  ak8jet0_subjet01_dr,   ak8jet0_subjet01_deta, ak8jet0_subjet01_dphi,  b_weight,  b_weightGen,  b_weightPU,  b_weightBtag, b_weightBtag_veto,    b_runNumber, b_eventNumber, b_lumiBlock,  goodTau1_tautau, goodTau2_tautau.tlv(),  goodTau2_tautau.byIsolationMVArun2v1PWnewDMwLTraw(),goodMet,  goodTau2_tautau.decayMode(),0, goodTau2_tautau.TauType() );
	channel_alreadyassigned=true; 
	event_written=true;
	tau_tau++;
    }
    else {
      throw SError( SError::SkipEvent );  
    }
    m_logger << VERBOSE << "after ntuple" << SLogger::endmsg;
  }    
  return;
}

  bool VHTausAnalysis::isGoodEvent(int runNumber, int lumiSection) {
  
  bool isGood = true;
  if (m_isData) {
    isGood = m_grl.HasRunLumiBlock( runNumber, lumiSection );
    if( !isGood ) {
      m_logger << WARNING << "Bad event! Run: " << runNumber <<  " - Lumi Section: " << lumiSection << SLogger::endmsg;
      // throw SError( SError::SkipEvent );
    }
    else m_logger << VERBOSE << "Good event! Run: " << runNumber <<  " - Lumi Section: " << lumiSection << SLogger::endmsg;
  }

  return isGood;
  
  }


bool VHTausAnalysis::passTrigger() {
  
  bool passTrigger = false;
  
  for (std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
    for (unsigned int t = 0; t < m_triggerNames.size(); ++t ){
      if ((it->first).find(m_triggerNames[t]) != std::string::npos) {
        if (it->second == true) {
          m_logger << VERBOSE << "Trigger pass: " << (it->first) << SLogger::endmsg;
          passTrigger = true;
          return passTrigger;
        }
      }
    }
  }
  
  return passTrigger;
  
}


bool VHTausAnalysis::passMETFilters() {
  
  bool passMetFilters = true;
  
  // using only what's recommended in https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2
    
  if( !(m_eventInfo.PV_filter) ) {
    passMetFilters = false;
    m_logger << VERBOSE << "PV_filter" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(1);
  }
  if( !(m_eventInfo.passFilter_CSCHalo) ) {
    passMetFilters = false;
    m_logger << VERBOSE << "passFilter_CSCHalo" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(2);
  }
  if( !(m_eventInfo.passFilter_HBHELoose) ) {
    passMetFilters = false;
    m_logger << VERBOSE << "passFilter_HBHELoose" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(3);
  }
  if( !(m_eventInfo.passFilter_HBHEIso) ) {
    passMetFilters = false;
    m_logger << VERBOSE << "passFilter_HBHEIso" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(4);
  }
  if( !(m_eventInfo.passFilter_EEBadSc) ) {
    passMetFilters = false;
    m_logger << VERBOSE << "passFilter_EEBadSc" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(5);
  }
 
  if( !(m_eventInfo.passFilter_globalTightHalo2016) ){
    passMetFilters = false;
    m_logger << VERBOSE << "passFilter_EEBadSc" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(6);
  }

  if( !(m_eventInfo.passFilter_chargedHadronTrackResolution) ){
    passMetFilters = false;
    m_logger << VERBOSE << "passFilter_EEBadSc" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(7);
  }

  if( !(m_eventInfo.passFilter_muonBadTrack) ) {        
    passMetFilters = false;
    m_logger << VERBOSE << "passFilter_EEBadSc" << SLogger::endmsg;
    Hist( "METFilters" )->Fill(8);
  } 

 if (m_isData)  {
  
    if( !(!m_eventInfo.flag_badMuons) ) {        
     passMetFilters = false;
     m_logger << VERBOSE << "flag_badMuons" << SLogger::endmsg;
     //Hist( "METFilters" )->Fill(9);
    }
    if( !(!m_eventInfo.flag_duplicateMuons) ) {        
     passMetFilters = false;
     m_logger << VERBOSE << "flag_duplicateMuons" << SLogger::endmsg;
     //Hist( "METFilters" )->Fill(9);
    }
    if( !(!m_eventInfo.flag_nobadMuons) ) {        
     passMetFilters = false;
     m_logger << VERBOSE << "flag_nobadMuons" << SLogger::endmsg;
     //Hist( "METFilters" )->Fill(9);
    }
  
  }


  return passMetFilters;
  
}

void VHTausAnalysis::initPuppiSDWeight(void  ){

 
  
  auto fileSD = TFile::Open( m_PUPPIJEC_.c_str(),"READ");
  puppisd_corrGEN      = (TF1*)fileSD->Get("puppiJECcorr_gen");
  puppisd_corrRECO_cen = (TF1*)fileSD->Get("puppiJECcorr_reco_0eta1v3");
  puppisd_corrRECO_for = (TF1*)fileSD->Get("puppiJECcorr_reco_1v3eta2v5");
  // puppisd_corrRECO_for->SetDirectory(0);
  //puppisd_corrRECO_cen->SetDirectory(0);
 
  fileSD->Close();
   
}
float  VHTausAnalysis::getPUPPIweight(float puppipt, float puppieta ){
  
  float genCorr  = 1.;
  float recoCorr = 1.;
  float totalWeight = 1.;
        
  genCorr =  puppisd_corrGEN->Eval( puppipt );
  
  // std::cout<< "for PUPPI jet pT = 300 GeV : 1./genCorr = " << 1./(puppisd_corrGEN->Eval( 300. )) << " recoCorr = " <<puppisd_corrRECO_cen->Eval( 300 ) << " Total corr = " << (1./(puppisd_corrGEN->Eval( 300. )))*puppisd_corrRECO_cen->Eval( 300 )<<  std::endl;

  if( fabs(puppieta)  <= 1.3 ) recoCorr = puppisd_corrRECO_cen->Eval( puppipt );
  else
    if( fabs(puppieta) > 1.3 ) recoCorr = puppisd_corrRECO_for->Eval( puppipt );
  
  totalWeight = genCorr * recoCorr;
  // std::cout<< "for PUPPI jet pT = " << puppipt << " and eta = " << puppieta << " my weight is: " << std::endl;
  // std::cout<< "Gen corr = " << (1./genCorr) << " recoCorr = " << recoCorr << " Total = " << totalWeight << std::endl;
  return totalWeight;
}



double VHTausAnalysis::getEventWeight() {
  
  double weight = 1.;
  for( unsigned int v = 0; v < (m_eventInfo.actualIntPerXing)->size(); ++v ){
    
    if ( (*m_eventInfo.bunchCrossing)[v] == 0 ) {
      b_weightPU = m_pileupReweightingTool.getPileUpweight( (*m_eventInfo.actualIntPerXing)[v] , 0);

      if (m_doNormSyst){
	b_weightPU_up = m_pileupReweightingTool.getPileUpweight( (*m_eventInfo.actualIntPerXing)[v] ,1);
	b_weightPU_down = m_pileupReweightingTool.getPileUpweight( (*m_eventInfo.actualIntPerXing)[v],-1 );

      }
      m_logger << VERBOSE << "Weight: " << b_weightPU << " for true: " << (*m_eventInfo.actualIntPerXing)[v] << SLogger::endmsg;
      //std::cout << "Weight: " << b_weightPU << " for true: " << (*m_eventInfo.actualIntPerXing)[v] <<std::endl;
      break;
    }
  }

  TString File_at_the_moment_evweight = TString(this->GetHistInputFile()->GetName());	       
  if ( File_at_the_moment_evweight.Contains( "TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8") && m_eventInfo.lheHT>600)b_weightGen = 0;
  else   b_weightGen = (m_eventInfo.genEventWeight < 0) ? -1 : 1; 
  weight *= b_weightPU*b_weightGen;
  // std::cout << "Weight Gen : " << b_weightGen <<std::endl;
  return weight;
  
}

void VHTausAnalysis::clearBranches() {

  b_weight = 1.;
  b_weightGen = 1.;
  b_weightGen_corr = 1.;

  if ( m_doNormSyst){
    b_weightPU_up = 1.;
    b_weightPU_down = 1.;
  }
  b_weightBtag = 1.;
  b_weightBtag_veto = 1.;
  if ( m_doNormSyst){
    b_weightBtag_bc_up = 1.;
    b_weightBtag_veto_bc_up = 1.;
    b_weightBtag_l_up = 1.;
    b_weightBtag_veto_l_up = 1.;
    b_weightBtag_bc_down = 1.;
    b_weightBtag_veto_bc_down = 1.;
    b_weightBtag_l_down = 1.;
    b_weightBtag_veto_l_down = 1.;
  }
  b_weightLumi = 1.;
  
  b_runNumber = -99;;
  b_eventNumber = -99;
  b_lumiBlock = -99;
  
  b_ak8jet0_pt = -99;
  b_ak8jet0_phi = -99;
  b_ak8jet0_eta = -99;
  b_ak8jet0_e = -99;
  b_ak8jet0_tau21 = -99;
  b_ak8jet0_m = -99;
  b_ak8jet0_mpruned = -99;
  b_ak8jet0_csv = -99;
  b_ak8jet1_pt = -99;
  b_ak8jet1_phi = -99;
  b_ak8jet1_eta = -99;
  b_ak8jet1_e = -99;
  b_ak8jet1_tau21 = -99;
  b_ak8jet1_m = -99;
  b_ak8jet1_mpruned = -99;
  b_ak8jet1_csv = -99;
  b_ak8jet0_subjet01_dr = -99;
  b_ak8jet0_subjet01_deta = -99;
  b_ak8jet0_subjet01_dphi = -99;
  b_ak8jet0_subjet0_pt = -99;
  b_ak8jet0_subjet1_pt = -99;
  b_ak8jet0_subjet0_csv = -99;
  b_ak8jet0_subjet1_csv = -99;
  b_ak8jet1_subjet01_dr = -99;
  b_ak8jet1_subjet01_deta = -99;
  b_ak8jet1_subjet01_dphi = -99;
  b_ak8jet1_subjet0_pt = -99;
  b_ak8jet1_subjet1_pt = -99;
  b_ak8jet1_subjet0_csv = -99;
  b_ak8jet1_subjet1_csv = -99;
  
  b_selection_bits.clear();
  b_selection_lastcut.clear();



  GenEvent_Htata_filter=false;
  GenEvent_Ztata_filter=false;
  GenEvent_Gammatata_filter=false;
  isBoostedTau=false;
  isBoostedTau_eletau=false;
  isBoostedTau_mutau=false;
  
  N_AK4=-99;
  taggedAK4_L=-99;
  taggedAK4_M=-99;
  taggedAK4_T=-99;

  N_tight_muons=-99;
  N_tight_electrons=-99;
  met_phi_ak4=99;
  for(unsigned int ch=0;ch< channels_.size();ch++){

  

    b_weight_[channels_[ch]] = 1.;
    b_weightGen_[channels_[ch]] = 1.;
    b_weightGen_corr_[channels_[ch]] = 1.;
    b_weightMET_[channels_[ch]] = 1.;
    b_weightPU_[channels_[ch]] = 1.;
    if (m_doNormSyst){
      b_weightPU_up_[channels_[ch]] = 1.;
      b_weightPU_down_[channels_[ch]] = 1.;
    }
    b_weightBtag_[channels_[ch]] = 1.;
    b_weightBtag_veto_[channels_[ch]] = 1.;
    if (m_doNormSyst){
      b_weightBtag_bc_up_[channels_[ch]] = 1.;
      b_weightBtag_veto_bc_up_[channels_[ch]] = 1.;
      b_weightBtag_l_up_[channels_[ch]] = 1.;
      b_weightBtag_veto_l_up_[channels_[ch]] = 1.;
      b_weightBtag_bc_down_[channels_[ch]] = 1.;
      b_weightBtag_veto_bc_down_[channels_[ch]] = 1.;
      b_weightBtag_l_down_[channels_[ch]] = 1.;
      b_weightBtag_veto_l_down_[channels_[ch]] = 1.;
    }
    b_weightTrigSemi_[channels_[ch]] = 1.;
    b_weightTauID_[channels_[ch]] = 1.;
    b_weightLepID_[channels_[ch]] = 1.;
    b_weightLepIso_[channels_[ch]] = 1.;

    if (m_doNormSyst){
      b_weightTrigSemi_up_[channels_[ch]] = 1.;
      b_weightLepID_up_[channels_[ch]] = 1.;
      b_weightLepIso_up_[channels_[ch]] = 1.;
      b_weightTrigSemi_down_[channels_[ch]] = 1.;
      b_weightLepID_down_[channels_[ch]] = 1.;
      b_weightLepIso_down_[channels_[ch]] = 1.;
    }
    b_weightLumi_[channels_[ch]] = 1.;
    b_weightTot_[channels_[ch]] = 1.;
    // b_weightTot_woBtags_[channels_[ch]] = 1.;
    // b_weightTot_woBtagVeto_[channels_[ch]] = 1.;
    // b_weightTot_woBtagSJ_[channels_[ch]] = 1.;

    if (m_doPDFSyst){
      b_weightFacUp_[channels_[ch]] = 1.;
      b_weightFacDown_[channels_[ch]] = 1.;
      b_weightRenUp_[channels_[ch]] = 1.;
      b_weightRenDown_[channels_[ch]] = 1.;
      b_weightFacRenUp_[channels_[ch]] = 1.;
      b_weightFacRenDown_[channels_[ch]] = 1.;
      b_weightPDF_rms_[channels_[ch]] = 1.;
   
    }


    b_runNumber_[channels_[ch]] = -99;
    b_eventNumber_[channels_[ch]] = -99;
    b_lumiBlock_[channels_[ch]] = -99;
  
    b_ak8jet0_pt_[channels_[ch]] = -99;
    b_ak8jet0_phi_[channels_[ch]] = -99;
    b_ak8jet0_eta_[channels_[ch]] = -99;
    b_ak8jet0_e_[channels_[ch]] = -99;
    b_ak8jet0_tau21_[channels_[ch]] = -99;
    b_ak8jet0_sd_tau21_[channels_[ch]] = -99;
    b_ak8jet0_m_[channels_[ch]] = -99;
    b_ak8jet0_mpruned_[channels_[ch]] = -99;
    b_ak8jet0_msoftdrop_[channels_[ch]] = -99;
    b_ak8jet0_mpruned_corr_[channels_[ch]] = -99;
    b_ak8jet0_msoftdrop_corr_[channels_[ch]] = -99;
    b_ak8jet0_msoftdrop_corr_smear_[channels_[ch]] = -99;
    if( m_doShapeSyst){
      b_ak8jet0_msoftdrop_corr_smear_jmsUp_[channels_[ch]] = -99;
      b_ak8jet0_msoftdrop_corr_smear_jmsDown_[channels_[ch]] = -99;
      b_ak8jet0_msoftdrop_corr_smear_jmrUp_[channels_[ch]] = -99;
      b_ak8jet0_msoftdrop_corr_smear_jmrDown_[channels_[ch]] = -99;
    }

    b_ak8jet0_csv_[channels_[ch]] = -99;
    b_ak8jet0_Hbbtag_[channels_[ch]] = -99;
    b_ak8jet0_subjet01_dr_[channels_[ch]] = -99;
    b_ak8jet0_subjet01_deta_[channels_[ch]] = -99;
    b_ak8jet0_subjet01_dphi_[channels_[ch]] = -99;
    b_ak8jet0_subjet0_pt_[channels_[ch]] = -99;
    b_ak8jet0_subjet1_pt_[channels_[ch]] = -99;
    b_ak8jet0_subjet0_csv_[channels_[ch]] = -99;
    b_ak8jet0_subjet1_csv_[channels_[ch]] = -99;
    b_ak8jet0_puppi_subjet0_csv_[channels_[ch]] = -99;
    b_ak8jet0_puppi_subjet1_csv_[channels_[ch]] = -99;

    b_HT_[channels_[ch]] = -99;

    b_tau_pt_[channels_[ch]] =-99;
    b_tau_eta_[channels_[ch]] =-99;
    b_tau_phi_[channels_[ch]] =-99;
    b_tau_Iso_[channels_[ch]] =-99;
    b_tau_IsoRaw_[channels_[ch]] =-99;
    b_tau_againstEle_[channels_[ch]] =-99;
    b_tau_againstMu_[channels_[ch]] =-99;
    b_seclepton_pt_[channels_[ch]] =-99;
    b_seclepton_eta_[channels_[ch]] =-99;
    b_seclepton_phi_[channels_[ch]] =-99;
    b_seclepton_iso_[channels_[ch]] =-99;
    b_met_[channels_[ch]] =-99;
    b_met_phi_[channels_[ch]] =-99;
    b_met_phi_ak4_[channels_[ch]] =-99;
    b_vis_mass_ll_[channels_[ch]] =-99;
    b_dR_ll_[channels_[ch]] =-99;

    b_H_Mass_SVFit_[channels_[ch]] =-99;
    b_H_Pt_SVFit_[channels_[ch]] =-99;
    b_H_Eta_SVFit_[channels_[ch]] =-99;
    b_H_Phi_SVFit_[channels_[ch]] =-99;
    b_X_Mass_SVFit_[channels_[ch]] =-99;
    b_X_Pt_SVFit_[channels_[ch]] =-99;
    b_X_Eta_SVFit_[channels_[ch]] =-99;
    b_X_Phi_SVFit_[channels_[ch]] =-99;
 

    if( m_doSVFit_new){
      b_H_Mass_SVFit_new_[channels_[ch]] =-99;
      b_H_Pt_SVFit_new_[channels_[ch]] =-99;
      b_H_Eta_SVFit_new_[channels_[ch]] =-99;
      b_H_Phi_SVFit_new_[channels_[ch]] =-99;
      b_X_Mass_SVFit_new_[channels_[ch]] =-99;
      b_X_Pt_SVFit_new_[channels_[ch]] =-99;
      b_X_Eta_SVFit_new_[channels_[ch]] =-99;
      b_X_Phi_SVFit_new_[channels_[ch]] =-99;
 
    }

    b_H_Mass_Collinear_[channels_[ch]] =-99;
    b_H_Pt_Collinear_[channels_[ch]] =-99;
    b_H_Eta_Collinear_[channels_[ch]] =-99;
    b_H_Phi_Collinear_[channels_[ch]] =-99;
    b_X_Mass_Collinear_[channels_[ch]] =-99;
    b_X_Pt_Collinear_[channels_[ch]] =-99;
    b_X_Eta_Collinear_[channels_[ch]] =-99;
    b_X_Phi_Collinear_[channels_[ch]] =-99;

    b_trig_HT800_[channels_[ch]] =false;
    b_trig_HT900_[channels_[ch]] =false;
    b_trig_HT650_[channels_[ch]] =false;
    b_trig_HT650_WideJetMJJ950DEtaJJ1p5_[channels_[ch]] =false;
    b_trig_HT650_WideJetMJJ900DEtaJJ1p5_[channels_[ch]] =false;

    b_trig_METNoMu90_[channels_[ch]]=false;
    b_trig_METNoMu110_[channels_[ch]]=false;
    b_trig_METNoMu120_[channels_[ch]]=false;
    b_trig_METNoMu90_JetId_[channels_[ch]]=false;
    b_trig_METNoMu110_JetId_[channels_[ch]]=false;
    b_trig_METNoMu120_JetId_[channels_[ch]]=false;
    b_trig_MET110_[channels_[ch]]=false;
    b_trig_MET120_[channels_[ch]]=false; 
    b_trig_MET170_[channels_[ch]]=false;
    


    b_trig_ak8Jet360_Trim30_[channels_[ch]] =false;
    b_trig_Ele22Loose_[channels_[ch]] =false;
    b_trig_Ele27Loose_[channels_[ch]] =false;
    b_trig_Ele27Tight_[channels_[ch]] =false;
    b_trig_Ele115_[channels_[ch]] =false;
    b_trig_Mu20_[channels_[ch]] =false;
    b_trig_Mu22_[channels_[ch]] =false;
    b_trig_Mu22_eta2p1_[channels_[ch]] =false;
    b_trig_Mu24_[channels_[ch]] =false;
    b_trig_Mu27_[channels_[ch]] =false;
    b_trig_Mu50_[channels_[ch]] =false;

    b_selection_bits_.clear();
    b_selection_lastcut_.clear();


    b_N_AK4_[channels_[ch]] = -99 ;
    b_N_taggedAK4_L_[channels_[ch]] = -99 ;
    b_N_taggedAK4_M_[channels_[ch]] = -99;
    b_N_taggedAK4_T_[channels_[ch]] = -99 ;
    b_isBoostedTau_[channels_[ch]]=false;
    b_PV_N_[channels_[ch]] =-99;
    b_N_tight_muons_[channels_[ch]] =-99;
    b_N_tight_electrons_[channels_[ch]] =-99;
    b_N_tight_leptons_[channels_[ch]] =-99;
    b_N_loose_muons_[channels_[ch]] =-99;
    b_N_loose_electrons_[channels_[ch]] =-99;

    b_GenEvent_Htata_[channels_[ch]]=false;
    b_GenEvent_Ztata_[channels_[ch]]=false;
    b_GenEvent_Gammatata_[channels_[ch]]=false;
    // b_ChannelInt_[channels_[ch]] =-99;

    if (m_doGenRecoMatchVariables){
      b_dRG_tau_lep_[channels_[ch]] =-99;
      b_dPtRG_tau_[channels_[ch]] =-99;
      b_dRRG_tau_[channels_[ch]] =-99;
      b_N_hadronicTaus_[channels_[ch]] =-99;
      b_N_leptonicTaus_[channels_[ch]] =-99;
      b_dPtRG_lepton_[channels_[ch]] =-99;
      b_dRRG_lepton_[channels_[ch]] =-99;
      b_dPtRG_Res_[channels_[ch]] =-99;
      b_dRRG_Res_[channels_[ch]] =-99;
      b_dPtRG_Boson_qq_[channels_[ch]] =-99;
      b_dRRG_Boson_qq_[channels_[ch]] =-99;
      b_dPtRG_Higgs_tata_[channels_[ch]] =-99;
      b_dRRG_Higgs_tata_[channels_[ch]] =-99;
      b_IdentityRG_tau_[channels_[ch]] =-99;
      b_IdentityRG_lepton_[channels_[ch]] =-99;
      b_IdentityRG_Boson_qq_[channels_[ch]] =-99;
      b_IdentityRG_Higgs_tata_[channels_[ch]] =-99;

    }


    if (m_doShapeSyst){
      b_X_Mass_Collinear_jesUp_[channels_[ch]] =-99;
      b_H_Mass_Collinear_jesUp_[channels_[ch]] =-99;
      b_met_jesUp_[channels_[ch]] =-99;
      b_ak8jet0_pt_jesUp_[channels_[ch]] =-99;

      b_X_Mass_Collinear_jesDown_[channels_[ch]] =-99;
      b_H_Mass_Collinear_jesDown_[channels_[ch]] =-99;
      b_met_jesDown_[channels_[ch]] =-99;
      b_ak8jet0_pt_jesDown_[channels_[ch]] =-99;

      b_X_Mass_Collinear_jerUp_[channels_[ch]] =-99;
      b_H_Mass_Collinear_jerUp_[channels_[ch]] =-99;
      b_met_jerUp_[channels_[ch]] =-99;
      b_ak8jet0_pt_jerUp_[channels_[ch]] =-99;

      b_X_Mass_Collinear_jerDown_[channels_[ch]] =-99;
      b_H_Mass_Collinear_jerDown_[channels_[ch]] =-99;
      b_met_jerDown_[channels_[ch]] =-99;
      b_ak8jet0_pt_jerDown_[channels_[ch]] =-99;

      b_X_Mass_Collinear_jer_[channels_[ch]] =-99;
      b_H_Mass_Collinear_jer_[channels_[ch]] =-99;
      b_met_jer_[channels_[ch]] =-99;
      b_ak8jet0_pt_jer_[channels_[ch]] =-99;

      b_X_Mass_Collinear_UncEnUp_[channels_[ch]] =-99;
      b_H_Mass_Collinear_UncEnUp_[channels_[ch]] =-99;
      b_met_UncEnUp_[channels_[ch]] =-99;
    
      b_X_Mass_Collinear_UncEnDown_[channels_[ch]] =-99;
      b_H_Mass_Collinear_UncEnDown_[channels_[ch]] =-99;
      b_met_UncEnDown_[channels_[ch]] =-99;
     
      b_X_Mass_Collinear_tesUp_[channels_[ch]] =-99;
      b_H_Mass_Collinear_tesUp_[channels_[ch]] =-99;
      b_met_tesUp_[channels_[ch]] =-99;
    
      b_X_Mass_Collinear_tesDown_[channels_[ch]] =-99;
      b_H_Mass_Collinear_tesDown_[channels_[ch]] =-99;
      b_met_tesDown_[channels_[ch]] =-99;
    

    }
  }
  
}

void VHTausAnalysis::fillCutflow( const std::string histName, const std::string dirName, const TBits& cutmap, const Double_t weight ) {

  // bool writeNtuple = false;
  // sequential cut flow -> stop at first failed cut
  // m_logger << INFO << histName << "\t" << dirName << SLogger::endmsg;
  for( UInt_t i = 0; i < cutmap.GetNbits(); ++i ) {
    // m_logger << INFO << i << ":\t" << cutmap.TestBitNumber( i ) << SLogger::endmsg;
    if( cutmap.TestBitNumber( i ) ) {
      Hist( histName.c_str(), dirName.c_str() )->Fill( i+1, weight );
      // if (i == (unsigned int) m_ntupleLevel) {
      //   writeNtuple = true;
      // }
    } else {
      break;
    }
  }
  // if (!writeNtuple) {
  // // this does something really bad...
  //   throw SError( SError::SkipEvent );
  // }
}
void VHTausAnalysis::bookTriggerHistos(const TString& directory  ){
  Book( TH1F( "Trigger", "Trigger",60,0,60  ), directory);
  Book( TH1F( "Trigger_Tau", "Trigger_Tau", 60,0,60), directory);
  Book( TH1F( "Trigger_Jet", "Trigger_Jet", 60,0,60), directory);
  Book( TH1F( "Trigger_TauOrJet", "Trigger_TauOrJet",10,-1.5,8.5  ) , directory);

  Book( TH1F( "Trigger_Offline", "Trigger offline selection",60,0,60  ), directory);
  Book( TH1F( "Trigger_Tau_Offline", "Trigger_Tau offline selection", 60,0,60), directory);
  Book( TH1F( "Trigger_Jet_Offline", "Trigger_Jet offline selection", 60,0,60), directory);
  Book( TH1F( "Trigger_TauOrJet_Offline", "Trigger_TauOrJet  offline selection",10,-1.5,8.5  ) , directory);
  Book( TH1F( "Trigger_Attempts", "Trigger attempts fro 2016",30,0,30 ) , directory);

 

  std::vector<std::string> trignames;

  trignames.push_back("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v");
  trignames.push_back("HLT_AK8PFJet360_TrimMass30_v");
  trignames.push_back("HLT_AK8PFHT650_TrimR0p1PT0p03Mass50_v");
  trignames.push_back("HLT_AK8PFHT660_TrimR0p1PT0p03Mass50_BTagCSV_p20_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p45_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p41_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20_v");
  trignames.push_back("HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20_v");
  trignames.push_back("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v");
  trignames.push_back("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v");
  trignames.push_back("HLT_PFHT600_v");
  trignames.push_back("HLT_PFHT650_v") ;
  trignames.push_back("HLT_PFHT800_v") ;
  trignames.push_back("HLT_PFHT900_v") ;
  trignames.push_back("HLT_PFJet320_v") ;
  trignames.push_back("HLT_PFJet450_v") ;  
    
  trignames.push_back("HLT_IsoMu20_eta2p1_v") ;
  trignames.push_back("HLT_IsoMu24_eta2p1_v") ;
  trignames.push_back("HLT_Mu45_eta2p1_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v") ;
  trignames.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele105_CaloIdVT_GsfTrkIdT_v") ;
  trignames.push_back("HLT_Ele115_CaloIdVT_GsfTrkIdT_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau35_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_IsoMu16_eta2p1_MET30_JetIDCleaned_LooseIsoPFTau50_v") ;
  //H->tautau triggers   
  trignames.push_back("HLT_LooseIsoPFTau50_v") ;
  trignames.push_back("HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_IsoMu17_eta2p1_v") ;
  trignames.push_back("HLT_IsoMu18_v") ;
  trignames.push_back("HLT_IsoMu27_v") ;
  trignames.push_back("HLT_IsoMu20_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WP75_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele23_WPLoose_Gsf_v") ;
  trignames.push_back("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v") ;
  trignames.push_back("HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v") ;

  //Alternative triggers
  trignames.push_back("HLT_PFMET120_PFMHT120_IDTight_v") ;
  trignames.push_back("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v"); 
  trignames.push_back("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");

  trignames.push_back("HLT_Ele15_IsoVVVL_PFHT600");
  trignames.push_back("HLT_Mu15_IsoVVVL_PFHT600");
  trignames.push_back("HLT_Ele15_IsoVVVL_PFHT350_PFMET50");
  trignames.push_back("HLT_Mu15_IsoVVVL_PFHT350_PFMET50");
  trignames.push_back("HLT_Ele15_IsoVVVL_BTagCSV0p72_PFHT400"); 
  trignames.push_back("HLT_Mu15_IsoVVVL_BTagCSV0p72_PFHT400");
 
 
  std::vector<std::string> trignamesTau;
  //commenting out the prescaled ones
  // trignamesTau.push_back("HLT_IsoMu20_eta2p1_v");
  // trignamesTau.push_back("HLT_IsoMu24_eta2p1_v");
  // trignamesTau.push_back("HLT_Mu45_eta2p1_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WP75_Gsf_v");
  // trignamesTau.push_back("HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v");
  // trignamesTau.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v");
  // trignamesTau.push_back("HLT_Ele105_CaloIdVT_GsfTrkIdT_v");
  // trignamesTau.push_back("HLT_Ele115_CaloIdVT_GsfTrkIdT_v");
  // trignamesTau.push_back("HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau35_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v");
  // trignamesTau.push_back("HLT_IsoMu16_eta2p1_MET30_JetIDCleaned_LooseIsoPFTau50_v");  
  //H->tautau triggers   
  // trignamesTau.push_back("HLT_LooseIsoPFTau50_v");
  trignamesTau.push_back("HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v");
  trignamesTau.push_back("HLT_IsoMu24_eta2p1_v");
  trignamesTau.push_back("HLT_IsoMu17_eta2p1_v");
  trignamesTau.push_back("HLT_IsoMu18_v");
  trignamesTau.push_back("HLT_IsoMu27_v");
  trignamesTau.push_back("HLT_IsoMu20_v");
  trignamesTau.push_back("HLT_Ele22_eta2p1_WP75_Gsf_LooseIsoPFTau20_v");
  trignamesTau.push_back("HLT_Ele22_eta2p1_WP75_Gsf_v");
  trignamesTau.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v");
  trignamesTau.push_back("HLT_Ele23_WPLoose_Gsf_v");
  trignamesTau.push_back("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v");
  trignamesTau.push_back("HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v");
 


  //Alternative triggers
  trignamesTau.push_back("HLT_PFMET120_PFMHT120_IDTight_v");
  trignamesTau.push_back("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");
  trignamesTau.push_back("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");

 
  std::vector<std::string> trignamesJet;
  trignamesJet.push_back("HLT_AK8PFJet360_TrimMass30_v") ;
  trignamesJet.push_back("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v") ;
  trignamesJet.push_back("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v") ;
  trignamesJet.push_back("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v") ;
  trignamesJet.push_back("HLT_PFHT800_v") ;
  trignamesJet.push_back("HLT_PFHT900_v") ;



  const char *TriggerNames[trignames.size()];
  const char *TriggerTauNames[trignamesTau.size()];
  const char *TriggerJetNames[trignamesJet.size()];




  Hist( "Trigger"  	, directory )->GetXaxis()->SetTitle( "Triggers" );
  Hist( "Trigger"  	, directory )->GetXaxis()->LabelsOption("v");
  Hist( "Trigger_Tau"  	, directory )->GetXaxis()->SetTitle( "Triggers" );
  Hist( "Trigger_Tau"  	, directory )->GetXaxis()->LabelsOption("v");
  Hist( "Trigger_Jet"  	, directory )->GetXaxis()->SetTitle( "Triggers" );
  Hist( "Trigger_Jet"   , directory )->GetXaxis()->LabelsOption("v");
 
  Hist( "Trigger_Offline"  	, directory )->GetXaxis()->SetTitle( "Triggers" );
  // Hist( "Trigger_Offline"  	, directory )->GetXaxis()->LabelsOption("v");
  Hist( "Trigger_Tau_Offline"  	, directory )->GetXaxis()->SetTitle( "Triggers" );
  //Hist( "Trigger_TaU_Offline"  	, directory )->GetXaxis()->LabelsOption("v");
  Hist( "Trigger_Jet_Offline"  	, directory )->GetXaxis()->SetTitle( "Triggers" );
  //Hist( "Trigger_Jet_Offline"   , directory )->GetXaxis()->LabelsOption("v");
 


  for ( unsigned int t=0; t< trignames.size(); t++){
    TriggerNames[t]= trignames[t].c_str();
    Hist( "Trigger"  , directory 	)->GetXaxis()->SetBinLabel(t+1, TriggerNames[t] );
    Hist( "Trigger_Offline"  , directory 	)->GetXaxis()->SetBinLabel(t+1, TriggerNames[t] );
  }

  for ( unsigned int t=0; t< trignamesTau.size(); t++){ 
    TriggerTauNames[t]= trignamesTau[t].c_str();
    Hist( "Trigger_Tau"  , directory 	)->GetXaxis()->SetBinLabel(t+1, TriggerTauNames[t] );
    Hist( "Trigger_Tau_Offline"  , directory 	)->GetXaxis()->SetBinLabel(t+1, TriggerTauNames[t] );
    }

  for ( unsigned int t=0; t< trignamesJet.size(); t++){ 
    TriggerJetNames[t]= trignamesJet[t].c_str();
    Hist( "Trigger_Jet"  , directory 	)->GetXaxis()->SetBinLabel(t+1, TriggerJetNames[t] );
    Hist( "Trigger_Jet_Offline"  , directory 	)->GetXaxis()->SetBinLabel(t+1, TriggerJetNames[t] );
  }

  std::vector<std::string> trignames_attempts;

  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass20_v1");
  trignames_attempts.push_back("HLT_HT400_MET50_AK8PFJet200_TrimMass20_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass30_v1");
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet200_TrimMass30_v1"); 
  // trignames_attempts.push_back("HLT_HT200_MET50_AK8DiPFJet120_TrimMass20_v1");
  // trignames_attempts.push_back("HLT_HT250_MET50_AK8DiPFJet120_TrimMass20_v1");
  // trignames_attempts.push_back("HLT_HT300_MET50_AK8DiPFJet120_TrimMass20_v1");
  // trignames_attempts.push_back("HLT_HT300_MET50_AK8DiPFJet200_TrimMass20_v1");
  trignames_attempts.push_back("HLT_HT200_MET50_AK8PFJet120_TrimMass20_LooseIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT400_LooseIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT450_LooseIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT450_LooseIsoPFTau30_Trk30_v1");
  trignames_attempts.push_back("HLT_HT200_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT400_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass30_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet200_TrimMass30_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT350_MET60_AK8PFJet200_TrimMass20_NotIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet200_TrimMass20_NotIsoPFTau50_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet120_TrimMass30_NotIsoPFTau50_v1");
  trignames_attempts.push_back("HLT_HT400_MET50_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT400_MET70_NotIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT450_MET50_NotIsoPFTau50_Trk30_v1"); 

  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet120_TrimMass30_NotIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT300_MET60_AK8PFJet120_TrimMass30_NotIsoPFTau50_Trk30_v1");  
  trignames_attempts.push_back("HLT_HT400_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");  

  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30_v1");   
  trignames_attempts.push_back("HLT_HT300_MET60_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30v1"); 
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30v1"); 
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet150_TrimMass20_NotIsoPFTau50_Trk30v1"); 


  const char *TriggerAttemptsNames[trignames_attempts.size()];
  for ( unsigned int t=0; t< trignames_attempts.size(); t++){ 
    TriggerAttemptsNames[t]= trignames_attempts[t].c_str();
    Hist( "Trigger_Attempts"  , directory 	)->GetXaxis()->SetBinLabel(t+1,   TriggerAttemptsNames[t] );
 
  }

}


void VHTausAnalysis::bookHistograms( const TString& directory ) {
  
  // kinematics histograms
  Book( TH1F( "vjet_pt", "Vjet p_{T};Vjet p_{T} [GeV]", 200, 0, 2000 ), directory ); 
  Book( TH1F( "vjet_eta", "Vjet #eta;Vjet #eta", 50, -2.5, 2.5 ), directory ); 
  Book( TH1F( "vjet_phi", "Vjet #phi;Vjet #phi", 50, -3.15, 3.15 ), directory ); 
  Book( TH1F( "vjet_m", "Vjet m;Vjet m [GeV]", 40, 0, 200 ), directory ); 
  Book( TH1F( "vjet_mpruned", "Vjet softdrop m;Vjet softdrop m [GeV]", 40, 0, 200 ), directory ); 
  Book( TH1F( "vjet_tau1", "Vjet #tau_{1};Vjet #tau_{1}", 50, 0, 1 ), directory ); 
  Book( TH1F( "vjet_tau2", "Vjet #tau_{2};Vjet #tau_{2}", 50, 0, 1 ), directory ); 
  Book( TH1F( "vjet_tau3", "Vjet #tau_{3};Vjet #tau_{3}", 50, 0, 1 ), directory ); 
  Book( TH1F( "vjet_tau21", "Vjet #tau_{21};Vjet #tau_{21}", 50, 0, 1 ), directory ); 
  Book( TH1F( "vjet_tau31", "Vjet #tau_{31};Vjet #tau_{31}", 50, 0, 1 ), directory ); 
  Book( TH1F( "vjet_tau32", "Vjet #tau_{32};Vjet #tau_{32}", 50, 0, 1 ), directory ); 
  Book( TH1F( "vjet_nSubjets", "Vjet N subjets;Vjet N subjets", 10, -.5, 9.5 ), directory ); 
  Book( TH1F( "vjet_nTaggedSubjets", "Vjet N tagged subjets;Vjet N tagged subjets", 10, -.5, 9.5 ), directory ); 
  Book( TH1F( "vjet_subjet0_csv", "Vjet subjet 0 CSV;Vjet subjet0 CSV", 50, 0, 1 ), directory ); 
  Book( TH1F( "vjet_subjet1_csv", "Vjet subjet 1 CSV;Vjet subjet1 CSV", 50, 0, 1 ), directory ); 

  Book( TH1F( "hjet_pt", "Hjet p_{T};Hjet p_{T} [GeV]", 200, 0, 2000 ), directory ); 
  Book( TH1F( "hjet_eta", "Hjet #eta;Hjet #eta", 50, -2.5, 2.5 ), directory ); 
  Book( TH1F( "hjet_phi", "Hjet #phi;Hjet #phi", 50, -3.15, 3.15 ), directory ); 
  Book( TH1F( "hjet_m", "Hjet m;Hjet m [GeV]", 40, 0, 200 ), directory ); 
  Book( TH1F( "hjet_mpruned", "Hjet softdrop m;Hjet softdrop m [GeV]", 40, 0, 200 ), directory ); 
  Book( TH1F( "hjet_tau1", "Hjet #tau_{1};Hjet #tau_{1}", 50, 0, 1 ), directory ); 
  Book( TH1F( "hjet_tau2", "Hjet #tau_{2};Hjet #tau_{2}", 50, 0, 1 ), directory ); 
  Book( TH1F( "hjet_tau3", "Hjet #tau_{3};Hjet #tau_{3}", 50, 0, 1 ), directory ); 
  Book( TH1F( "hjet_tau21", "Hjet #tau_{21};Hjet #tau_{21}", 50, 0, 1 ), directory ); 
  Book( TH1F( "hjet_tau31", "Hjet #tau_{31};Hjet #tau_{31}", 50, 0, 1 ), directory ); 
  Book( TH1F( "hjet_tau32", "Hjet #tau_{32};Hjet #tau_{32}", 50, 0, 1 ), directory ); 
  Book( TH1F( "hjet_nSubjets", "Hjet N subjets;Hjet N subjets", 10, -.5, 9.5 ), directory ); 
  Book( TH1F( "hjet_nTaggedSubjets", "Hjet N tagged subjets;Hjet N tagged subjets", 10, -.5, 9.5 ), directory ); 
  Book( TH1F( "hjet_subjet0_csv", "Hjet subjet 0 CSV;Hjet subjet0 CSV", 100, -1, 1 ), directory ); 
  Book( TH1F( "hjet_subjet1_csv", "Hjet subjet 1 CSV;Hjet subjet1 CSV", 100, -1, 1 ), directory ); 

  Book( TH1F( "jets_deta", "jets #Delta #eta;jets #Delta #eta", 50, 0, 5 ), directory ); 
  Book( TH1F( "jets_dphi", "jets #Delta #phi;jets #Delta #phi", 50, 0, 6.3 ), directory ); 
  Book( TH1F( "jets_dr", "jets #Delta R;jets #Delta R", 50, 0, 5 ), directory ); 

  Book( TH1F( "dijet_pt", "dijet p_{T};dijet p_{T} [GeV]", 100, 0, 1000 ), directory ); 
  Book( TH1F( "dijet_eta", "dijet #eta;dijet #eta", 50, -2.5, 2.5 ), directory ); 
  Book( TH1F( "dijet_phi", "dijet #phi;Vdijet #phi", 50, -3.15, 3.15 ), directory ); 
  Book( TH1F( "dijet_m", "dijet m;dijet m [GeV]", 400, 0, 4000 ), directory );
  Book( TH1F( "dijet_template_m", "dijet m;dijet m [GeV]", 7000, 0, 7000 ), directory ); 
  
}



void VHTausAnalysis::fillHistograms( const TString& directory, const UZH::Jet& vectorJet, const UZH::Jet& higgsJet, const TLorentzVector& diJet, const double& vJet_tau21, const double& vJet_tau31, const double& vJet_tau32, const int& vJet_nTaggedSubjets, const double& vJet_subjet0_csv, const double& vJet_subjet1_csv, const double& hJet_tau21, const double& hJet_tau31, const double& hJet_tau32, const int& hJet_nTaggedSubjets, const double& hJet_subjet0_csv, const double& hJet_subjet1_csv, const double& deta, const double& dphi, const double& dr ) {
  
  // fill all histograms
  Hist( "vjet_pt", directory )->Fill( vectorJet.pt() , b_weight);
  Hist( "vjet_eta", directory )->Fill( vectorJet.eta() , b_weight);
  Hist( "vjet_phi", directory )->Fill( vectorJet.phi() , b_weight);
  Hist( "vjet_m", directory )->Fill( vectorJet.m() , b_weight);
  Hist( "vjet_mpruned", directory )->Fill( vectorJet.pruned_massCorr() , b_weight);
  Hist( "vjet_tau1", directory )->Fill( vectorJet.tau1() , b_weight);
  Hist( "vjet_tau2", directory )->Fill( vectorJet.tau2() , b_weight);
  Hist( "vjet_tau3", directory )->Fill( vectorJet.tau3() , b_weight);
  Hist( "vjet_tau21", directory )->Fill( vJet_tau21 , b_weight);
  Hist( "vjet_tau31", directory )->Fill( vJet_tau31 , b_weight);
  Hist( "vjet_tau32", directory )->Fill( vJet_tau32 , b_weight);
  Hist( "vjet_nSubjets", directory )->Fill( vectorJet.subjet_softdrop_N() , b_weight);
  Hist( "vjet_nTaggedSubjets", directory )->Fill( vJet_nTaggedSubjets , b_weight);
  Hist( "vjet_subjet0_csv", directory )->Fill( vJet_subjet0_csv , b_weight);
  Hist( "vjet_subjet1_csv", directory )->Fill( vJet_subjet1_csv , b_weight);

  Hist( "hjet_pt", directory )->Fill( higgsJet.pt() , b_weight);
  Hist( "hjet_eta", directory )->Fill( higgsJet.eta() , b_weight);
  Hist( "hjet_phi", directory )->Fill( higgsJet.phi() , b_weight);
  Hist( "hjet_m", directory )->Fill( higgsJet.m() , b_weight);
  Hist( "hjet_mpruned", directory )->Fill( higgsJet.pruned_massCorr() , b_weight);
  Hist( "hjet_tau1", directory )->Fill( higgsJet.tau1() , b_weight);
  Hist( "hjet_tau2", directory )->Fill( higgsJet.tau2() , b_weight);
  Hist( "hjet_tau3", directory )->Fill( higgsJet.tau3() , b_weight);
  Hist( "hjet_tau21", directory )->Fill( hJet_tau21 , b_weight);
  Hist( "hjet_tau31", directory )->Fill( hJet_tau31 , b_weight);
  Hist( "hjet_tau32", directory )->Fill( hJet_tau32 , b_weight);
  Hist( "hjet_nSubjets", directory )->Fill( higgsJet.subjet_softdrop_N() , b_weight);
  Hist( "hjet_nTaggedSubjets", directory )->Fill( hJet_nTaggedSubjets , b_weight);
  Hist( "hjet_subjet0_csv", directory )->Fill( hJet_subjet0_csv , b_weight);
  Hist( "hjet_subjet1_csv", directory )->Fill( hJet_subjet1_csv , b_weight);

  Hist( "jets_deta", directory )->Fill( deta , b_weight);
  Hist( "jets_dphi", directory )->Fill( dphi , b_weight);
  Hist( "jets_dr", directory )->Fill( dr , b_weight);

  Hist( "dijet_pt", directory )->Fill( diJet.Pt() , b_weight);
  Hist( "dijet_eta", directory )->Fill( diJet.Eta() , b_weight);
  Hist( "dijet_phi", directory )->Fill( diJet.Phi() , b_weight);
  Hist( "dijet_m", directory )->Fill( diJet.M() , b_weight);
  Hist( "dijet_template_m", directory )->Fill( diJet.M() , b_weight);
  
}


void VHTausAnalysis::checkTriggerOfflineSelection(const TString& directory,  const UZH::Jet& Jet,const UZH::Tau& tau,const  TLorentzVector& lepton, const  bool& ID,const  double& leptonIso, const UZH::MissingEt& met,const std::string& channel,const  double& HT  ){
  // std:: cout << "directory " <<  directory << " channel " << channel <<" jet pt " <<  Jet.pt() <<" tau tlv  pt " << tau.tlv().Pt()<< " lepton pt " << lepton.Pt() << std::endl;
  std::vector<std::string> trignames;

  trignames.push_back("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v");
  trignames.push_back("HLT_AK8PFJet360_TrimMass30_v");
  trignames.push_back("HLT_AK8PFHT650_TrimR0p1PT0p03Mass50_v");
  trignames.push_back("HLT_AK8PFHT660_TrimR0p1PT0p03Mass50_BTagCSV_p20_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p45_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p41_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20_v");
  trignames.push_back("HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20_v");

  trignames.push_back("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v");

  trignames.push_back("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v");
  trignames.push_back("HLT_PFHT600_v");
  trignames.push_back("HLT_PFHT650_v") ;
  trignames.push_back("HLT_PFHT800_v") ;
  trignames.push_back("HLT_PFHT900_v") ;
  trignames.push_back("HLT_PFJet320_v") ;
  trignames.push_back("HLT_PFJet450_v") ;  
    
  trignames.push_back("HLT_IsoMu20_eta2p1_v") ;
  trignames.push_back("HLT_IsoMu24_eta2p1_v") ;
  trignames.push_back("HLT_Mu45_eta2p1_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v") ;
  trignames.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele105_CaloIdVT_GsfTrkIdT_v") ;
  trignames.push_back("HLT_Ele115_CaloIdVT_GsfTrkIdT_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau35_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_IsoMu16_eta2p1_MET30_JetIDCleaned_LooseIsoPFTau50_v") ;
  //H->tautau triggers   
  trignames.push_back("HLT_LooseIsoPFTau50_v") ;
  trignames.push_back("HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_IsoMu17_eta2p1_v") ;
  trignames.push_back("HLT_IsoMu18_v") ;
  trignames.push_back("HLT_IsoMu27_v") ;
  trignames.push_back("HLT_IsoMu20_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WP75_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele23_WPLoose_Gsf_v") ;
  trignames.push_back("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v") ;
  trignames.push_back("HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v") ;

  //Alternative triggers
  trignames.push_back("HLT_PFMET120_PFMHT120_IDTight_v") ;
  trignames.push_back("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v"); 
  trignames.push_back("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");
  trignames.push_back("HLT_Ele15_IsoVVVL_PFHT600");
  trignames.push_back("HLT_Mu15_IsoVVVL_PFHT600");
  trignames.push_back("HLT_Ele15_IsoVVVL_PFHT350_PFMET50");
  trignames.push_back("HLT_Mu15_IsoVVVL_PFHT350_PFMET50");
  trignames.push_back("HLT_Ele15_IsoVVVL_BTagCSV0p72_PFHT400"); 
  trignames.push_back("HLT_Mu15_IsoVVVL_BTagCSV0p72_PFHT400");
 
 
  std::vector<std::string> trignamesTau;
  //commenting out the prescaled ones
  // trignamesTau.push_back("HLT_IsoMu20_eta2p1_v");
  // trignamesTau.push_back("HLT_IsoMu24_eta2p1_v");
  // trignamesTau.push_back("HLT_Mu45_eta2p1_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WP75_Gsf_v");
  // trignamesTau.push_back("HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v");
  // trignamesTau.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v");
  // trignamesTau.push_back("HLT_Ele105_CaloIdVT_GsfTrkIdT_v");
  // trignamesTau.push_back("HLT_Ele115_CaloIdVT_GsfTrkIdT_v");
  // trignamesTau.push_back("HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau35_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v");
  // trignamesTau.push_back("HLT_IsoMu16_eta2p1_MET30_JetIDCleaned_LooseIsoPFTau50_v");  
  //H->tautau triggers   
  // trignamesTau.push_back("HLT_LooseIsoPFTau50_v");
  trignamesTau.push_back("HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v");
  trignamesTau.push_back("HLT_IsoMu24_eta2p1_v");
  trignamesTau.push_back("HLT_IsoMu17_eta2p1_v");
  trignamesTau.push_back("HLT_IsoMu18_v");
  trignamesTau.push_back("HLT_IsoMu27_v");
  trignamesTau.push_back("HLT_IsoMu20_v");
  trignamesTau.push_back("HLT_Ele22_eta2p1_WP75_Gsf_LooseIsoPFTau20_v");
  trignamesTau.push_back("HLT_Ele22_eta2p1_WP75_Gsf_v");
  trignamesTau.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v");
  trignamesTau.push_back("HLT_Ele23_WPLoose_Gsf_v");
  trignamesTau.push_back("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v");
  trignamesTau.push_back("HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v");


  //Alternative triggers
  trignamesTau.push_back("HLT_PFMET120_PFMHT120_IDTight_v");
  trignamesTau.push_back("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");
  trignamesTau.push_back("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");

 
  std::vector<std::string> trignamesJet;
  trignamesJet.push_back("HLT_AK8PFJet360_TrimMass30_v") ;
  trignamesJet.push_back("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v") ;
  // trignames.push_back("AK8DiPFJet280_200_TrimMass30_BTagCSV0p45") ;
  trignamesJet.push_back("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v") ;
  trignamesJet.push_back("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v") ;
  trignamesJet.push_back("HLT_PFHT800_v") ;
  trignamesJet.push_back("HLT_PFHT900_v") ;

  
  std::string GENChannel_="General";
  bool isfired = false;
 
  bool isfired_Offline = false;
 
  if (GENChannel_=="General"){
    bool isfiredTau = false;
    bool isfiredJet = false;
    bool isfiredTau_Offline = false;
    bool isfiredJet_Offline = false;
 
    for( std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
      //   std::cout <<"TriggerFired : "<< (it->first) <<std::endl; 

      for( unsigned int t = 0; t < trignames.size(); ++t ){
      
	if( (it->first).find(trignames[t]) != std::string::npos){
	  // std::cout <<"TriggerFired : "<< (it->first) <<" and number "<<t<< std::endl; 
	  if(it->second  ){
	    isfired = true;
	    if (isfired &&( trignames[t].find("AK8") != std::string::npos|| trignames[t].find("HT") != std::string::npos )) {
	      if(Jet.pt()>500)	    Hist( "Trigger_Offline",directory )->Fill( t );
	    }
	    else   if (isfired && trignames[t].find("Ele") != std::string::npos ){
	      if( trignames[t].find("Ele105") == std::string::npos  && lepton.Pt()> 30  ){///not trigger ele 105, unisolated
		Hist( "Trigger_Offline",directory )->Fill( t );	
	      }
	      else if (isfired && lepton.Pt()>110){
		Hist( "Trigger_Offline",directory )->Fill( t );	
	      }
	    }
	    else if(isfired && trignames[t].find("Mu") != std::string::npos){
	      if( trignames[t].find("Mu45") == std::string::npos  && lepton.Pt()> 22){///not trigger ele 105, unisolated
		Hist( "Trigger_Offline",directory )->Fill( t );	
	      }
	      else if (isfired && lepton.Pt()>50){
		Hist( "Trigger_Offline",directory )->Fill( t );	
	      }
	    }else if(isfired && trignames[t].find("DoubleMediumIsoPFTau") != std::string::npos){
	      if (tau.tlv().Pt()>50 && lepton.Pt() >50)  Hist( "Trigger_Offline",directory )->Fill( t );	
	    }  
	    // else std::cout <<"Trigger NOT Fired : "<< trignames[t] <<std::endl;  	
	  }
	}
      }      
      
      
      for( unsigned int t = 0; t < trignamesJet.size(); ++t ){
	if( (it->first).find(trignamesJet[t]) != std::string::npos &&  it->second ){
	  isfiredJet = true;
	  if (isfiredJet &&( trignamesJet[t].find("AK8") != std::string::npos|| trignamesJet[t].find("HT") != std::string::npos )) {
	    if(Jet.pt()>500)	 {
	      Hist( "Trigger_Jet_Offline",directory )->Fill( t );
	      isfiredJet_Offline = true;
	    }
	  }
	}
      }
      for( unsigned int t = 0; t < trignamesTau.size(); ++t ){
	if( (it->first).find(trignamesTau[t]) != std::string::npos && it->second  // && !isfiredJet
	    ){
	  isfiredTau = true;
	  if (isfiredTau&& trignamesTau[t].find("Ele") != std::string::npos  && channel=="eletau"){
	    if( trignamesTau[t].find("Ele105") == std::string::npos   && channel=="eletau" && ID && lepton.Pt()> 30){///not trigger ele 105, unisolated
	      Hist( "Trigger_Tau_Offline",directory )->Fill( t );	
	      isfiredTau_Offline = true;
	    }
	    else if (isfiredTau &&lepton.Pt()>110 && ID && channel=="eletau"){
	      Hist( "Trigger_Tau_Offline",directory )->Fill( t );	
	      isfiredTau_Offline = true;
	    }
	  }
	  else if(isfiredTau && trignamesTau[t].find("Mu") != std::string::npos && channel=="mutau"){
	    if(isfiredTau && trignamesTau[t].find("Mu45") == std::string::npos && channel=="mutau" && ID  && lepton.Pt()> 22 &&  leptonIso/lepton.Pt() <0.15 ){///not trigger mu45, unisolated
	      Hist( "Trigger_Tau_Offline",directory )->Fill( t );
	      isfiredTau_Offline = true;
	    }
	    else if (isfiredTau &&lepton.Pt()>50 && ID && channel=="mutau"){
	      Hist( "Trigger_Tau_Offline",directory )->Fill( t );	
	      isfiredTau_Offline = true;
	    }
	  }else if(isfiredTau && trignamesTau[t].find("DoubleMediumIsoPFTau") != std::string::npos && channel=="tautau"){
	    if (isfiredTau &&tau.tlv().Pt()>50 && lepton.Pt() >50 && channel=="tautau" ){
	      Hist( "Trigger_Tau_Offline",directory )->Fill( t );	
	      isfiredTau_Offline = true;
	    }  
	  }
	}
      }  
    }
    Hist( "Trigger_TauOrJet_Offline" ,directory)->Fill( -1 ); 	 
    // if ( (isfiredJet && isfiredJet_Offline) || ( isfiredTau  && isfiredTau_Offline )  )  Hist( "Trigger_TauOrJet_Offline" ,directory)->Fill( 0 ); 	 
    // if (isfiredJet && isfiredJet_Offline )    Hist( "Trigger_TauOrJet_Offline" ,directory)->Fill( 1 ); 
    // if (isfiredTau && isfiredTau_Offline )    Hist( "Trigger_TauOrJet_Offline" ,directory)->Fill( 2 ); 
    // if (isfiredJet && isfiredJet_Offline &&  !(isfiredTau_Offline))    Hist( "Trigger_TauOrJet_Offline",directory )->Fill( 3 ); 
    // if (isfiredTau && !( isfiredJet_Offline ))    Hist( "Trigger_TauOrJet_Offline",directory )->Fill( 4 ); 
    // if (!( isfiredTau_Offline ) && !( isfiredJet_Offline ))    Hist( "Trigger_TauOrJet_Offline",directory )->Fill( 5 ); 
   if ( isfiredJet_Offline ||isfiredTau_Offline   )  Hist( "Trigger_TauOrJet_Offline" ,directory)->Fill( 0 ); 	 
    if ( isfiredJet_Offline )    Hist( "Trigger_TauOrJet_Offline" ,directory)->Fill( 1 ); 
    if (isfiredTau_Offline )    Hist( "Trigger_TauOrJet_Offline" ,directory)->Fill( 2 ); 
    if ( isfiredJet_Offline &&  !(isfiredTau_Offline))    Hist( "Trigger_TauOrJet_Offline",directory )->Fill( 3 ); 
    if (isfiredTau_Offline && !( isfiredJet_Offline ))    Hist( "Trigger_TauOrJet_Offline",directory )->Fill( 4 ); 
    if (!( isfiredTau_Offline ) && !( isfiredJet_Offline ))    Hist( "Trigger_TauOrJet_Offline",directory )->Fill( 5 ); 
    
  }
  
  
  
  trignames.clear();
  trignamesJet.clear();
  trignamesTau.clear();

  std::vector<std::string> trignames_attempts;

  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass20_v1");
  trignames_attempts.push_back("HLT_HT400_MET50_AK8PFJet200_TrimMass20_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass30_v1");
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet200_TrimMass30_v1"); 
  // trignames_attempts.push_back("HLT_HT200_MET50_AK8DiPFJet120_TrimMass20_v1");
  // trignames_attempts.push_back("HLT_HT250_MET50_AK8DiPFJet120_TrimMass20_v1");
  // trignames_attempts.push_back("HLT_HT300_MET50_AK8DiPFJet120_TrimMass20_v1");
  // trignames_attempts.push_back("HLT_HT300_MET50_AK8DiPFJet200_TrimMass20_v1");
  trignames_attempts.push_back("HLT_HT200_MET50_AK8PFJet120_TrimMass20_LooseIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT400_LooseIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT450_LooseIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT450_LooseIsoPFTau30_Trk30_v1");
  trignames_attempts.push_back("HLT_HT200_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT400_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass20_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet200_TrimMass30_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet200_TrimMass30_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT350_MET60_AK8PFJet200_TrimMass20_NotIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet200_TrimMass20_NotIsoPFTau50_v1");
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet120_TrimMass30_NotIsoPFTau50_v1");
  trignames_attempts.push_back("HLT_HT400_MET50_NotIsoPFTau50_Trk30_v1");
  trignames_attempts.push_back("HLT_HT400_MET70_NotIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT450_MET50_NotIsoPFTau50_Trk30_v1"); 

  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet120_TrimMass30_NotIsoPFTau50_Trk30_v1"); 
  trignames_attempts.push_back("HLT_HT300_MET60_AK8PFJet120_TrimMass30_NotIsoPFTau50_Trk30_v1");  
  trignames_attempts.push_back("HLT_HT400_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");  

  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30_v1");   
  trignames_attempts.push_back("HLT_HT300_MET60_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30v1"); 
  trignames_attempts.push_back("HLT_HT350_MET50_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30v1"); 
  trignames_attempts.push_back("HLT_HT300_MET50_AK8PFJet150_TrimMass20_NotIsoPFTau50_Trk30v1"); 





  if  (HT >350 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>30)  Hist( "Trigger_Attempts"  , directory 	)->Fill(0);//"HLT_HT300_MET50_AK8PFJet200_TrimMass20_v1");
  if  (HT >450 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>30)  Hist( "Trigger_Attempts"  , directory 	)->Fill(1);//"HLT_HT400_MET50_AK8PFJet200_TrimMass20_v1");
  if  (HT >350 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>40)  Hist( "Trigger_Attempts"  , directory 	)->Fill(2);//"HLT_HT300_MET50_AK8PFJet200_TrimMass30_v1");
  if  (HT >400 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>40)  Hist( "Trigger_Attempts"  , directory 	)->Fill(3);//"HLT_HT350_MET50_AK8PFJet200_TrimMass30_v1");


  if  (HT >250 && met.et() > 70 && Jet.pt()>180 && Jet.pruned_massCorr()>30 && tau.pt()>70 && tau.byLooseCombinedIsolationDeltaBetaCorr3Hits() >0.5)  Hist( "Trigger_Attempts"  , directory 	)->Fill(4);//"HLT_HT200_MET50_AK8PFJet120_TrimMass20_LooseIsoPFTau50_Trk30_v1");
  if  (HT >450 && met.et() > 70 && tau.pt()>70 && tau.byLooseCombinedIsolationDeltaBetaCorr3Hits() >0.5)  Hist( "Trigger_Attempts"  , directory 	)->Fill(5);//"HLT_HT400_LooseIsoPFtau50_Trk30_v1");
  if  (HT >500 && met.et() > 70 && tau.pt()>70 && tau.byLooseCombinedIsolationDeltaBetaCorr3Hits() >0.5)  Hist( "Trigger_Attempts"  , directory 	)->Fill(6);//"HLT_HT450_LooseIsoPFTau50_Trk30_v1");
  if  (HT >500 && met.et() > 70 && tau.pt()>50 && tau.byLooseCombinedIsolationDeltaBetaCorr3Hits() >0.5)  Hist( "Trigger_Attempts"  , directory 	)->Fill(7);//"HLT_HT450_LooseIsoPFTau30_Trk30_v1");

  if  (HT >250 && met.et() > 70 && Jet.pt()>180 && Jet.pruned_massCorr()>30 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(8);//"HLT_HT200_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  if  (HT >350 && met.et() > 70 && Jet.pt()>180 && Jet.pruned_massCorr()>30 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(9);//"HLT_HT300_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  if  (HT >450 && met.et() > 70 && Jet.pt()>180 && Jet.pruned_massCorr()>30 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(10);//"HLT_HT400_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30_v1");
  if  (HT >350 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>30 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(11);//"HLT_HT300_MET50_AK8PFJet200_TrimMass20_NotIsoPFTau50_Trk30_v1");
  if  (HT >350 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>40 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(12);//"HLT_HT300_MET50_AK8PFJet200_TrimMass30_NotIsoPFTau50_Trk30_v1");
  if  (HT >400 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>40 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(13);//"HLT_HT350_MET50_AK8PFJet200_TrimMass30_NotIsoPFTau50_Trk30_v1");
  if  (HT >400 && met.et() > 80 && Jet.pt()>300 && Jet.pruned_massCorr()>30 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(14);//"HLT_HT350_MET60_AK8PFJet200_TrimMass20_NotIsoPFTau50_Trk30_v1");
 if  (HT >400 && met.et() > 70 && Jet.pt()>300 && Jet.pruned_massCorr()>30 && tau.pt()>65)  Hist( "Trigger_Attempts"  , directory 	)->Fill(15);//"HLT_HT350_MET50_AK8PFJet200_TrimMass20_NotIsoPFTau50_v1");
 if  (HT >350 && met.et() > 70 && Jet.pt()>180 && Jet.pruned_massCorr()>40 && tau.pt()>65)  Hist( "Trigger_Attempts"  , directory 	)->Fill(16);//"HLT_HT300_MET50_AK8PFJet120_TrimMass30_NotIsoPFTau50_v1");

 if  (HT >450 && met.et() > 70  && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(17);//"HLT_HT400_MET50_NotIsoPFTau50_Trk30_v1");
 if  (HT >450 && met.et() > 90  && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(18);//"HLT_HT400_MET70_NotIsoPFTau50_Trk30_v1"); 
 if  (HT >500 && met.et() > 70  && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(19);//"HLT_HT450_MET50_NotIsoPFTau50_Trk30_v1");

 if  (HT >350 && met.et() > 70 && Jet.pt()>180 && Jet.pruned_massCorr()>45 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(20); // HLT_HT300_MET50_AK8PFJet120_TrimMass30_NotIsoPFTau50_Trk30v1 
if  (HT >350 && met.et() > 80 && Jet.pt()>180 && Jet.pruned_massCorr()>45 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(21);// HLT_HT300_MET60_AK8PFJet120_TrimMass30_NotIsoPFTau50_Trk30v1 
if  (HT >450 && met.et() > 70 && Jet.pt()>180 && Jet.pruned_massCorr()>40 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(22);// HLT_HT400_MET50_AK8PFJet120_TrimMass20_NotIsoPFTau50_Trk30v1 

 if  (HT >350 && met.et() > 70 && Jet.pt()>210 && Jet.pruned_massCorr()>45 && tau.pt()>70)  Hist( "Trigger_Attempts"  , directory 	)->Fill(23);// HLT_HT300_MET50_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30v1  
 if  (HT >350 && met.et() > 80 && Jet.pt()>210 && Jet.pruned_massCorr()>45 && tau.pt()>70) Hist( "Trigger_Attempts"  , directory 	)->Fill(24);// HLT_HT300_MET60_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30v1
 if  (HT >400 && met.et() > 70 && Jet.pt()>210 && Jet.pruned_massCorr()>45 && tau.pt()>70) Hist( "Trigger_Attempts"  , directory 	)->Fill(25);// HLT_HT350_MET50_AK8PFJet150_TrimMass30_NotIsoPFTau50_Trk30v1
 if  (HT >350 && met.et() > 70 && Jet.pt()>210 && Jet.pruned_massCorr()>40 && tau.pt()>70) Hist( "Trigger_Attempts"  , directory 	)->Fill(26);// HLT_HT300_MET50_AK8PFJet150_TrimMass20_NotIsoPFTau50_Trk30v1



}

void VHTausAnalysis::FillBranches( const std::string& channel,  const UZH::Jet& Jet, const double& Jet_tau21, const double& Jet_tau31, const double& Jet_tau32, const int& Jet_nTaggedSubjets, const double& Jet_subjet0_csv, const double& Jet_subjet1_csv, const double& ak8jet0_subjet01_dr,  const double& ak8jet0_subjet01_deta, const double& ak8jet0_subjet01_dphi, const double& b_weight, const double& b_weightGen,  const double& b_weightPU, const double& b_weightBtag,  const double& b_weightBtag_veto,  const double& b_runNumber, const double& b_eventNumber,const double& b_lumiBlock,  const UZH::Tau& tau,const  TLorentzVector& lepton, const double& lep2_iso, const UZH::MissingEt& met, int tau2_DM  , float ele_supercluster_eta,  int tau2_boosted ){
   // std::cout<< "Filling branches" <<std::endl;
  b_weight_[channel.c_str()] = b_weight;
  b_weightGen_[channel.c_str()] =  b_weightGen;
  b_weightPU_[channel.c_str()] =  b_weightPU;

  TF1 fa2 =  TF1("fa2","0.983254*ROOT::Math::erf((x-19.1382)/125.506)",20.,40000.);
  

  b_weightMET_[channel.c_str()] =fa2.Eval(met.et())  ;
  if (m_doNormSyst) {
    b_weightPU_up_[channel.c_str()] =  b_weightPU_up;
    b_weightPU_down_[channel.c_str()] =  b_weightPU_down;
  }
  b_weightBtag_[channel.c_str()] =   b_weightBtag;
  b_weightBtag_veto_[channel.c_str()] =   b_weightBtag_veto;

  if (m_doPDFSyst){
    b_weightFacUp_[channel.c_str()] = m_eventInfo.weightFacUp;
    b_weightFacDown_[channel.c_str()] = m_eventInfo.weightFacDown;
    b_weightRenUp_[channel.c_str()] = m_eventInfo.weightRenUp;
    b_weightRenDown_[channel.c_str()] = m_eventInfo.weightRenDown;
    b_weightFacRenUp_[channel.c_str()] = m_eventInfo.weightFacRenUp;
    b_weightFacRenDown_[channel.c_str()] = m_eventInfo.weightFacRenDown;
    b_weightPDF_rms_[channel.c_str()] = m_eventInfo.PDF_rms;
  }
  if (m_doNormSyst) {
    b_weightBtag_bc_up_[channel.c_str()] =   b_weightBtag_bc_up;
    b_weightBtag_veto_bc_up_[channel.c_str()] =   b_weightBtag_veto_bc_up;
    b_weightBtag_bc_down_[channel.c_str()] =   b_weightBtag_bc_down;
    b_weightBtag_veto_bc_down_[channel.c_str()] =   b_weightBtag_veto_bc_down;
    b_weightBtag_l_up_[channel.c_str()] =   b_weightBtag_l_up;
    b_weightBtag_veto_l_up_[channel.c_str()] =   b_weightBtag_veto_l_up;
    b_weightBtag_l_down_[channel.c_str()] =   b_weightBtag_l_down;
    b_weightBtag_veto_l_down_[channel.c_str()] =   b_weightBtag_veto_l_down;

  }


  // std::cout << " b_weightLumi " <<  b_weightLumi <<std::endl;
  b_weightLumi_[channel.c_str()] =   b_weightLumi;
  // std::cout << " b_weightLumi_[channel.c_str()] " <<b_weightLumi_[channel.c_str()] <<std::endl;
 
  b_runNumber_[channel.c_str()] = b_runNumber;
  b_eventNumber_[channel.c_str()] = b_eventNumber;
  b_lumiBlock_[channel.c_str()] = b_lumiBlock;
  
  b_ak8jet0_pt_[channel.c_str()] = Jet.pt();
  b_ak8jet0_phi_[channel.c_str()] = Jet.phi() ;
  b_ak8jet0_eta_[channel.c_str()] = Jet.eta();
  b_ak8jet0_e_[channel.c_str()] = Jet.e() ;
  b_ak8jet0_tau21_[channel.c_str()] = Jet_tau21;
  b_ak8jet0_m_[channel.c_str()] =  Jet.m();
  b_ak8jet0_mpruned_[channel.c_str()] =  Jet.pruned_mass() ;
  b_ak8jet0_mpruned_corr_[channel.c_str()] =  Jet.pruned_massCorr() ;
  b_HT_[channel.c_str()] =  HT ;

  
  UZH::PuppiJet goodPuppiJet; 
  float DR_puppijet_ak8 =0.8; bool FoundPuppiJet=false;
  for ( unsigned int i = 0; i < (*m_jetAK8Puppi.jetAK8_puppi_pt).size(); ++i ) {
    UZH::PuppiJet myjetPuppi( &m_jetAK8Puppi, i );
    
    if (myjetPuppi.jetAK8_puppi_pt() > m_jetPtCut) {
      if (fabs(myjetPuppi.jetAK8_puppi_eta()) < m_jetEtaCut) {
	TLorentzVector puppijet;
	puppijet.SetPtEtaPhiE(myjetPuppi.jetAK8_puppi_pt(),myjetPuppi.jetAK8_puppi_eta(),myjetPuppi.jetAK8_puppi_phi(),myjetPuppi.jetAK8_puppi_e());
	
	if (puppijet.DeltaR(Jet.tlv()) <  DR_puppijet_ak8){
	  DR_puppijet_ak8=puppijet.DeltaR(Jet.tlv());
	  goodPuppiJet=myjetPuppi;
	  FoundPuppiJet=true;
	  // std:: cout << "myjetPuppi.jetAK8_puppi_pt() " << myjetPuppi.jetAK8_puppi_pt() << "ak8 Jet.pt() " <<  Jet.pt() << std::endl;
	}

      }
    }	
  } 


  float puppiCorr, puppi_sd_mass, puppi_sd_tau21;
  if (FoundPuppiJet){
    puppiCorr = getPUPPIweight( goodPuppiJet.jetAK8_puppi_pt() , goodPuppiJet.jetAK8_puppi_eta() );
    // std:: cout << " puppiCorr " << puppiCorr << "  goodPuppiJet.jetAK8_puppi_softdrop_mass()  " << goodPuppiJet.jetAK8_puppi_softdrop_mass()  <<std::endl;
  b_ak8jet0_msoftdrop_[channel.c_str()] =  goodPuppiJet.jetAK8_puppi_softdrop_mass() ;
  b_ak8jet0_msoftdrop_corr_[channel.c_str()] = goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr;
  b_ak8jet0_sd_tau21_[channel.c_str()] = goodPuppiJet.jetAK8_puppi_tau1() ? goodPuppiJet.jetAK8_puppi_tau2()/goodPuppiJet.jetAK8_puppi_tau1() : -99;
  }
  

  if (m_isSignal &&  FoundPuppiJet){ 
    TLorentzVector puppijet_tlv;
    puppijet_tlv.SetPtEtaPhiE(goodPuppiJet.jetAK8_puppi_pt(),goodPuppiJet.jetAK8_puppi_eta(),goodPuppiJet.jetAK8_puppi_phi(),goodPuppiJet.jetAK8_puppi_e());
    b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()] = getPuppi_sd_mass_ScaleAndResolution(true, goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr, puppijet_tlv,Jet.tlv(),Jet.jer_sigma_pt(), 0, 0);
    if (m_doShapeSyst){
      b_ak8jet0_msoftdrop_corr_smear_jmsUp_[channel.c_str()] = getPuppi_sd_mass_ScaleAndResolution(true, goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr, puppijet_tlv,Jet.tlv(),Jet.jer_sigma_pt(), 1, 0);
      b_ak8jet0_msoftdrop_corr_smear_jmsDown_[channel.c_str()] = getPuppi_sd_mass_ScaleAndResolution(true, goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr, puppijet_tlv,Jet.tlv(),Jet.jer_sigma_pt(), -1, 0);
      b_ak8jet0_msoftdrop_corr_smear_jmrUp_[channel.c_str()] = getPuppi_sd_mass_ScaleAndResolution(true, goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr, puppijet_tlv,Jet.tlv(),Jet.jer_sigma_pt(), 0, +1);
      b_ak8jet0_msoftdrop_corr_smear_jmrDown_[channel.c_str()] = getPuppi_sd_mass_ScaleAndResolution(true, goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr, puppijet_tlv,Jet.tlv(),Jet.jer_sigma_pt(), 0, -1);
    }
    // std:: cout << "Mass corr "<< goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr <<" Mass smeared" << getPuppi_sd_mass_ScaleAndResolution(true, goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr, puppijet_tlv,Jet.tlv(),Jet.jer_sigma_pt(), 0, 0);
  } else if (!m_isSignal  &&  FoundPuppiJet){
    b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()]= goodPuppiJet.jetAK8_puppi_softdrop_mass() *puppiCorr;
  }
  if(FoundPuppiJet){ 
    for (int i = 0; i < goodPuppiJet.jetAK8_subjet_puppi_softdrop_N(); ++i) {
      switch(i) {
      case 0:
	b_ak8jet0_puppi_subjet0_csv_[channel.c_str()] = goodPuppiJet.jetAK8_subjet_puppi_softdrop_csv()[i];
	break;
      case 1:
	b_ak8jet0_puppi_subjet1_csv_[channel.c_str()]  = goodPuppiJet.jetAK8_subjet_puppi_softdrop_csv()[i];
	break;
      }
    }
  }



  b_ak8jet0_Hbbtag_[channel.c_str()] = Jet.Hbbtag();
  b_ak8jet0_csv_[channel.c_str()] = Jet.csv();
  b_ak8jet0_subjet01_dr_[channel.c_str()] = ak8jet0_subjet01_dr;
  b_ak8jet0_subjet01_deta_[channel.c_str()] = ak8jet0_subjet01_deta;
  b_ak8jet0_subjet01_dphi_[channel.c_str()] = ak8jet0_subjet01_dphi;
  b_ak8jet0_subjet0_pt_[channel.c_str()] = b_ak8jet0_subjet0_pt;
  b_ak8jet0_subjet1_pt_[channel.c_str()] = b_ak8jet0_subjet1_pt;
  b_ak8jet0_subjet0_csv_[channel.c_str()] = Jet_subjet0_csv;
  b_ak8jet0_subjet1_csv_[channel.c_str()] = Jet_subjet1_csv ;

  b_tau_pt_[channel.c_str()]=tau.tlv().Pt();
  b_tau_eta_[channel.c_str()]=tau.tlv().Eta();
  b_tau_phi_[channel.c_str()]=tau.tlv().Phi();
  b_tau_IsoRaw_[channel.c_str()]=tau.byIsolationMVArun2v1PWnewDMwLTraw();
  b_tau_Iso_[channel.c_str()]=tau.byVLooseIsolationMVArun2v1PWnewDMwLT();
  b_tau_againstEle_[channel.c_str()]=tau.againstElectronVLooseMVA6();
  b_tau_againstMu_[channel.c_str()]=tau.againstMuonLoose3();

  b_seclepton_pt_[channel.c_str()]=lepton.Pt();
  b_seclepton_eta_[channel.c_str()]=lepton.Eta();
  b_seclepton_phi_[channel.c_str()]=lepton.Phi();
  b_seclepton_iso_[channel.c_str()]=lep2_iso;
  
  b_vis_mass_ll_[channel.c_str()]=(lepton +tau.tlv()).M();
  b_dR_ll_[channel.c_str()]= lepton.DeltaR(tau.tlv());
  
  // if (channel== "mutau") {
  //   std:: cout<<" Branch Dr mutau" <<  lepton.DeltaR(tau.tlv()) <<std::endl;
  //   std::cout << " Branch goodMuons[0].tlv().pt() " << lepton.Pt() <<std::endl;
  //   std::cout << " Branch Tau .pt() " << tau.tlv().Pt() <<std::endl;
  // }
  b_met_[channel.c_str()] =met.et();
  b_met_phi_[channel.c_str()] =met.phi();
  b_met_phi_ak4_[channel.c_str()] = met_phi_ak4;
  //std::cout << " exiting branch filling " << tau.tlv().Pt() <<std::endl;
  b_N_AK4_[channel.c_str()] = N_AK4;
  b_N_taggedAK4_L_[channel.c_str()] = taggedAK4_L ;
  b_N_taggedAK4_M_[channel.c_str()] = taggedAK4_M ;
  b_N_taggedAK4_T_[channel.c_str()] = taggedAK4_T ;



 
 

  if (channel=="mutau") { 
    // std::cout << "isBoostedTau_mutau "<<isBoostedTau_mutau  << std::endl;
    b_isBoostedTau_[channel.c_str()] = isBoostedTau_mutau;
    b_ChannelInt_[channel.c_str()] = 0; 
    if (!m_isData) {
      
      b_weightTrigSemi_[channel.c_str()]= m_ScaleFactorTool.get_EfficiencyTrigMu(lepton.Pt(),fabs(lepton.Eta()));
      b_weightLepID_[channel.c_str()]= m_ScaleFactorTool.get_ScaleFactor_IDMuIchep(lepton.Pt(),fabs(lepton.Eta()));
      b_weightLepIso_[channel.c_str()]= m_ScaleFactorTool.get_ScaleFactor_IsoMuIchep(lepton.Pt(),fabs(lepton.Eta()));
      if (m_doNormSyst) {
	b_weightTrigSemi_up_[channel.c_str()]= b_weightTrigSemi_[channel.c_str()] + m_ScaleFactorTool.get_EfficiencyTrigMu_unc(lepton.Pt(),fabs(lepton.Eta())); 
	b_weightTrigSemi_down_[channel.c_str()]= b_weightTrigSemi_[channel.c_str()] - m_ScaleFactorTool.get_EfficiencyTrigMu_unc(lepton.Pt(),fabs(lepton.Eta()));
	b_weightLepID_up_[channel.c_str()]= b_weightLepID_[channel.c_str()] + m_ScaleFactorTool.get_ScaleFactor_IDMuIchep_unc(lepton.Pt(),fabs(lepton.Eta()));
	b_weightLepIso_up_[channel.c_str()]= b_weightLepIso_[channel.c_str()] +  m_ScaleFactorTool.get_ScaleFactor_IsoMuIchep_unc(lepton.Pt(),fabs(lepton.Eta())); 
	b_weightLepID_down_[channel.c_str()]= b_weightLepID_[channel.c_str()] - m_ScaleFactorTool.get_ScaleFactor_IDMuIchep_unc(lepton.Pt(),fabs(lepton.Eta()));
	b_weightLepIso_down_[channel.c_str()]= b_weightLepIso_[channel.c_str()] -  m_ScaleFactorTool.get_ScaleFactor_IsoMuIchep_unc(lepton.Pt(),fabs(lepton.Eta())); 
      }
    }
  }
  if (channel=="eletau")  {   
    //    std::cout << " ele SF " <<std::endl;
    b_isBoostedTau_[channel.c_str()] = isBoostedTau_eletau;
    b_ChannelInt_[channel.c_str()] = 1;
    
    if (!m_isData) {  
      b_weightTrigSemi_[channel.c_str()]= 1;  
     
      b_weightLepID_[channel.c_str()]= m_ScaleFactorTool.get_ScaleFactor_IDEleIchep(lepton.Pt(), ele_supercluster_eta) * m_ScaleFactorTool.get_ScaleFactor_RecoEleIchep(lepton.Pt(),ele_supercluster_eta);
      if (m_doNormSyst) {

	float SF=  m_ScaleFactorTool.get_ScaleFactor_IDEleIchep(lepton.Pt(), ele_supercluster_eta) * m_ScaleFactorTool.get_ScaleFactor_RecoEleIchep(lepton.Pt(),ele_supercluster_eta);

	// std::cout <<" reco ele SF"<< SF <<std::endl;
	float	SF_unc=0;
	SF_unc= pow (   pow(
			    m_ScaleFactorTool.get_ScaleFactor_IDEleIchep_unc(lepton.Pt(), ele_supercluster_eta)/m_ScaleFactorTool.get_ScaleFactor_IDEleIchep(lepton.Pt(), ele_supercluster_eta) ,2) 
			+  pow(m_ScaleFactorTool.get_ScaleFactor_RecoEleIchep_unc(lepton.Pt(), ele_supercluster_eta)/m_ScaleFactorTool.get_ScaleFactor_RecoEleIchep(lepton.Pt(), ele_supercluster_eta) ,2)
			,0.5 )* SF;


	// std::cout <<" reco ele SF unc"<<SF_unc << " SFid  "<< ( m_ScaleFactorTool.get_ScaleFactor_IDEleIchep_unc(lepton.Pt(), ele_supercluster_eta)/m_ScaleFactorTool.get_ScaleFactor_IDEleIchep(lepton.Pt(), ele_supercluster_eta))<< " SF reco " <<( m_ScaleFactorTool.get_ScaleFactor_RecoEleIchep_unc(lepton.Pt(), ele_supercluster_eta)/m_ScaleFactorTool.get_ScaleFactor_RecoEleIchep(lepton.Pt(), ele_supercluster_eta))<<  std::endl;

	b_weightLepID_up_[channel.c_str()]= SF+SF_unc;
	b_weightLepID_down_[channel.c_str()]= SF-SF_unc;
      }
    }
  }
  if (channel=="tautau")  {
    b_isBoostedTau_[channel.c_str()] = isBoostedTau_eletau; //FIX
    b_ChannelInt_[channel.c_str()] = 2;
    b_weightTrigSemi_[channel.c_str()]= 1;
    }

  
  //b_weightTot_[channel.c_str()] =  b_weightGen_[channel.c_str()] *   b_weightLumi_[channel.c_str()] *  b_weightPU_[channel.c_str()] *     b_weightBtag_[channel.c_str()] *    b_weightBtag_veto_[channel.c_str()] * 
  //b_weightTrigSemi_[channel.c_str()] * b_weightLepID_[channel.c_str()] *   b_weightLepIso_[channel.c_str()] ;
  
  // std::cout <<"  b_weightTot_[channel.c_str()] " << b_weightTot_[channel.c_str()]<<   " =  b_weightGen_[channel.c_str()] "<< b_weightGen_[channel.c_str()]<<"   b_weightLumi_[channel.c_str()]"<<   b_weightLumi_[channel.c_str()] <<
  //   " b_weightPU_[channel.c_str()] " <<  b_weightPU_[channel.c_str()] << " b_weightBtag_[channel.c_str()] "<<  b_weightBtag_[channel.c_str()] <<
  //   " b_weightBtag_veto_[channel.c_str()] " <<   b_weightBtag_veto_[channel.c_str()] <<" b_weightTrigSemi_[channel.c_str()] " <<b_weightTrigSemi_[channel.c_str()] <<
  //   "  b_weightLepID_[channel.c_str()]  " <<  b_weightLepID_[channel.c_str()] << "   b_weightLepIso_[channel.c_str()]  "<<    b_weightLepIso_[channel.c_str()]  <<std::endl;
  

  
  // b_weightTot_woBtags_[channel.c_str()] = b_weightTot_[channel.c_str()]/(  b_weightBtag_[channel.c_str()] *    b_weightBtag_veto_[channel.c_str()] )  ;
  // b_weightTot_woBtagVeto_[channel.c_str()] =  b_weightTot_[channel.c_str()]/( b_weightBtag_veto_[channel.c_str()] ) ;
  // b_weightTot_woBtagSJ_[channel.c_str()] = b_weightTot_[channel.c_str()]/(  b_weightBtag_[channel.c_str()] )  ;


  
  b_PV_N_[channel.c_str()]= m_eventInfo.PV_N;
  b_N_tight_muons_[channel.c_str()]=N_tight_muons;
  b_N_tight_electrons_[channel.c_str()]=N_tight_electrons;
  b_N_loose_muons_[channel.c_str()]=N_loose_muons;
  b_N_loose_electrons_[channel.c_str()]=N_loose_electrons;
    
  passTrigger_branches(channel.c_str());
  if (channel == "mutau"){
    b_N_tight_leptons_[channel.c_str()] = b_N_tight_muons_[channel.c_str()];
  }
  else if (channel == "eletau"){
    b_N_tight_leptons_[channel.c_str()] = b_N_tight_electrons_[channel.c_str()];
  }
  else b_N_tight_leptons_[channel.c_str()] = b_N_tight_muons_[channel.c_str()] + b_N_tight_electrons_[channel.c_str()]; 

  TLorentzVector H_ll;
  bool doSVfit= true ;// !m_doShapeSyst; //
  
  if (doSVfit){
    
    H_ll.SetPtEtaPhiM(0,0,0,0);
    
    if (channel=="tautau"){
      // std::cout <<"doing SVfit "  <<std::endl;
      
      H_ll=applySVFitHadronic(met.cov00(),met.cov10(),met.cov11(),met.et(),met.phi(),tau.tlv(),lepton);
      // std::cout <<"done SVfit "  <<std::endl;
      
      
    }
    else  {
      // if (channel=="mutau"){ std::cout <<"doing SVfit for muon"  <<std::endl;}
      // if (channel=="eletau"){ std::cout <<"doing SVfit for electron"  <<std::endl;}
      
      H_ll=applySVFitSemileptonic (met.cov00(),met.cov10(),met.cov11(),met.et(),met.phi(),tau.tlv(),lepton);
    }
    
    b_H_Mass_SVFit_[channel.c_str()]=H_ll.M();
    b_H_Pt_SVFit_[channel.c_str()]=H_ll.Pt();
    b_H_Eta_SVFit_[channel.c_str()]=H_ll.Eta();
    b_H_Phi_SVFit_[channel.c_str()]=H_ll.Phi();
    b_X_Mass_SVFit_[channel.c_str()]=(H_ll+Jet.tlv()).M();
    b_X_Pt_SVFit_[channel.c_str()]=(H_ll+Jet.tlv()).Pt();
    b_X_Eta_SVFit_[channel.c_str()]=(H_ll+Jet.tlv()).Eta();
    b_X_Phi_SVFit_[channel.c_str()]=(H_ll+Jet.tlv()).Phi();

   
    
    if (m_doSVFit_new){
      TLorentzVector H_ll_new;
      H_ll_new.SetPtEtaPhiM(0,0,0,0);
    
      if (channel=="tautau"){
	// std::cout <<"doing SVfit "  <<std::endl;
      
	H_ll_new=applySVFitStandaloneHadronic(met.cov00(),met.cov10(),met.cov11(),met.et(),met.phi(),tau,lepton, tau2_DM);
	// std::cout <<"done SVfit "  <<std::endl;
      
      
      }
      else if (channel=="mutau") {
	H_ll_new=applySVFitStandaloneSemileptonicMu (met.cov00(),met.cov10(),met.cov11(),met.et(),met.phi(),tau,lepton);
      }
      else if (channel=="eletau"){
	H_ll_new=applySVFitStandaloneSemileptonicEle(met.cov00(),met.cov10(),met.cov11(),met.et(),met.phi(),tau,lepton);
      
      }
      b_H_Mass_SVFit_new_[channel.c_str()]=H_ll_new.M();
      b_H_Pt_SVFit_new_[channel.c_str()]=H_ll_new.Pt();
      b_H_Eta_SVFit_new_[channel.c_str()]=H_ll_new.Eta();
      b_H_Phi_SVFit_new_[channel.c_str()]=H_ll_new.Phi();
      b_X_Mass_SVFit_new_[channel.c_str()]=(H_ll_new+Jet.tlv()).M();
      b_X_Pt_SVFit_new_[channel.c_str()]=(H_ll_new+Jet.tlv()).Pt();
      b_X_Eta_SVFit_new_[channel.c_str()]=(H_ll_new+Jet.tlv()).Eta();
      b_X_Phi_SVFit_new_[channel.c_str()]=(H_ll_new+Jet.tlv()).Phi();

     

      // std::cout <<"written  SVfit "  <<std::endl;

    }
  }

  TLorentzVector H_ll_Collinear;
  TLorentzVector met_tlv;
  H_ll_Collinear.SetPtEtaPhiM(0,0,0,0);
  met_tlv.SetPtEtaPhiM(met.et(),0,met.phi(),0);
  
  H_ll_Collinear=doCollinearApprox(met_tlv,tau.tlv(),lepton );
  b_H_Mass_Collinear_[channel.c_str()]=H_ll_Collinear.M();
  b_H_Pt_Collinear_[channel.c_str()]=H_ll_Collinear.Pt();
  b_H_Eta_Collinear_[channel.c_str()]=H_ll_Collinear.Eta();
  b_H_Phi_Collinear_[channel.c_str()]=H_ll_Collinear.Phi();
  b_X_Mass_Collinear_[channel.c_str()]=(H_ll_Collinear+Jet.tlv()).M();
  b_X_Pt_Collinear_[channel.c_str()]=(H_ll_Collinear+Jet.tlv()).Pt();
  b_X_Eta_Collinear_[channel.c_str()]=(H_ll_Collinear+Jet.tlv()).Eta();
  b_X_Phi_Collinear_[channel.c_str()]=(H_ll_Collinear+Jet.tlv()).Phi();

  if (m_doShapeSyst && !m_isData){
    TLorentzVector H_ll_Coll_Syst; 
    H_ll_Coll_Syst.SetPtEtaPhiM(0,0,0,0);

    H_ll_Coll_Syst=doCollinearApprox(goodMet_jesUp,tau.tlv(),lepton );
    b_H_Mass_Collinear_jesUp_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_jesUp_[channel.c_str()]=(H_ll_Coll_Syst+ goodJetsAK8_jesUp).M();
    b_met_jesUp_[channel.c_str()] = goodMet_jesUp.Et();
    b_ak8jet0_pt_jesUp_[channel.c_str()] = goodJetsAK8_jesUp.Pt();

    H_ll_Coll_Syst=doCollinearApprox(goodMet_jesDown,tau.tlv(),lepton );
    b_H_Mass_Collinear_jesDown_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_jesDown_[channel.c_str()]=(H_ll_Coll_Syst+ goodJetsAK8_jesDown).M();
    b_met_jesDown_[channel.c_str()] = goodMet_jesDown.Et();
    b_ak8jet0_pt_jesDown_[channel.c_str()] = goodJetsAK8_jesDown.Pt();


    H_ll_Coll_Syst=doCollinearApprox(goodMet_jerUp,tau.tlv(),lepton );
    b_H_Mass_Collinear_jerUp_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_jerUp_[channel.c_str()]=(H_ll_Coll_Syst+ goodJetsAK8_jerUp).M();
    b_met_jerUp_[channel.c_str()] = goodMet_jerUp.Et();
    b_ak8jet0_pt_jerUp_[channel.c_str()] = goodJetsAK8_jerUp.Pt();

    H_ll_Coll_Syst=doCollinearApprox(goodMet_jerDown,tau.tlv(),lepton );
    b_H_Mass_Collinear_jerDown_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_jerDown_[channel.c_str()]=(H_ll_Coll_Syst+ goodJetsAK8_jerDown).M();
    b_met_jerDown_[channel.c_str()] = goodMet_jerDown.Et();
    b_ak8jet0_pt_jerDown_[channel.c_str()] = goodJetsAK8_jerDown.Pt();

    H_ll_Coll_Syst=doCollinearApprox(goodMet_jer,tau.tlv(),lepton );
    b_H_Mass_Collinear_jer_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_jer_[channel.c_str()]=(H_ll_Coll_Syst+ goodJetsAK8_jer).M();
    b_met_jer_[channel.c_str()] = goodMet_jer.Et();
    b_ak8jet0_pt_jer_[channel.c_str()] = goodJetsAK8_jer.Pt();

    H_ll_Coll_Syst=doCollinearApprox(goodMet_UncEnUp,tau.tlv(),lepton );
    b_H_Mass_Collinear_UncEnUp_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_UncEnUp_[channel.c_str()]=(H_ll_Coll_Syst+ Jet.tlv()).M();
    b_met_UncEnUp_[channel.c_str()] = goodMet_UncEnUp.Et();
  

    H_ll_Coll_Syst=doCollinearApprox(goodMet_UncEnDown,tau.tlv(),lepton );
    b_H_Mass_Collinear_UncEnDown_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_UncEnDown_[channel.c_str()]=(H_ll_Coll_Syst+ Jet.tlv()).M();
    b_met_UncEnDown_[channel.c_str()] = goodMet_UncEnDown.Et();
   
    
    TLorentzVector tau_syst,lep_syst,met_syst,met_tlv;
    met_tlv.SetPtEtaPhiM(met.et(),  0, met.phi(), 0 );
    tau_syst.SetPtEtaPhiM(tau.pt()*1.03,  tau.eta(), tau.phi(), tau.m());
    if( channel=="tautau" )lep_syst.SetPtEtaPhiM(lepton.Pt()*1.03,  lepton.Eta(), lepton.Phi(), lepton.M());
    else lep_syst.SetPtEtaPhiM(lepton.Pt(),  lepton.Eta(), lepton.Phi(), lepton.M());
  
    met_syst.SetPtEtaPhiM( (met_tlv + (tau.tlv() - tau_syst) + lepton -lep_syst).Pt(),  0, (met_tlv + (tau.tlv() - tau_syst) + lepton -lep_syst).Phi(), 0 );
    
    H_ll_Coll_Syst=doCollinearApprox(met_syst,tau_syst,lep_syst );
    b_H_Mass_Collinear_tesUp_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_tesUp_[channel.c_str()]=(H_ll_Coll_Syst+ Jet.tlv()).M();
    b_met_tesUp_[channel.c_str()] = met_syst.Et();
   
    met_tlv.SetPtEtaPhiM(met.et(),  0, met.phi(), 0 );
    tau_syst.SetPtEtaPhiM(tau.pt()*0.97,  tau.eta(), tau.phi(), tau.m());
    if( channel=="tautau" )lep_syst.SetPtEtaPhiM(lepton.Pt()*0.97,  lepton.Eta(), lepton.Phi(), lepton.M());
    else lep_syst.SetPtEtaPhiM(lepton.Pt(),  lepton.Eta(), lepton.Phi(), lepton.M());
  
    met_syst.SetPtEtaPhiM( (met_tlv + (tau.tlv() - tau_syst) + lepton -lep_syst).Pt(),  0, (met_tlv + (tau.tlv() - tau_syst) + lepton -lep_syst).Phi(), 0 );
    
    H_ll_Coll_Syst=doCollinearApprox(met_syst,tau_syst,lep_syst );
    b_H_Mass_Collinear_tesDown_[channel.c_str()]=H_ll_Coll_Syst.M();
    b_X_Mass_Collinear_tesDown_[channel.c_str()]=(H_ll_Coll_Syst+ Jet.tlv()).M();
    b_met_tesDown_[channel.c_str()] = met_syst.Et();
   
  }


  if (!m_isData){
  
    b_GenEvent_Htata_[channel.c_str()] = GenEvent_Htata_filter ;
    b_GenEvent_Ztata_[channel.c_str()] = GenEvent_Ztata_filter ;
    b_GenEvent_Gammatata_[channel.c_str()] = GenEvent_Gammatata_filter ;
   }

  float tauSF=1, tauSF_unc=0, tauSF1=1, tauSF1_unc,tauSF2=1, tauSF2_unc=0;
  if  (tau.TauType()==2 ){tauSF1=0.98;tauSF1_unc=0.05;}
  else{tauSF1=0.97;tauSF1_unc=0.05;} 
  if  (tau2_boosted==2 ){tauSF2=0.98; tauSF2_unc=  0.05;}
  else if  (tau2_boosted==1 ) {tauSF2=0.97;tauSF2_unc=0.05;} 

  if (tau2_boosted!=0) {tauSF = tauSF1*tauSF2; tauSF_unc = tauSF* pow((tauSF1_unc/tauSF1)*(tauSF1_unc/tauSF1) + (tauSF2_unc/tauSF2)*(tauSF2_unc/tauSF2),0.5);}
  else {tauSF = tauSF1; tauSF_unc =tauSF1_unc;}


  if (GenEvent_Htata_filter==1 || GenEvent_Ztata_filter ==1 || GenEvent_Gammatata_filter ==1){
    b_weightTauID_[channel.c_str()]=tauSF;
  }
  else{  b_weightTauID_[channel.c_str()]=1;}

  TString File_at_the_moment_sig = TString(this->GetHistInputFile()->GetName());
  if (!m_isData && channel!="tautau" && m_doGenRecoMatchVariables && (File_at_the_moment_sig.Contains("Radion") || File_at_the_moment_sig.Contains("Wprime") || File_at_the_moment_sig.Contains("Zprime")|| File_at_the_moment_sig.Contains("BulkGrav")  )){
    
    PairingRecoGenLevel( Jet, lepton , tau,  met , H_ll_Collinear,  (H_ll_Collinear+Jet.tlv()) ,channel ) ;
    
  }


  if (!m_isData){
    float ZewkWeight, WewkWeight, TopWeight ; 
    ZewkWeight=1; WewkWeight =1; TopWeight =1; 
    TLorentzVector GenV;
    TLorentzVector LepP;
    TLorentzVector LepM;
    float GenVPt(-1.), GenTopPt(-1.), GenAntiTopPt(-1.), TopPtWeight(1.), AntiTopPtWeight(1.);
    // Z and W electroweak corrections
    std::string sample(this->GetHistInputFile()->GetName());
    if(!m_isData && (sample.find("DYJetsToLL") != std::string::npos || sample.find("ZJetsToNuNu") != std::string::npos || sample.find("WJetsToLNu") != std::string::npos)) {
      for(int i = 0; i < m_genParticle.N; ++i) {
	UZH::GenParticle gen(&m_genParticle, i);
	if(GenV.Pt()<=0. && (gen.pdgId() == 23 || gen.pdgId() == 24) && gen.nDau()>0. && fabs(gen.dau()[0])!=gen.pdgId()) GenV = gen.tlv();
	if(LepP.Pt()<=0. && ((gen.pdgId() >= +11 && gen.pdgId() <= +16 && gen.status()==1) || (gen.pdgId() == +15 && gen.status()==2)) ) LepP = gen.tlv();
	if(LepM.Pt()<=0. && ((gen.pdgId() >= -16 && gen.pdgId() <= -11 && gen.status()==1) || (gen.pdgId() == -15 && gen.status()==2)) ) LepM = gen.tlv();
      }
      if(GenV.Pt()>0.) GenVPt = GenV.Pt();
      else if(LepP.Pt()>0. && LepM.Pt()>0.) GenVPt = (LepP+LepM).Pt();
        
      if(GenVPt > 0. && (sample.find("DYJetsToLL") != std::string::npos || sample.find("ZJetsToNuNu") != std::string::npos)) ZewkWeight *= m_ScaleFactorTool.GetEwkZ(GenVPt);
      if(GenVPt > 0. && (sample.find("WJetsToLNu") != std::string::npos))  WewkWeight *= m_ScaleFactorTool.GetEwkW(GenVPt);
      b_weightGen_corr *= WewkWeight * ZewkWeight;
    }
    // TTbar pT reweighting
    if(!m_isData && (sample.find("TT") != std::string::npos)&& (sample.find("powheg") != std::string::npos)) {
      if(m_eventInfo.lheNj < 6) TopWeight *= 0.5;
      for(int i = 0; i < m_genParticle.N; ++i) {
	UZH::GenParticle gen(&m_genParticle, i);
	if(gen.pdgId() == +6) GenTopPt = gen.pt();
	if(gen.pdgId() == -6) GenAntiTopPt = gen.pt();
	//if(gen.status()==23 || gen.status()==33) std::cout << gen.pdgId() << "   " << gen.status() << std::endl;
      }
      if(GenTopPt>0.) TopPtWeight = exp(0.0615 - 0.0005*GenTopPt);
      if(GenAntiTopPt>0.) AntiTopPtWeight = exp(0.0615 - 0.0005*GenAntiTopPt);
      TopWeight *= sqrt(TopPtWeight * AntiTopPtWeight);
    
      b_weightGen_corr *= TopWeight;
    }







    b_weightGen_corr_[channel.c_str()] = b_weightGen_corr;



    if( m_doNormSyst && m_doHistNotTree){


     

      if (  b_ak8jet0_pt_[channel.c_str()] >200  &&  b_met_[channel.c_str()] >200  && b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()]>65 && b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()]<135 &&


	    (( b_seclepton_pt_[channel.c_str()] >20  && channel =="tautau" ) || b_seclepton_pt_[channel.c_str()] >10))
	{


	  Hist( "PDF_rms", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(0. ,  1. ) ;
	  Hist( "PDF_rms" , channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(1 ,   m_eventInfo.PDF_rms     );
  
	  Hist( "weightFac", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(0. ,  1. ) ;
	  Hist( "weightFac", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(1. , m_eventInfo.weightFacUp ) ;
	  Hist( "weightFac", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(2. , m_eventInfo.weightFacDown ) ;

	  Hist( "weightRen", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(0. ,  1. ) ;
	  Hist( "weightRen", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(1. , m_eventInfo.weightRenUp ) ;
	  Hist( "weightRen", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(2. , m_eventInfo.weightRenDown ) ;

	  Hist( "weightFacRen", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(0. ,  1. ) ;
	  Hist( "weightFacRen", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(1. , m_eventInfo.weightFacRenUp ) ;
	  Hist( "weightFacRen", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag"  )->Fill(2. , m_eventInfo.weightFacRenDown ) ;

	  Hist( "X_Mass_Collinear_PDF_rms", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ,  m_eventInfo.PDF_rms  ); 
	  Hist( "X_Mass_Collinear_weightFacUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ,  m_eventInfo.weightFacUp  ); 
	  Hist( "X_Mass_Collinear_weightFacDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ,  m_eventInfo.weightFacDown  ); 
	  Hist( "X_Mass_Collinear_weightRenUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ,  m_eventInfo.weightRenUp  ); 
	  Hist( "X_Mass_Collinear_weightRenDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ,  m_eventInfo.weightRenDown  ); 
	  Hist( "X_Mass_Collinear_weightFacRenUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ,  m_eventInfo.weightFacRenUp  ); 
	  Hist( "X_Mass_Collinear_weightFacRenDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ,  m_eventInfo.weightFacRenDown  ); 
 

	  Hist( "X_Mass_Collinear", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ); 

	  Hist( "H_Mass_Collinear", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_[channel.c_str()] );
	  Hist( "ak8jet0_msoftdrop_corr_smear", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()] ); 
	
	 
	  //if(  lepton.DeltaR(tau.tlv())<=0.4){tauSF=0.94;tauSF_unc=0.15;}
	  //else{tauSF=1.01;tauSF_unc=0.08;}
	 

	  Hist( "TauSF_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 0. , tauSF);
	  Hist( "TauSF_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 1 , tauSF+tauSF_unc);
	  Hist( "TauSF_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 2 , tauSF-tauSF_unc);
	 
	  //b-tag medium SF syst cumulative : For signal, not for Top CR
	  Hist( "Btag_BC_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 0. , b_weightBtag_[channel.c_str()]*  b_weightBtag_veto_[channel.c_str()] );
	  Hist( "Btag_BC_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 1 ,  b_weightBtag_bc_up_[channel.c_str()]*  b_weightBtag_veto_bc_up_[channel.c_str()]);
	  Hist( "Btag_BC_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 2 , b_weightBtag_bc_down_[channel.c_str()]*  b_weightBtag_veto_bc_down_[channel.c_str()]);

	  
	  Hist( "Btag_LF_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 0. , b_weightBtag_[channel.c_str()]*  b_weightBtag_veto_[channel.c_str()] );
	  Hist( "Btag_LF_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 1 ,  b_weightBtag_l_up_[channel.c_str()]*  b_weightBtag_veto_l_up_[channel.c_str()]);
	  Hist( "Btag_LF_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 2 , b_weightBtag_l_down_[channel.c_str()]*  b_weightBtag_veto_l_down_[channel.c_str()]);
	 

	  Hist( "LepID_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 0. , b_weightLepID_[channel.c_str()]*b_weightLepIso_[channel.c_str()]);
	  Hist( "LepID_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 1 , b_weightLepID_up_[channel.c_str()]*b_weightLepIso_up_[channel.c_str()]);
	  Hist( "LepID_syst", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill( 2 , b_weightLepIso_down_[channel.c_str()]*b_weightLepIso_down_[channel.c_str()]);
	  
	  
	  ///tausyst

	  if  (b_met_jesUp_[channel.c_str()] >200 && b_ak8jet0_pt_jesUp_[channel.c_str()] >200  )  {
	    Hist( "X_Mass_Collinear_jesUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jesUp_[channel.c_str()] );
	    Hist( "H_Mass_Collinear_jesUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jesUp_[channel.c_str()] );
	  }
	  
	  if  (b_met_jesDown_[channel.c_str()] >200 && b_ak8jet0_pt_jesDown_[channel.c_str()] >200 ) {
	    Hist( "X_Mass_Collinear_jesDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jesDown_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_jesDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jesDown_[channel.c_str()] ); 
	  }
	  if  (b_met_jerUp_[channel.c_str()] >200 && b_ak8jet0_pt_jerUp_[channel.c_str()] >200 ) {
	    Hist( "X_Mass_Collinear_jerUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jerUp_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_jerUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jerUp_[channel.c_str()] ); 
	  }
	  if  (b_met_jerDown_[channel.c_str()] >200 && b_ak8jet0_pt_jerDown_[channel.c_str()] >200 ) {
	    Hist( "X_Mass_Collinear_jerDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jerDown_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_jerDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jerDown_[channel.c_str()] ); 
	  }
	  if  (b_met_jer_[channel.c_str()] >200 && b_ak8jet0_pt_jer_[channel.c_str()] >200 ) {
	    Hist( "X_Mass_Collinear_jer", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jer_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_jer", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jer_[channel.c_str()] ); 
	  }
	  if  (b_met_UncEnUp_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	    Hist( "X_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	  }
	  if  (b_met_UncEnDown_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	    Hist( "X_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_UncEnDown_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_UncEnDown_[channel.c_str()] ); 

	  }
	  if  (b_met_tesUp_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*1.03 >20 ) {
	    Hist( "X_Mass_Collinear_tesUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_tesUp_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_tesUp", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_tesUp_[channel.c_str()] ); 
	  }
	  if  (b_met_tesDown_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*0.97 >20 ) {
	    Hist( "X_Mass_Collinear_tesDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_tesDown_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear_tesDown", channel.c_str()+(TString)"_NoWindow_NoTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_tesDown_[channel.c_str()] ); 
	  }
	  if (b_ak8jet0_sd_tau21_[channel.c_str()]<0.4){
	   
	    Hist( "X_Mass_Collinear", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()] ); 
	      
	      
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill( 0. ,1);
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill( 1 , 1 + 0.085 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill( 2 , 1 - 0.085 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	      
	  

	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsDown_[channel.c_str()] ); 
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrDown_[channel.c_str()] );
	  
	    if  (b_met_jesUp_[channel.c_str()] >200 && b_ak8jet0_pt_jesUp_[channel.c_str()] >200  )  {
	      Hist( "X_Mass_Collinear_jesUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jesUp_[channel.c_str()] );
	      Hist( "H_Mass_Collinear_jesUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jesUp_[channel.c_str()] );
	    }
	  
	    if  (b_met_jesDown_[channel.c_str()] >200 && b_ak8jet0_pt_jesDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jesDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jesDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jesDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jerUp_[channel.c_str()] >200 && b_ak8jet0_pt_jerUp_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jerUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jerUp_[channel.c_str()] ); 
	    }
	    if  (b_met_jerDown_[channel.c_str()] >200 && b_ak8jet0_pt_jerDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jerDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jerDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jer_[channel.c_str()] >200 && b_ak8jet0_pt_jer_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jer", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jer_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jer", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jer_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnUp_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnDown_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_UncEnDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_UncEnDown_[channel.c_str()] ); 

	    }
	    if  (b_met_tesUp_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*1.03 >20 ) {
	      Hist( "X_Mass_Collinear_tesUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_tesUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesUp", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_tesUp_[channel.c_str()] ); 
	    }
	    if  (b_met_tesDown_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*0.97 >20 ) {
	      Hist( "X_Mass_Collinear_tesDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_tesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesDown", channel.c_str()+(TString)"_VWindow_HPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_tesDown_[channel.c_str()] ); 
	    }

	  }
	  if (b_ak8jet0_sd_tau21_[channel.c_str()]>0.4 && b_ak8jet0_sd_tau21_[channel.c_str()]<0.75){
	   
	    Hist( "X_Mass_Collinear", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()] ); 
	      
	      
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill( 0. ,1);
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill( 1 , 1 + 0.039 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill( 2 , 1 - 0.039 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	      
	  

	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsDown_[channel.c_str()] ); 
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrDown_[channel.c_str()] );
	  
	    if  (b_met_jesUp_[channel.c_str()] >200 && b_ak8jet0_pt_jesUp_[channel.c_str()] >200  )  {
	      Hist( "X_Mass_Collinear_jesUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jesUp_[channel.c_str()] );
	      Hist( "H_Mass_Collinear_jesUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jesUp_[channel.c_str()] );
	    }
	  
	    if  (b_met_jesDown_[channel.c_str()] >200 && b_ak8jet0_pt_jesDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jesDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jesDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jesDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jerUp_[channel.c_str()] >200 && b_ak8jet0_pt_jerUp_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jerUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jerUp_[channel.c_str()] ); 
	    }
	    if  (b_met_jerDown_[channel.c_str()] >200 && b_ak8jet0_pt_jerDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jerDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jerDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jer_[channel.c_str()] >200 && b_ak8jet0_pt_jer_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jer", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_jer_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jer", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_jer_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnUp_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnDown_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_UncEnDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_UncEnDown_[channel.c_str()] ); 

	    }
	    if  (b_met_tesUp_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*1.03 >20 ) {
	      Hist( "X_Mass_Collinear_tesUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_tesUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesUp", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_tesUp_[channel.c_str()] ); 
	    }
	    if  (b_met_tesDown_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*0.97 >20 ) {
	      Hist( "X_Mass_Collinear_tesDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_X_Mass_Collinear_tesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesDown", channel.c_str()+(TString)"_VWindow_LPTau21_SubjetPreTag" )->Fill(b_H_Mass_Collinear_tesDown_[channel.c_str()] ); 
	    }

	  }

	  if ( b_ak8jet0_subjet0_csv_[channel.c_str()]>0.5426 && b_ak8jet0_subjet1_csv_[channel.c_str()]>0.5426){


	    Hist( "X_Mass_Collinear", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()] ); 
	      
	      
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill( 0. ,1);
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill( 1 , 1 + 0.039 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill( 2 , 1 - 0.039 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	      
	  

	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsDown_[channel.c_str()] ); 
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrDown_[channel.c_str()] ); 
	    if  (b_met_jesUp_[channel.c_str()] >200 && b_ak8jet0_pt_jesUp_[channel.c_str()] >200  )  {
	      Hist( "X_Mass_Collinear_jesUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_jesUp_[channel.c_str()] );
	      Hist( "H_Mass_Collinear_jesUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_jesUp_[channel.c_str()] );
	    }
	  
	    if  (b_met_jesDown_[channel.c_str()] >200 && b_ak8jet0_pt_jesDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jesDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_jesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jesDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_jesDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jerUp_[channel.c_str()] >200 && b_ak8jet0_pt_jerUp_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_jerUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_jerUp_[channel.c_str()] ); 
	    }
	    if  (b_met_jerDown_[channel.c_str()] >200 && b_ak8jet0_pt_jerDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_jerDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_jerDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jer_[channel.c_str()] >200 && b_ak8jet0_pt_jer_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jer", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_jer_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jer", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_jer_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnUp_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnDown_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_UncEnDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_UncEnDown_[channel.c_str()] ); 

	    }
	    if  (b_met_tesUp_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*1.03 >20 ) {
	      Hist( "X_Mass_Collinear_tesUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_tesUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesUp", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_tesUp_[channel.c_str()] ); 
	    }
	    if  (b_met_tesDown_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*0.97 >20 ) {
	      Hist( "X_Mass_Collinear_tesDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_X_Mass_Collinear_tesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesDown", channel.c_str()+(TString)"_HWindow_NoTau21_2SubjetTag" )->Fill(b_H_Mass_Collinear_tesDown_[channel.c_str()] ); 
	    }
	  }
	  else if ( b_ak8jet0_subjet0_csv_[channel.c_str()]>0.5426 || b_ak8jet0_subjet1_csv_[channel.c_str()]>0.5426){
	   


	    Hist( "X_Mass_Collinear", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_[channel.c_str()] ); 
	    Hist( "H_Mass_Collinear", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_[channel.c_str()] ); 
	      
	      
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill( 0. ,1);
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill( 1 , 1 + 0.039 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	    Hist( "Wtag_PtExtr_syst", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill( 2 , 1 - 0.039 * TMath::Log( b_ak8jet0_pt_[channel.c_str()] /200) );
	      
	  

	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmsDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmsDown_[channel.c_str()] ); 
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrUp_[channel.c_str()] );
	    Hist( "ak8jet0_msoftdrop_corr_smear_jmrDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_ak8jet0_msoftdrop_corr_smear_jmrDown_[channel.c_str()] );

	    if  (b_met_jesUp_[channel.c_str()] >200 && b_ak8jet0_pt_jesUp_[channel.c_str()] >200  )  {
	      Hist( "X_Mass_Collinear_jesUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_jesUp_[channel.c_str()] );
	      Hist( "H_Mass_Collinear_jesUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_jesUp_[channel.c_str()] );
	    }
	  
	    if  (b_met_jesDown_[channel.c_str()] >200 && b_ak8jet0_pt_jesDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jesDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_jesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jesDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_jesDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jerUp_[channel.c_str()] >200 && b_ak8jet0_pt_jerUp_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_jerUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_jerUp_[channel.c_str()] ); 
	    }
	    if  (b_met_jerDown_[channel.c_str()] >200 && b_ak8jet0_pt_jerDown_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jerDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_jerDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jerDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_jerDown_[channel.c_str()] ); 
	    }
	    if  (b_met_jer_[channel.c_str()] >200 && b_ak8jet0_pt_jer_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_jer", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_jer_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_jer", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_jer_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnUp_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_UncEnUp_[channel.c_str()] ); 
	    }
	    if  (b_met_UncEnDown_[channel.c_str()] >200 && b_ak8jet0_pt_[channel.c_str()] >200 ) {
	      Hist( "X_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_UncEnDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_UncEnDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_UncEnDown_[channel.c_str()] ); 

	    }
	    if  (b_met_tesUp_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*1.03 >20 ) {
	      Hist( "X_Mass_Collinear_tesUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_tesUp_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesUp", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_tesUp_[channel.c_str()] ); 
	    }
	    if  (b_met_tesDown_[channel.c_str()] >200 && b_tau_pt_[channel.c_str()]*0.97 >20 ) {
	      Hist( "X_Mass_Collinear_tesDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_X_Mass_Collinear_tesDown_[channel.c_str()] ); 
	      Hist( "H_Mass_Collinear_tesDown", channel.c_str()+(TString)"_HWindow_NoTau21_1SubjetTag" )->Fill(b_H_Mass_Collinear_tesDown_[channel.c_str()] ); 
	    }

	  }
	  
	}
      
      
    }
    
  }
  
}
// // if (m_doSVFit_new){
// // TLorentzVector VHTausAnalysis::applySVFitStandaloneHadronic( float cov00, float cov10, float cov11, float met, float met_phi,UZH::Tau lep1 , TLorentzVector lep2, int tau2_DM){
// //     std::cout<<"inside applySVFitHadronic standalone Had "<<std::endl;

  
// //     TLorentzVector   lBoson4;
// //     lBoson4.SetPtEtaPhiE(0,0,0,0);
  
// //     TMatrixD covMETraw(2,2);
// //     covMETraw[0][0]=  cov00;
// //     covMETraw[1][0]=  cov10;
// //     covMETraw[0][1]=  cov10;
// //     covMETraw[1][1]=  cov11;
// //     float lcov00 =  cov00;
// //     float lcov10 =  cov10;
// //     float lcov01 =  cov10;
// //     float lcov11 =  cov11;
  
// //     TLorentzVector   nullo;
// //     nullo.SetPtEtaPhiE(0,0,0,0);
// //     if(lcov00*lcov11-lcov01*lcov01 == 0) {
// //       std::cout<<"covMat det null "<<std::endl;
// //       return nullo;
// //     }

 
// //     lBoson4.SetPtEtaPhiE(0,0,0,0);
// //     svFitStandalone::Vector measuredMET(met *TMath::Cos(met_phi), met *TMath::Sin(met_phi), 0); 
// //     // setup measure tau lepton vectors 
  
// //     // std::vector<MeasuredTauLepton> measuredTauLeptons;
// //     std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;

// //     measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToHadDecay,lep1.tlv().Pt(), lep1.tlv().Eta(), lep1.tlv().Phi(), lep1.tlv().M(), lep1.decayMode()));
// //     measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToHadDecay, lep2.Pt(), lep2.Eta(), lep2.Phi(),lep2.M(),tau2_DM));
// //     // construct the class object from the minimal necesarry information
// //     // SVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMET, covMETraw, 0); 
// //     SVfitStandaloneAlgorithm algo(measuredTauLeptons,  met *TMath::Cos(met_phi),met *TMath::Sin(met_phi), covMETraw, 0);
// //     // apply customized configurations if wanted (examples are given below)
// //     //algo.maxObjFunctionCalls(5000);
// //     algo.addLogM(false);
// //     //algo.metPower(0.5)
// //     //algo.fit();
// //     // integration by VEGAS (same as function algo.integrate() that has been in use when markov chain integration had not yet been implemented)
// //     //algo.integrateVEGAS();
// //     // integration by markov chain MC

// //     // auto inputFile_visPtResolution = TFile::Open("../SVfitStandalone/data/svFitVisMassAndPtResolutionPDF.root");
// //     // // edm::FileInPath inputFileName_visPtResolution("../SVfitStandalone/data/svFitVisMassAndPtResolutionPDF.root");
// //     // TH1::AddDirectory(false);  
// //     // //TFile* inputFile_visPtResolution = new TFile(inputFileName_visPtResolution.fullPath().data());
// //     // algo.shiftVisPt(true, inputFile_visPtResolution);

// //     algo.integrateMarkovChain();

// //     if(algo.isValidSolution()){
// //       lBoson4.SetPtEtaPhiM( algo.pt(), algo.eta(), algo.phi(), algo.getMass());
// //     }
// //     else{
// //       std::cout << "sorry -- status of NLL is not valid [" << algo.isValidSolution() << "]" << std::endl;
// //       lBoson4.SetPtEtaPhiM( 0.,0,0,0);
// //     }
  
// //     measuredTauLeptons.clear(); 
// //     // delete inputFile_visPtResolution;
// //     return   lBoson4 ;
// //   }

// // TLorentzVector VHTausAnalysis::applySVFitStandaloneSemileptonicEle(float cov00, float cov10, float cov11,  float met, float met_phi, UZH::Tau lep1 , TLorentzVector lep2){
// //    std::cout<<"inside applySVFitSemileptonic standalone ele"<<std::endl;
  
// //   TLorentzVector   lBoson4;
// //   lBoson4.SetPtEtaPhiE(0,0,0,0);

// //   TMatrixD covMETraw(2,2);
// //   covMETraw[0][0]=  cov00;
// //   covMETraw[1][0]=  cov10;
// //   covMETraw[0][1]=  cov10;
// //   covMETraw[1][1]=  cov11;

// //   float lcov00 =  cov00;
// //   float lcov10 =  cov10;
// //   float lcov01 =  cov10;
// //   float lcov11 =  cov11;
  
  
// //   TLorentzVector   nullo;
// //   nullo.SetPtEtaPhiE(0,0,0,0);
// //   if(lcov00*lcov11-lcov01*lcov01 == 0) {
// //     std::cout<<"covMat det null "<<std::endl;
// //     return nullo;
// //   }
  

// //   svFitStandalone::Vector measuredMET(met *TMath::Cos(met_phi), met *TMath::Sin(met_phi), 0); 
// //   // setup measure tau lepton vectors 
// //   // svFitStandalone::LorentzVector l1(lep1.Px(), lep1.Py(), lep1.Pz(), TMath::Sqrt(lep1.M()*lep1.M()+lep1.Px()*lep1.Px()+lep1.Py()*lep1.Py()+lep1.Pz()*lep1.Pz()));
// //   // svFitStandalone::LorentzVector l2(lep2.Px(), lep2.Py(), lep2.Pz(), TMath::Sqrt(lep2.M()*lep2.M()+lep2.Px()*lep2.Px()+lep2.Py()*lep2.Py()+lep2.Pz()*lep2.Pz()));

// //   std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;

// //   //std::vector<MeasuredTauLepton> measuredTauLeptons;
// //   measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToHadDecay, lep1.tlv().Pt(), lep1.tlv().Eta(), lep1.tlv().Phi(), lep1.tlv().M(),lep1.decayMode()));
// //   measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToMuDecay,lep2.Pt(), lep2.Eta(), lep2.Phi(), lep2.M()));
// //   // construct the class object from the minimal necesarry information
// //   SVfitStandaloneAlgorithm algo(measuredTauLeptons, met *TMath::Cos(met_phi),met *TMath::Sin(met_phi), covMETraw, 0);

// //   // SVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMET, covMETraw, 0);
// //   // apply customized configurations if wanted (examples are given below)
// //   //algo.maxObjFunctionCalls(5000);
// //   algo.addLogM(false);
// //   //algo.metPower(0.5)
// //   //algo.fit();
// //   // integration by VEGAS (same as function algo.integrate() that has been in use when markov chain integration had not yet been implemented)
// //   //algo.integrateVEGAS();
// //   // integration by markov chain MC

// //   // auto inputFile_visPtResolution = TFile::Open("../SVfitStandalone/data/svFitVisMassAndPtResolutionPDF.root");
// //   // // edm::FileInPath inputFileName_visPtResolution("../SVfitStandalone/data/svFitVisMassAndPtResolutionPDF.root");
// //   // TH1::AddDirectory(false);  
// //   // //TFile* inputFile_visPtResolution = new TFile(inputFileName_visPtResolution.fullPath().data());
// //   // algo.shiftVisPt(true, inputFile_visPtResolution);

// //   algo.integrateMarkovChain();
// //   if(algo.isValidSolution()){
   
// //     lBoson4.SetPtEtaPhiM( algo.pt(), algo.eta(), algo.phi(), algo.getMass());
// //   }
// //   else{
// //     std::cout << "sorry -- status of NLL is not valid [" << algo.isValidSolution() << "]" << std::endl;
// //    lBoson4.SetPtEtaPhiM( 0.,0,0,0);
// //   }
  
// //   measuredTauLeptons.clear();
// //   //delete inputFile_visPtResolution;
// //   return   lBoson4 ;
  
// // }

// // TLorentzVector VHTausAnalysis::applySVFitStandaloneSemileptonicMu(float cov00, float cov10, float cov11,  float met, float met_phi, UZH::Tau lep1 , TLorentzVector lep2){
// //    std::cout<<"inside applySVFitSemileptonic standalone Mu "<<std::endl;
  
// //   TLorentzVector   lBoson4;
// //   lBoson4.SetPtEtaPhiE(0,0,0,0);

// //   TMatrixD covMETraw(2,2);
// //   covMETraw[0][0]=  cov00;
// //   covMETraw[1][0]=  cov10;
// //   covMETraw[0][1]=  cov10;
// //   covMETraw[1][1]=  cov11;

// //   float lcov00 =  cov00;
// //   float lcov10 =  cov10;
// //   float lcov01 =  cov10;
// //   float lcov11 =  cov11;
  
  
// //   TLorentzVector   nullo;
// //   nullo.SetPtEtaPhiE(0,0,0,0);
// //   if(lcov00*lcov11-lcov01*lcov01 == 0) {
// //     std::cout<<"covMat det null "<<std::endl;
// //     return nullo;
// //   }
  
// //   std::cout<<" measuredMET "<< std::endl;
// //   svFitStandalone::Vector measuredMET(met *TMath::Cos(met_phi), met *TMath::Sin(met_phi), 0); 
// //   // setup measure tau lepton vectors 

// //   std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;
  
// //   //std::vector<MeasuredTauLepton> measuredTauLeptons;
// //   std::cout<< "tau mass " <<lep1.tlv().M()<< " dm  " << lep1.decayMode()<< std::endl;
// //   measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToHadDecay, lep1.tlv().Pt(), lep1.tlv().Eta(), lep1.tlv().Phi(), lep1.tlv().M(), lep1.decayMode()));
// //   measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(svFitStandalone::kTauToMuDecay,lep2.Pt(), lep2.Eta(), lep2.Phi(), lep2.M()));
// //   // construct the class object from the minimal necesarry information

// //   //std::cout<<" measuredMeasuredtau algo1  "<< std::endl;

// //   SVfitStandaloneAlgorithm algo(measuredTauLeptons, met *TMath::Cos(met_phi),met *TMath::Sin(met_phi), covMETraw, 0);
// //   //std::cout<<" measuredMeasuredtau algo2  "<< std::endl;
 
// //   // SVfitStandaloneAlgorithm algo(measuredTauLeptons, measuredMET, covMETraw, 0);
// //   // apply customized configurations if wanted (examples are given below)
// //   //algo.maxObjFunctionCalls(5000);
// //   algo.addLogM(false);
// //   //algo.metPower(0.5)
// //   //algo.fit();
// //   // integration by VEGAS (same as function algo.integrate() that has been in use when markov chain integration had not yet been implemented)
// //   //algo.integrateVEGAS();
// //   // integration by markov chain MC
// //   std::cout<<" addLogM  "<< std::endl;
// //   // auto inputFile_visPtResolution = TFile::Open(((std::string) (std::getenv("SFRAME_DIR"))+ "/../SVFitTools/data/svFitVisMassAndPtResolutionPDF.root").c_str(),"READ");

// //   // std::cout<<" got file .root  "<< std::endl;
// //   // // edm::FileInPath inputFileName_visPtResolution("../SVfitStandalone/data/svFitVisMassAndPtResolutionPDF.root");
// //   // TH1::AddDirectory(false);  
// //   // //TFile* inputFile_visPtResolution = new TFile(inputFileName_visPtResolution.fullPath().data());
// //   // //algo.shiftVisPt(true, inputFile_visPtResolution);
 
// //   algo.integrateMarkovChain();
// //   std::cout<<" Called markov chain  "<< std::endl;
// //   if(algo.isValidSolution()){
   
// //     lBoson4.SetPtEtaPhiM( algo.pt(), algo.eta(), algo.phi(), algo.getMass());
// //   }
// //   else{
// //     std::cout << "sorry -- status of NLL is not valid [" << algo.isValidSolution() << "]" << std::endl;
// //    lBoson4.SetPtEtaPhiM( 0.,0,0,0);
// //   }
  
// //   measuredTauLeptons.clear();
// //   // inputFile_visPtResolution->Close();
// //   return   lBoson4 ;
  
// // }

// //  }

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

TLorentzVector VHTausAnalysis::applySVFitSemileptonic(float cov00, float cov10, float cov11,  float met, float met_phi, TLorentzVector lep1 , TLorentzVector lep2){
  // std::cout<<"inside applySVFitSemileptonic "<<std::endl;
  
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
  
  // std::cout <<"lepton 4 vec : pt "<< lep2.Pt() << " eta " <<  lep2.Eta() << " phi " <<  lep2.Phi()   << " E " <<  lep2.E()  << " M  " <<  lep2.M()  <<std::endl;
  NSVfitStandalone::Vector measuredMET(met *TMath::Cos(met_phi), met *TMath::Sin(met_phi), 0); 
  // setup measure tau lepton vectors 
  NSVfitStandalone::LorentzVector l1(lep1.Px(), lep1.Py(), lep1.Pz(), TMath::Sqrt(lep1.M()*lep1.M()+lep1.Px()*lep1.Px()+lep1.Py()*lep1.Py()+lep1.Pz()*lep1.Pz()));
  NSVfitStandalone::LorentzVector l2(lep2.Px(), lep2.Py(), lep2.Pz(), TMath::Sqrt(lep2.M()*lep2.M()+lep2.Px()*lep2.Px()+lep2.Py()*lep2.Py()+lep2.Pz()*lep2.Pz()));
  std::vector<NSVfitStandalone::MeasuredTauLepton> measuredTauLeptons;
  measuredTauLeptons.push_back(NSVfitStandalone::MeasuredTauLepton(NSVfitStandalone::kHadDecay, l1));
  measuredTauLeptons.push_back(NSVfitStandalone::MeasuredTauLepton(NSVfitStandalone::kLepDecay, l2));
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



void VHTausAnalysis::checkTrigger(const TString& directory,const std::string& channel
				  ){
  
  // std::string channel;
  // if (directory.find("mutau") != std::string::npos) channel="mutau";
  // if (directory.find("eletau") != std::string::npos) channel="eletau";
  // if (directory.find("tautau") != std::string::npos) channel="tautau";

  std::vector<std::string> trignames;

  trignames.push_back("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v");
  trignames.push_back("HLT_AK8PFJet360_TrimMass30_v");
  trignames.push_back("HLT_AK8PFHT650_TrimR0p1PT0p03Mass50_v");
  trignames.push_back("HLT_AK8PFHT660_TrimR0p1PT0p03Mass50_BTagCSV_p20_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p45_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p41_v");
  trignames.push_back("HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20_v");
  trignames.push_back("HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20_v");

  trignames.push_back("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v");

  trignames.push_back("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v");
  trignames.push_back("HLT_PFHT600_v");
  trignames.push_back("HLT_PFHT650_v") ;
  trignames.push_back("HLT_PFHT800_v") ;
  trignames.push_back("HLT_PFHT900_v") ;
  trignames.push_back("HLT_PFJet320_v") ;
  trignames.push_back("HLT_PFJet450_v") ;  
    
  trignames.push_back("HLT_IsoMu20_eta2p1_v") ;
  trignames.push_back("HLT_IsoMu24_eta2p1_v") ;
  trignames.push_back("HLT_Mu45_eta2p1_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v") ;
  trignames.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele105_CaloIdVT_GsfTrkIdT_v") ;
  trignames.push_back("HLT_Ele115_CaloIdVT_GsfTrkIdT_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau35_v") ;
  trignames.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_IsoMu16_eta2p1_MET30_JetIDCleaned_LooseIsoPFTau50_v") ;
  //H->tautau triggers   
  trignames.push_back("HLT_LooseIsoPFTau50_v") ;
  trignames.push_back("HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_IsoMu17_eta2p1_v") ;
  trignames.push_back("HLT_IsoMu18_v") ;
  trignames.push_back("HLT_IsoMu27_v") ;
  trignames.push_back("HLT_IsoMu20_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WP75_Gsf_LooseIsoPFTau20_v") ;
  trignames.push_back("HLT_Ele22_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v") ;
  trignames.push_back("HLT_Ele23_WPLoose_Gsf_v") ;
  trignames.push_back("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v") ;
  trignames.push_back("HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v") ;

  //Alternative triggers
  trignames.push_back("HLT_PFMET120_PFMHT120_IDTight_v") ;
  trignames.push_back("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v"); 
  trignames.push_back("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");
  trignames.push_back("HLT_Ele15_IsoVVVL_PFHT600");
  trignames.push_back("HLT_Mu15_IsoVVVL_PFHT600");
  trignames.push_back("HLT_Ele15_IsoVVVL_PFHT350_PFMET50");
  trignames.push_back("HLT_Mu15_IsoVVVL_PFHT350_PFMET50");
  trignames.push_back("HLT_Ele15_IsoVVVL_BTagCSV0p72_PFHT400"); 
  trignames.push_back("HLT_Mu15_IsoVVVL_BTagCSV0p72_PFHT400");
 
 
  std::vector<std::string> trignamesTau;
  //commenting out the prescaled ones
  // trignamesTau.push_back("HLT_IsoMu20_eta2p1_v");
  // trignamesTau.push_back("HLT_IsoMu24_eta2p1_v");
  // trignamesTau.push_back("HLT_Mu45_eta2p1_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WP75_Gsf_v");
  // trignamesTau.push_back("HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v");
  // trignamesTau.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v");
  // trignamesTau.push_back("HLT_Ele105_CaloIdVT_GsfTrkIdT_v");
  // trignamesTau.push_back("HLT_Ele115_CaloIdVT_GsfTrkIdT_v");
  // trignamesTau.push_back("HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau35_v");
  // trignamesTau.push_back("HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v");
  // trignamesTau.push_back("HLT_IsoMu16_eta2p1_MET30_JetIDCleaned_LooseIsoPFTau50_v");  
  //H->tautau triggers   
  // trignamesTau.push_back("HLT_LooseIsoPFTau50_v");
  trignamesTau.push_back("HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v");
  trignamesTau.push_back("HLT_IsoMu24_eta2p1_v");
  trignamesTau.push_back("HLT_IsoMu17_eta2p1_v");
  trignamesTau.push_back("HLT_IsoMu18_v");
  trignamesTau.push_back("HLT_IsoMu27_v");
  trignamesTau.push_back("HLT_IsoMu20_v");
  trignamesTau.push_back("HLT_Ele22_eta2p1_WP75_Gsf_LooseIsoPFTau20_v");
  trignamesTau.push_back("HLT_Ele22_eta2p1_WP75_Gsf_v");
  trignamesTau.push_back("HLT_Ele32_eta2p1_WP75_Gsf_v");
  trignamesTau.push_back("HLT_Ele23_WPLoose_Gsf_v");
  trignamesTau.push_back("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v");
  trignamesTau.push_back("HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v");


  //Alternative triggers
  trignamesTau.push_back("HLT_PFMET120_PFMHT120_IDTight_v");
  trignamesTau.push_back("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");
  trignamesTau.push_back("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v");

 
  std::vector<std::string> trignamesJet;
  trignamesJet.push_back("HLT_AK8PFJet360_TrimMass30_v") ;
  trignamesJet.push_back("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v") ;
  // trignames.push_back("AK8DiPFJet280_200_TrimMass30_BTagCSV0p45") ;
  trignamesJet.push_back("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v") ;
  trignamesJet.push_back("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v") ;
  trignamesJet.push_back("HLT_PFHT800_v") ;
  trignamesJet.push_back("HLT_PFHT900_v") ;

  
  std::string GENChannel_="General";
  
  bool isfired = false;
 
  if (GENChannel_=="General"){
    bool isfiredTau = false;
    bool isfiredJet = false;
 
    for( std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
      //   std::cout <<"TriggerFired : "<< (it->first) <<std::endl; 

      for( unsigned int t = 0; t < trignames.size(); ++t ){
      
	if( (it->first).find(trignames[t]) != std::string::npos){
	  // std::cout <<"TriggerFired : "<< (it->first) <<" and number "<<t<< std::endl; 
	  if(it->second ){
	    isfired = true;
	    Hist( "Trigger",directory )->Fill( t ); 




	  }

	  // else std::cout <<"Trigger NOT Fired : "<< trignames[t] <<std::endl;  	
	}
     
      }

      for( unsigned int t = 0; t < trignamesJet.size(); ++t ){
      	if( (it->first).find(trignamesJet[t]) != std::string::npos &&  it->second ){
      	  isfiredJet = true;
      	  Hist( "Trigger_Jet" ,directory)->Fill( t ); 
      	}
      }




      for( unsigned int t = 0; t < trignamesTau.size(); ++t ){
      // 	if( (it->first).find(trignamesTau[t]) != std::string::npos && it->second   // && !isfiredJet
      // 	    ){
      // 	  isfiredTau = true;
      // 	  Hist( "Trigger_Tau",directory )->Fill( t ); 
      // 	}
	if( (it->first).find(trignamesTau[t]) != std::string::npos && it->second  // && !isfiredJet
	    ){
	
	  
	  if( trignamesTau[t].find("Ele") != std::string::npos   && channel=="eletau" ){///not trigger ele 105, unisolated
	    Hist( "Trigger_Tau",directory )->Fill( t );	
	    isfiredTau = true;
	    
	  }
	  
	  else if(trignamesTau[t].find("Mu") != std::string::npos && channel=="mutau"){
	    
	    Hist( "Trigger_Tau",directory )->Fill( t );
	    isfiredTau = true;
	  }
	  
	  else if(trignamesTau[t].find("DoubleMediumIsoPFTau") != std::string::npos && channel=="tautau"){
	    
	    Hist( "Trigger_Tau",directory )->Fill( t );	
	    isfiredTau = true;
	  }  
	}
	
      } 
    }
    Hist( "Trigger_TauOrJet" ,directory)->Fill( -1 ); 	 
    if (isfiredJet || isfiredTau)    Hist( "Trigger_TauOrJet" ,directory)->Fill( 0 ); 	 
    if (isfiredJet )    Hist( "Trigger_TauOrJet" ,directory)->Fill( 1 ); 
    if (isfiredTau)    Hist( "Trigger_TauOrJet" ,directory)->Fill( 2 ); 
    if (isfiredJet && !isfiredTau )    Hist( "Trigger_TauOrJet",directory )->Fill( 3 ); 
    if (isfiredTau && !isfiredJet)    Hist( "Trigger_TauOrJet",directory )->Fill( 4 ); 
    if (!isfiredTau && !isfiredJet)    Hist( "Trigger_TauOrJet",directory )->Fill( 5 ); 
  }
  
  
  
  trignames.clear();
  trignamesJet.clear();
  trignamesTau.clear();
}


void VHTausAnalysis::checkTriggerTurnOn(){

  Book( TH1F( "HLT_AK8PFJet360_TrimMass30_pt","HLT_AK8PFJet360_TrimMass30_pt", 2000 , 0, 4000 ), "Trigger_eff") ;
  Book( TH1F( "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_pt","HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_pt", 2000 , 0, 4000 ), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_pt" ,"HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_pt" , 2000 , 0, 4000 ), "Trigger_eff");
  Book( TH1F( "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_pt","HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_pt", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT800_pt","HLT_PFHT800_pt",  2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT900_pt","HLT_PFHT900_pt", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_pt","HLT_PFHT650_pt", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFJet320_pt","HLT_PFJet320_pt", 2000 , 0, 4000), "Trigger_eff") ;

  Book( TH1F( "HLT_AK8PFJet360_TrimMass30_softdropMass","HLT_AK8PFJet360_TrimMass30_softdropMass", 200 , 0, 400 ), "Trigger_eff") ;
  Book( TH1F( "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_softdropMass","HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_softdropMass", 200 , 0, 400 ), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_softdropMass","HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_softdropMass" , 200 , 0, 400 ), "Trigger_eff");
  Book( TH1F( "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_softdropMass","HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_softdropMass", 200 , 0, 400), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT800_softdropMass","HLT_PFHT800_softdropMass", 200 , 0, 400), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT900_softdropMass","HLT_PFHT900_softdropMass", 200 , 0, 400), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_softdropMass","HLT_PFHT650_softdropMass", 200 , 0, 400), "Trigger_eff") ;
  Book( TH1F( "HLT_PFJet320_softdropMass","HLT_PFJet320_softdropMass", 200 , 0, 400), "Trigger_eff") ;

  Book( TH1F( "HLT_AK8PFJet360_TrimMass30_ht","HLT_AK8PFJet360_TrimMass30_ht", 2000 , 0, 4000 ), "Trigger_eff") ;
  Book( TH1F( "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_ht","HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_ht", 2000 , 0, 4000 ), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_ht","HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_ht" , 2000 , 0, 4000 ), "Trigger_eff");
  Book( TH1F( "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_ht","HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_ht", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT800_ht","HLT_PFHT800_ht", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT900_ht","HLT_PFHT900_ht", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_ht","HLT_PFHT650_ht", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFJet320_ht","HLT_PFJet320_ht", 2000 , 0, 4000), "Trigger_eff") ;
 
  Book( TH1F( "HLT_AK8PFJet360_TrimMass30_XMass","HLT_AK8PFJet360_TrimMass30_XMass", 2000 , 0, 4000 ), "Trigger_eff") ;
  Book( TH1F( "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_XMass","HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_XMass", 2000 , 0, 4000 ), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_XMass","HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_XMass" , 2000 , 0, 4000 ), "Trigger_eff");
  Book( TH1F( "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_XMass","HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_XMass", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT800_XMass","HLT_PFHT800_XMass", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT900_XMass","HLT_PFHT900_XMass", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFHT650_XMass","HLT_PFHT650_XMass", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PFJet320_XMass","HLT_PFJet320_XMass", 2000 , 0, 4000), "Trigger_eff") ;
 



  Book( TH1F( "HLT_OR_pt","HLT_OR_pt", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_OR_ht","HLT_OR_ht", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_OR_softdropMass","HLT_OR_softdropMass", 200 , 0, 400), "Trigger_eff") ;
  Book( TH1F( "HLT_OR_XMass","HLT_OR_XMass", 2000 , 0, 4000), "Trigger_eff") ;
  
  Book( TH1F( "HLT_PreOR_pt","HLT_PreOR_pt", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PreOR_ht","HLT_PreOR_ht", 2000 , 0, 4000), "Trigger_eff") ;
  Book( TH1F( "HLT_PreOR_softdropMass","HLT_PreOR_softdropMass", 200 , 0, 400), "Trigger_eff") ;
  Book( TH1F( "HLT_PreOR_XMass","HLT_PreOR_XMass", 2000 , 0, 4000), "Trigger_eff") ;

  std::vector<UZH::Jet> goodFatJets_trig;
  for ( int i = 0; i < (m_jetAK8.N); ++i ) {
    UZH::Jet myjet( &m_jetAK8, i );
    if (myjet.pt() > m_jetPtCut) {
      if (fabs(myjet.eta()) < m_jetEtaCut) {
	if (myjet.IDTight()) {
	  goodFatJets_trig.push_back(myjet);
	}
      }
    }
  }
  UZH::Jet Jet_trig;
  if(goodFatJets_trig.size()>=1) Jet_trig=goodFatJets_trig[0];
 

  float HT_trig=0.;


  for ( int i = 0; i < (m_jetAK4.N); ++i ) {
    UZH::Jet myjetak4( &m_jetAK4, i );
    if (fabs(myjetak4.eta()) < m_AK4jetEtaCut) {
      if (myjetak4.pt() > m_AK4jetPtCut) {


	HT = HT + myjetak4.tlv().Et();
      }
    }
  }


  bool looseTrigHT=false;
  for( std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
    if( (it->first).find("HLT_PFHT650_v") != std::string::npos  && it->second ){  
      if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50 )   Hist( "HLT_PFHT650_pt", "Trigger_eff")->Fill(Jet_trig.pt()); 
      Hist( "HLT_PFHT650_ht", "Trigger_eff")->Fill(HT); 
      if(goodFatJets_trig.size()>=1 && Jet_trig.pt()> 500  )  Hist( "HLT_PFHT650_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()) ;
      looseTrigHT=true;
      break; 
    }
  }

  bool looseTrigJet=false;
  
  for( std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
    if( (it->first).find("HLT_PFJet320_v") != std::string::npos  && it->second ){  
      if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)   Hist( "HLT_PFJet320_pt", "Trigger_eff")->Fill(Jet_trig.pt()); 
      Hist( "HLT_PFJet320_ht", "Trigger_eff")->Fill(HT); 
      if(goodFatJets_trig.size()>=1  && Jet_trig.pt()> 500)  Hist( "HLT_PFJet320_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()) ;
      looseTrigJet=true;
      break;
    }
  }




  if (looseTrigJet || looseTrigHT){
    if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)   Hist( "HLT_PreOR_pt", "Trigger_eff")->Fill(Jet_trig.pt()); 
    Hist( "HLT_PreOR_ht", "Trigger_eff")->Fill(HT); 
    if(goodFatJets_trig.size()>=1  && Jet_trig.pt()> 500)  Hist( "HLT_PreOR_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()) ;
  }

  if (  looseTrigJet){
   
    for( std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
      if( (it->first).find("HLT_AK8PFJet360_TrimMass30_v") != std::string::npos  &&  it->second){  
	if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50 )	Hist( "HLT_AK8PFJet360_TrimMass30_pt", "Trigger_eff")->Fill(Jet_trig.pt());
	Hist( "HLT_AK8PFJet360_TrimMass30_ht", "Trigger_eff")->Fill(HT);
	if(goodFatJets_trig.size()>=1 && Jet_trig.pt()> 500)   Hist( "HLT_AK8PFJet360_TrimMass30_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()) ;
      }
    } 
  }
    


  
  if ( looseTrigHT ){
    for( std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
     
      if ( (it->first).find("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v") != std::string::npos  &&  it->second){  
	if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)	Hist( "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_pt", "Trigger_eff")->Fill(Jet_trig.pt());
	Hist( "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_ht", "Trigger_eff")->Fill(HT);
	if(goodFatJets_trig.size()>=1 && Jet_trig.pt()> 500)	Hist( "HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()); 
      }
      if ( (it->first).find("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v") != std::string::npos  &&  it->second){  
	if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)	Hist( "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_pt", "Trigger_eff")->Fill(Jet_trig.pt()); 
	Hist( "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_ht", "Trigger_eff")->Fill(HT);
	if(goodFatJets_trig.size()>=1 && Jet_trig.pt()> 500)	Hist( "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()) ;
      }
      if ( (it->first).find("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v") != std::string::npos  &&  it->second){  
	if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)	Hist( "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_pt", "Trigger_eff")->Fill(Jet_trig.pt()); 
	Hist( "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_ht", "Trigger_eff")->Fill(HT);
	if(goodFatJets_trig.size()>=1 && Jet_trig.pt()> 500)	Hist( "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()) ;
      }
      if ( (it->first).find("HLT_PFHT800_v") != std::string::npos  &&  it->second){  
	if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)	Hist( "HLT_PFHT800_pt", "Trigger_eff")->Fill(Jet_trig.pt()); 
	Hist( "HLT_PFHT800_ht", "Trigger_eff")->Fill(HT);
	if(goodFatJets_trig.size()>=1 && Jet_trig.pt()> 500)	Hist( "HLT_PFHT800_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()); 
      }
      if ( (it->first).find("HLT_PFHT900_v") != std::string::npos  &&  it->second){  
	if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)	Hist( "HLT_PFHT900_pt", "Trigger_eff")->Fill(Jet_trig.pt());
	Hist( "HLT_PFHT900_ht", "Trigger_eff")->Fill(HT);
	if(goodFatJets_trig.size()>=1 && Jet_trig.pt()> 500)	Hist( "HLT_PFHT900_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr());
      }
    }
  }
  
  if ( looseTrigHT || looseTrigJet ){

    for( std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
 
      if( ((it->first).find("HLT_PFHT800_v") != std::string::npos  &&  it->second   && looseTrigHT ) || ( (it->first).find("HLT_AK8PFJet360_TrimMass30_v") != std::string::npos  &&  it->second && looseTrigJet )){
      
	if(goodFatJets_trig.size()>=1 && Jet_trig.softdrop_massCorr()>50)   Hist( "HLT_OR_pt", "Trigger_eff")->Fill(Jet_trig.pt()); 
	Hist( "HLT_OR_ht", "Trigger_eff")->Fill(HT); 
	if(goodFatJets_trig.size()>=1  && Jet_trig.pt()> 500)  Hist( "HLT_OR_softdropMass", "Trigger_eff")->Fill(Jet_trig.softdrop_massCorr()) ;
	break;
      }

      
    }
  }

  
}


void VHTausAnalysis::passTrigger_branches(const std::string& channel) {
  
  
  for (std::map<std::string,bool>::iterator it = (m_eventInfo.trigDecision)->begin(); it != (m_eventInfo.trigDecision)->end(); ++it){
   
    if ((it->first).find("HLT_PFHT800_v") != std::string::npos) {
      if (it->second == true)  b_trig_HT800_[channel.c_str()]=true;
      else  b_trig_HT800_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFHT900_v") != std::string::npos) {
      if (it->second == true)  b_trig_HT900_[channel.c_str()]=true;
      else  b_trig_HT900_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFHT650_v") != std::string::npos) {
      if (it->second == true)  b_trig_HT650_[channel.c_str()]=true;
      else  b_trig_HT650_[channel.c_str()]=false;
    }


    if ((it->first).find("HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_METNoMu90_[channel.c_str()]=true;
      else  b_trig_METNoMu90_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_METNoMu110_[channel.c_str()]=true;
      else  b_trig_METNoMu110_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_METNoMu120_[channel.c_str()]=true;
      else  b_trig_METNoMu120_[channel.c_str()]=false;
    }

    if ((it->first).find("HLT_PFMETNoMu90_JetIdCleaned_PFMHTNoMu90_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_METNoMu90_JetId_[channel.c_str()]=true;
      else  b_trig_METNoMu90_JetId_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_METNoMu110_JetId_[channel.c_str()]=true;
      else  b_trig_METNoMu110_JetId_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_METNoMu120_JetId_[channel.c_str()]=true;
      else  b_trig_METNoMu120_JetId_[channel.c_str()]=false;
    }

    if ((it->first).find("HLT_PFMET170_NoiseCleaned_v") != std::string::npos || (it->first).find("HLT_PFMET170_HBHECleaned_v") != std::string::npos ||(it->first).find("HLT_PFMET170_HBHE_BeamHaloCleaned_v") != std::string::npos) {
      if (it->second == true)  b_trig_MET170_[channel.c_str()]=true;
      else  b_trig_MET170_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFMET110_PFMHT110_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_MET110_[channel.c_str()]=true;
      else  b_trig_MET110_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFMET120_PFMHT120_IDTight_v") != std::string::npos) {
      if (it->second == true)  b_trig_MET120_[channel.c_str()]=true;
      else  b_trig_MET120_[channel.c_str()]=false;
    }

    
   


    if ((it->first).find("HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v") != std::string::npos) {
      if (it->second == true)  b_trig_HT650_WideJetMJJ950DEtaJJ1p5_[channel.c_str()]=true;
      else  b_trig_HT650_WideJetMJJ950DEtaJJ1p5_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v") != std::string::npos) {
      if (it->second == true)  b_trig_HT650_WideJetMJJ900DEtaJJ1p5_[channel.c_str()]=true;
      else  b_trig_HT650_WideJetMJJ900DEtaJJ1p5_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_AK8PFJet360_TrimMass30_v") != std::string::npos) {
      if (it->second == true)  b_trig_ak8Jet360_Trim30_[channel.c_str()]=true;
      else b_trig_ak8Jet360_Trim30_[channel.c_str()]=false;
    }
    
    if ((it->first).find("HLT_Ele22_eta2p1_WPLoose_Gsf_v") != std::string::npos   ) {
      if (it->second == true)  b_trig_Ele22Loose_[channel.c_str()]=true;
      else b_trig_Ele22Loose_[channel.c_str()]=false;
    } 
    if ((it->first).find("HLT_Ele27_eta2p1_WPLoose_Gsf_v") != std::string::npos   ) {
      if (it->second == true)  b_trig_Ele27Loose_[channel.c_str()]=true;
      else b_trig_Ele27Loose_[channel.c_str()]=false;
    } 
    if ((it->first).find("HLT_Ele27_WPTight_Gsf_v") != std::string::npos   ) {
      if (it->second == true)  b_trig_Ele27Tight_[channel.c_str()]=true;
      else b_trig_Ele27Tight_[channel.c_str()]=false;
    } 
    if ((it->first).find("HLT_Ele115_CaloIdVT_GsfTrkIdT") != std::string::npos   ) {
      if (it->second == true)  b_trig_Ele115_[channel.c_str()]=true;
      else b_trig_Ele115_[channel.c_str()]=false;
    } 

    if ((it->first).find("HLT_IsoMu20_v") != std::string::npos || (it->first).find("HLT_IsoTkMu20_v") != std::string::npos) {
      if (it->second == true)  b_trig_Mu20_[channel.c_str()]=true;
      else b_trig_Mu20_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_IsoMu22_v") != std::string::npos || (it->first).find("HLT_IsoTkMu22_v") != std::string::npos) {
      if (it->second == true)  b_trig_Mu22_[channel.c_str()]=true;
      else b_trig_Mu22_[channel.c_str()]=false;
    } 
    if ((it->first).find("HLT_IsoMu22_eta2p1_v") != std::string::npos || (it->first).find("HLT_IsoTkMu22_eta2p1_v") != std::string::npos) {
      if (it->second == true)  b_trig_Mu22_eta2p1_[channel.c_str()]=true;
      else b_trig_Mu22_eta2p1_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_IsoMu24_v") != std::string::npos || (it->first).find("HLT_IsoTkMu24_v") != std::string::npos) {
      if (it->second == true)  b_trig_Mu24_[channel.c_str()]=true;
      else b_trig_Mu24_[channel.c_str()]=false;
    }
    if ((it->first).find("HLT_IsoMu27_v") != std::string::npos || (it->first).find("HLT_IsoTkMu27_v") != std::string::npos) {
      if (it->second == true)  b_trig_Mu27_[channel.c_str()]=true;
      else b_trig_Mu27_[channel.c_str()]=false;
    }
     
    if ((it->first).find("HLT_Mu50_v") != std::string::npos || (it->first).find("HLT_TkMu50_v") != std::string::npos) {
      if (it->second == true)  b_trig_Mu50_[channel.c_str()]=true;
      else b_trig_Mu50_[channel.c_str()]=false;
    }
  }
}

void VHTausAnalysis::genFilterZtautau() {
  
  std::vector<UZH::GenParticle> goodGenPart;
  
  for ( int p = 0; p <   (m_genParticle.N) ; ++p ) {
    UZH::GenParticle mygoodGenPart( &m_genParticle, p );

    // if (fabs(mygoodGenPart.pdgId()) == 23 &&  (mygoodGenPart.dau()).size()>0) {
    //   std::cout << " mygoodGenPart.pdgId() " <<mygoodGenPart.pdgId() << " mygoodGenPart.dau[0] " << (mygoodGenPart.dau())[0] << std::endl;
    // }
    if( fabs(mygoodGenPart.pdgId()) == 15 ){
      if ( mygoodGenPart.mother()[0]==25){
	// std::cout << " Inside H : mygoodGenPart.pdgId() " <<mygoodGenPart.pdgId() << " mygoodGenPart.mother()[0] " << (mygoodGenPart.mother())[0] << std::endl;
	GenEvent_Htata_filter= true;
      }
      if( mygoodGenPart.mother()[0]==23){
	// std::cout << " Inside Z : mygoodGenPart.pdgId() " <<mygoodGenPart.pdgId() << " mygoodGenPart.mother()[0] " << (mygoodGenPart.mother())[0] << std::endl;
	GenEvent_Ztata_filter= true;
      }
      if( mygoodGenPart.mother()[0]==22){
	// std::cout << " Inside gamma : mygoodGenPart.pdgId() " <<mygoodGenPart.pdgId() << " mygoodGenPart.mother()[0] " << (mygoodGenPart.mother())[0] << std::endl;
	GenEvent_Gammatata_filter = true;
      }
    } 
  }
}
void VHTausAnalysis::PairingRecoGenLevel(const UZH::Jet& Jet, TLorentzVector lep_tlv , const UZH::Tau& tau, const UZH::MissingEt& met , TLorentzVector Higgs_lep_tlv ,TLorentzVector Res_tlv , const std::string& channel){
  
  std::vector<UZH::GenParticle> goodGenPart;
  
  TLorentzVector Particle_that_decays_to_taus; Particle_that_decays_to_taus.SetPtEtaPhiE(0,0,0,0);
  TLorentzVector Particle_that_decays_to_quarks; Particle_that_decays_to_quarks.SetPtEtaPhiE(0,0,0,0);
  TLorentzVector Hadronic_tau;  Hadronic_tau.SetPtEtaPhiE(0,0,0,0);
  TLorentzVector lepton;  lepton.SetPtEtaPhiE(0,0,0,0);
  
  bool found_Particle_to_taus=false; bool found_Particle_to_quarks=false;
  bool found_hadronic_tau=false; bool found_lepton=false;
  int N_hadronicTaus=0; int N_leptonicTaus=0;
  // std:: cout<< "Starting loop on Gen  particles "<<std::endl;
  for ( int p = 0; p <   (m_genParticle.N) ; ++p ) {
    UZH::GenParticle mygoodGenPart( &m_genParticle, p );
    // std:: cout<< "- getting dau and status "<<std::endl;
 
    if (mygoodGenPart.dau().size()>0 &&  abs(mygoodGenPart.dau()[0])==15 ){
      // std:: cout<< "- Particle_that_decays_to_taus : pdg Id "<< mygoodGenPart.pdgId() << " status "<< mygoodGenPart.status()<<std::endl;
 
      if (mygoodGenPart.status()==22){
	//	std:: cout<< "- Particle_that_decays_to_taus "<<std::endl;
 
	Particle_that_decays_to_taus.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());

	b_IdentityRG_Higgs_tata_[channel.c_str()]=mygoodGenPart.pdgId();
	found_Particle_to_taus=true;

      }
    }
    
    if(mygoodGenPart.dau().size()>0 && abs ( mygoodGenPart.dau()[0])<= 5 ){
      // std:: cout<< "- Particle_that_decays_to_quarks : pdg Id "<< mygoodGenPart.pdgId() << " status "<< mygoodGenPart.status()<<std::endl;
 
      if (mygoodGenPart.status()==22){
	//std:: cout<< "- Particle_that_decays_to_quarks "<<std::endl;
 
	Particle_that_decays_to_quarks.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());

	b_IdentityRG_Boson_qq_[channel.c_str()]=mygoodGenPart.pdgId();
	found_Particle_to_quarks=true;
      }	
    }			
    if(  mygoodGenPart.mother().size()>0  && abs  (  mygoodGenPart.mother()[0])== 15  && mygoodGenPart.status()==1 &&
	 ( abs( mygoodGenPart.pdgId())==11 ||abs( mygoodGenPart.pdgId())==13)   ){ 
      // std:: cout<< "- lepton "<<std::endl;
 
      found_lepton =true;
      lepton.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());

      b_IdentityRG_lepton_[channel.c_str()]=mygoodGenPart.pdgId();
  
    }	

   
    if(abs  (  mygoodGenPart.pdgId())== 15){


      //std:: cout<< "- tau : pdg Id "<< mygoodGenPart.pdgId() << " status "<< mygoodGenPart.status()<<	" mygoodGenPart.dau().size() "<< mygoodGenPart.dau().size()<< " mygoodGenPart.mother()[0] "<<  mygoodGenPart.mother()[0] <<std::endl;
      
      if( // mygoodGenPart.status()==2 
	 mygoodGenPart.dau().size()!=1){
	for  (unsigned int daug=0; daug< mygoodGenPart.dau().size(); daug++){
	  //std:: cout<< "1- tau :dau "<< daug <<"  mygoodGenPart.dau()[dau] " << mygoodGenPart.dau()[daug] <<  std::endl;
	} 
	
	if(  
	   (mygoodGenPart.dau().size()>1 && mygoodGenPart.dau().size()!=2 &&  mygoodGenPart.dau().size()!=3 )
	   ||
	   ( mygoodGenPart.dau().size()==2    &&  abs(mygoodGenPart.dau()[0])!=22  &&   abs(mygoodGenPart.dau()[1])!=22)
	   ||
	   ( mygoodGenPart.dau().size()==3 && 
	     ( ( abs( mygoodGenPart.dau()[0])!= 11 && abs( mygoodGenPart.dau()[1])!= 11 && abs( mygoodGenPart.dau()[2])!= 11 ) && ( abs( mygoodGenPart.dau()[0])!= 13 && abs( mygoodGenPart.dau()[1])!= 13 && abs( mygoodGenPart.dau()[2])!= 13 ) ))){
	  // std:: cout<< "- tau : mygoodGenPart.dau()[0] "<< mygoodGenPart.dau()[0] << "  mygoodGenPart.mother()[0] "<< mygoodGenPart.mother()[0] <<  std::endl;
	  
	  for  (unsigned int daug=0; daug< mygoodGenPart.dau().size(); daug++){
	    // std:: cout<< "2- tau :dau "<< daug<< "  mygoodGenPart.dau()[dau] " <<mygoodGenPart.dau()[daug]<< std::endl;
	  } 
	  N_hadronicTaus++;
	  found_hadronic_tau =true;
	  Hadronic_tau.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());
	  
	  b_IdentityRG_tau_[channel.c_str()]=mygoodGenPart.pdgId();
	}
      }
    }
    if(abs  (  mygoodGenPart.pdgId())== 15){


        
      if(   mygoodGenPart.dau().size()==3 && mygoodGenPart.status()==2 ){
	     
	//std:: cout<< "-lep tau : mygoodGenPart.dau()[0] "<< mygoodGenPart.dau()[0] << "  mygoodGenPart.mother()[0] "<< mygoodGenPart.mother()[0] <<" status "<< mygoodGenPart.status()<<  std::endl ;
	for  (unsigned int daug=0; daug< mygoodGenPart.dau().size(); daug++){

	   
	  if( abs( mygoodGenPart.dau()[daug])==11 || abs( mygoodGenPart.dau()[daug])==13)   N_leptonicTaus++;
	  
	}
      }
    }
  }
  //std:: cout<< "channel: "<< channel.c_str() <<  "  Out of Gen particles loop : found_Particle_to_taus " << found_Particle_to_taus << " found_Particle_to_quarks " << found_Particle_to_quarks  << " found_hadronic_tau " <<found_hadronic_tau << " found_lepton " <<  found_lepton <<std::endl;


  if ( found_Particle_to_taus){
    b_dRRG_Higgs_tata_[channel.c_str()]= Particle_that_decays_to_taus.DeltaR(Higgs_lep_tlv);
    b_dPtRG_Higgs_tata_[channel.c_str()]= (Particle_that_decays_to_taus-Higgs_lep_tlv).Pt()/Particle_that_decays_to_taus.Pt();
  }
  
  if ( found_Particle_to_quarks){
    b_dRRG_Boson_qq_[channel.c_str()]=Particle_that_decays_to_quarks.DeltaR(Jet.tlv());
    b_dPtRG_Boson_qq_[channel.c_str()]=(Particle_that_decays_to_quarks - Jet.tlv()).Pt()/Particle_that_decays_to_quarks.Pt() ;
  }

  if (  Res_tlv.Pt()!=0  &&  found_Particle_to_taus &&  found_Particle_to_quarks ){
    b_dRRG_Res_[channel.c_str()]= Res_tlv.DeltaR(Particle_that_decays_to_quarks+Particle_that_decays_to_taus);
    b_dPtRG_Res_[channel.c_str()]= ( Particle_that_decays_to_quarks+Particle_that_decays_to_taus - Res_tlv ).Pt()/ (Particle_that_decays_to_quarks+Particle_that_decays_to_taus).Pt();
  }

  if (found_lepton) {
    b_dRRG_lepton_[channel.c_str()]=lepton.DeltaR(lep_tlv);
    b_dPtRG_lepton_[channel.c_str()]=( lepton-lep_tlv).Pt()/lepton.Pt() ;
  } 
  if (found_hadronic_tau // && N_hadronicTaus==1
      ) {
  
    b_dRRG_tau_[channel.c_str()]=Hadronic_tau.DeltaR(tau.tlv());
    b_dPtRG_tau_[channel.c_str()]=( Hadronic_tau-tau.tlv()).Pt()/Hadronic_tau.Pt() ;

  }
  b_N_hadronicTaus_[channel.c_str()]=N_hadronicTaus;
  b_N_leptonicTaus_[channel.c_str()]=N_leptonicTaus;


  if ( found_lepton && found_hadronic_tau) {

    b_dRG_tau_lep_[channel.c_str()]=Hadronic_tau.DeltaR(lepton);
  }
}


void VHTausAnalysis::PairingGenRecoLevel(){
  
  std::vector<UZH::GenParticle> goodGenPart;
  
  TLorentzVector Particle_that_decays_to_taus; Particle_that_decays_to_taus.SetPtEtaPhiE(0,0,0,0);
  TLorentzVector Particle_that_decays_to_quarks; Particle_that_decays_to_quarks.SetPtEtaPhiE(0,0,0,0);
  TLorentzVector Hadronic_tau;  Hadronic_tau.SetPtEtaPhiE(0,0,0,0);
  TLorentzVector lepton;  lepton.SetPtEtaPhiE(0,0,0,0);
  
  bool found_Particle_to_taus=false; bool found_Particle_to_quarks=false;
  bool found_hadronic_tau=false; bool found_lepton=false;
  int N_hadronicTaus=0; int N_leptonicTaus=0;
 
  int  index_V_qq =-99;  int  index_V_tata =-99; int  index_lep1 =-99; int  index_lep2 =-99; 
  int  index_tau1 =-99; int  index_tau2 =-99; 

  for ( int p = 0; p <   (m_genParticle.N) ; ++p ) {
    UZH::GenParticle mygoodGenPart( &m_genParticle, p );
   
 
    if (mygoodGenPart.dau().size()>0 &&  abs(mygoodGenPart.dau()[0])==15 ){
     
      if (mygoodGenPart.status()==22){

	Particle_that_decays_to_taus.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());

	//	b_IdentityRG_Higgs_tata_[channel.c_str()]=mygoodGenPart.pdgId();
	found_Particle_to_taus=true;
	index_V_tata =p;
      }

    }
    
    if(mygoodGenPart.dau().size()>0 && abs ( mygoodGenPart.dau()[0])<= 5 ){
     
      if (mygoodGenPart.status()==22){

	Particle_that_decays_to_quarks.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());


	found_Particle_to_quarks=true;
	index_V_qq =p;
      }	
    }			
    if(  mygoodGenPart.mother().size()>0  && abs  (  mygoodGenPart.mother()[0])== 15  && mygoodGenPart.status()==1 &&
	 ( abs( mygoodGenPart.pdgId())==11 ||abs( mygoodGenPart.pdgId())==13)   ){ 
    
      found_lepton =true;
      lepton.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());
      if (index_lep1==-99) index_lep1 =p;
      else  index_lep2 =p;
      
  
    }	

   
    if(abs  (  mygoodGenPart.pdgId())== 15){


      
      if(
	 mygoodGenPart.dau().size()!=1){

	
	if(  
	   (mygoodGenPart.dau().size()>1 && mygoodGenPart.dau().size()!=2 &&  mygoodGenPart.dau().size()!=3 )
	   ||
	   ( mygoodGenPart.dau().size()==2    &&  abs(mygoodGenPart.dau()[0])!=22  &&   abs(mygoodGenPart.dau()[1])!=22)
	   ||
	   ( mygoodGenPart.dau().size()==3 && 
	     ( ( abs( mygoodGenPart.dau()[0])!= 11 && abs( mygoodGenPart.dau()[1])!= 11 && abs( mygoodGenPart.dau()[2])!= 11 ) && ( abs( mygoodGenPart.dau()[0])!= 13 && abs( mygoodGenPart.dau()[1])!= 13 && abs( mygoodGenPart.dau()[2])!= 13 ) ))){
	  // std:: cout<< "- tau : mygoodGenPart.dau()[0] "<< mygoodGenPart.dau()[0] << "  mygoodGenPart.mother()[0] "<< mygoodGenPart.mother()[0] <<  std::endl;
	  
	 
	  N_hadronicTaus++;
	  found_hadronic_tau =true;
	  Hadronic_tau.SetPtEtaPhiE(mygoodGenPart.pt(),mygoodGenPart.eta(),mygoodGenPart.phi(),mygoodGenPart.e());
	  if (index_tau1==-99) index_tau1 =p;
	  else  index_tau2 =p;
     
	  
	}
      }
    }
    if(abs  (  mygoodGenPart.pdgId())== 15){


        
      if(   mygoodGenPart.dau().size()==3 && mygoodGenPart.status()==2 ){
	     
	//std:: cout<< "-lep tau : mygoodGenPart.dau()[0] "<< mygoodGenPart.dau()[0] << "  mygoodGenPart.mother()[0] "<< mygoodGenPart.mother()[0] <<" status "<< mygoodGenPart.status()<<  std::endl ;
	for  (unsigned int daug=0; daug< mygoodGenPart.dau().size(); daug++){

	   
	  if( abs( mygoodGenPart.dau()[daug])==11 || abs( mygoodGenPart.dau()[daug])==13)   N_leptonicTaus++;
	  
	}
      }
    }
  }
 


  
  


  
  if (N_leptonicTaus==1 && N_hadronicTaus==1  ) {


    
    UZH::GenParticle myV_tata( &m_genParticle,index_V_tata );
    // std:: cout<<"particle that decays to tau has status "  <<myV_tata.status()<<  "and PdgId"<< myV_tata.pdgId() <<std::endl ;
    UZH::GenParticle myV_qq( &m_genParticle,index_V_qq );
    UZH::GenParticle my_lep( &m_genParticle,index_lep1 );
    UZH::GenParticle my_tau( &m_genParticle,index_tau1 );
    // std::cout << "checking for Jet"<< std::endl;
    std::vector<UZH::Jet> goodFatJets_forGen; std::vector<UZH::Jet> FatJets_forGen; std::vector<UZH::Jet> btaggedFatJets_forGen;
    for ( int i = 0; i < (m_jetAK8.N); ++i ) {
      UZH::Jet myjet( &m_jetAK8, i );
      if (myjet.pt() > m_jetPtCut) {
	if (fabs(myjet.eta()) < m_jetEtaCut) {
	  FatJets_forGen.push_back(myjet);
	  if (myjet.IDTight()) {
	    goodFatJets_forGen.push_back(myjet);
	  
	  }
	}
      }
    }

    if(abs(my_lep.pdgId())==13 ){
      // std::cout  << "DeltaR jet -mu :" <<( myV_qq.tlv()).DeltaR( my_lep.tlv() )<< " DeltaR jet - tau "<<( myV_qq.tlv()).DeltaR( my_tau.tlv() )<< " DeltaR mu - tau " <<  (my_tau.tlv()).DeltaR( my_lep.tlv() )<< std::endl;

      if (index_V_qq!=-99)  Hist( "Cuts_GenReco_mutau")->Fill(0) ;


      if (index_V_qq!=-99  && FatJets_forGen.size()>0 &&   (myV_qq.tlv()).DeltaR( FatJets_forGen[0].tlv() )<0.3 ) Hist( "Cuts_GenReco_mutau")->Fill(1) ;
      if (index_V_qq!=-99  && goodFatJets_forGen.size()>0 &&  ( myV_qq.tlv()).DeltaR( goodFatJets_forGen[0].tlv() )<0.3 ) Hist( "Cuts_GenReco_mutau")->Fill(2) ;
      //// Maybe put b-tagging at 3 



       // std::cout << "checking for Muon"<< std::endl;
      if (index_lep1!=-99)  Hist( "Cuts_GenReco_mutau")->Fill(4) ;

      bool foundMuon_forGen=false;
      std::vector<UZH::Muon> goodMuons_forGen_match; std::vector<UZH::Muon> goodMuons_forGen;
  
      for ( int i = 0; i <   (m_muon.N)
	      ; ++i ) {
	UZH::Muon mymuon( &m_muon, i );
	if (mymuon.pt() > m_muonPtCut) {
	  if (fabs(mymuon.eta()) < m_muonEtaCut) { 

	 
	    if ( mymuon.isLooseMuon()==1  // &&( mymuon.SemileptonicCorrPFIso() /mymuon.pt() )<0.25
		 ){
	      goodMuons_forGen.push_back( mymuon);
	    }
	    if ( (my_lep.tlv()).DeltaR( mymuon.tlv())<0.05 ){
	    
	      //   std::cout<<" mymuon.SemileptonicPFIso() "<<  mymuon.SemileptonicPFIso() <<std::endl;
	      // std::cout<<" mymuon.pt() " << mymuon.pt() <<" mymuon.eta() " << mymuon.eta() <<" mymuon.isLooseMuon() "<<  mymuon.isLooseMuon() << " iso "  <<mymuon.SemileptonicCorrPFIso() /mymuon.pt() <<std::endl;
	    
	    
	    
	      // if (mymuon.SemileptonicCorrPFIso() /mymuon.pt() <0.25){
	      goodMuons_forGen_match.push_back(mymuon);
	    
	      foundMuon_forGen=true;
	    }
	    // }
	  }
	}
      }
    
    


      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0  ) Hist( "Cuts_GenReco_mutau")->Fill(5) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isLooseMuon()  ) Hist( "Cuts_GenReco_mutau")->Fill(6) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isLooseMuon() &&  (goodMuons_forGen_match[0].SemileptonicCorrPFIso() /goodMuons_forGen_match[0].pt() <0.25) ) Hist( "Cuts_GenReco_mutau")->Fill(7) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isLooseMuon() &&  (goodMuons_forGen_match[0].SemileptonicCorrPFIso() /goodMuons_forGen_match[0].pt() <0.15) ) Hist( "Cuts_GenReco_mutau")->Fill(8) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isLooseMuon() &&  (goodMuons_forGen_match[0].SemileptonicPFIso() /goodMuons_forGen_match[0].pt() <0.25) ) Hist( "Cuts_GenReco_mutau")->Fill(9) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isLooseMuon() &&  (goodMuons_forGen_match[0].SemileptonicPFIso() /goodMuons_forGen_match[0].pt() <0.15) ) Hist( "Cuts_GenReco_mutau")->Fill(10) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isTightMuon()  ) Hist( "Cuts_GenReco_mutau")->Fill(11) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isTightMuon() &&  (goodMuons_forGen_match[0].SemileptonicCorrPFIso() /goodMuons_forGen_match[0].pt() <0.25) ) Hist( "Cuts_GenReco_mutau")->Fill(12) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isTightMuon() &&  (goodMuons_forGen_match[0].SemileptonicCorrPFIso() /goodMuons_forGen_match[0].pt() <0.15) ) Hist( "Cuts_GenReco_mutau")->Fill(13) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isTightMuon() &&  (goodMuons_forGen_match[0].SemileptonicPFIso() /goodMuons_forGen_match[0].pt() <0.25) ) Hist( "Cuts_GenReco_mutau")->Fill(14) ;
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0 && goodMuons_forGen_match[0].isTightMuon() &&  (goodMuons_forGen_match[0].SemileptonicPFIso() /goodMuons_forGen_match[0].pt() <0.15) ) Hist( "Cuts_GenReco_mutau")->Fill(15) ;


    
    

      if (index_lep1!=-99  &&  goodMuons_forGen.size()>0  ) Hist( "Cuts_GenReco_mutau")->Fill(16); 
      if (index_lep1!=-99  &&  goodMuons_forGen.size()>0  && (my_lep.tlv()).DeltaR( goodMuons_forGen[0].tlv())<0.05) Hist( "Cuts_GenReco_mutau")->Fill(17);
      if (index_lep1!=-99  &&  goodMuons_forGen_match.size()>0  && goodMuons_forGen.size()>0  &&  (goodMuons_forGen_match[0].tlv()).DeltaR(goodMuons_forGen[0].tlv()) <0.05 ) Hist( "Cuts_GenReco_mutau")->Fill(18);
      //// Maybe leaving a blank one

      // std::cout << "checking for Tau 1"<< std::endl;
    
    
      bool foundTau=false;
    
      std::vector<UZH::Tau> goodTaus_forGen_match;	std::vector<UZH::Tau> goodTaus_forGen;
      for ( int i = 0; i <   (m_tau.N)
	      ; ++i ) {
	UZH::Tau mytau( &m_tau, i );
	if (mytau.pt() > m_tauPtCut && mytau.TauType() ==2 ){
	
	  if (fabs(mytau.eta()) < m_tauEtaCut){

	    if (mytau.byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5   )goodTaus_forGen.push_back(mytau);

	    if ((my_tau.tlv()).DeltaR( mytau.tlv())<0.05 ){
	      goodTaus_forGen_match.push_back(mytau);

	      foundTau=true;
	    }
	  }
	}
      }
  
  
      for ( int i = 0; i <   (m_tau.N); ++i ) {
	UZH::Tau mytau( &m_tau, i );
	if (mytau.pt() > m_tauPtCut && mytau.TauType()==1 ){
	
	  if (fabs(mytau.eta()) < m_tauEtaCut){
	    if (mytau.byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5   )goodTaus_forGen.push_back(mytau);
 

	    if ((my_tau.tlv()).DeltaR( mytau.tlv())<0.05 ){
	    
	      goodTaus_forGen_match.push_back(mytau);
	    
	      foundTau=true;
	    }
	  }
	}
      }

       // std::cout << "checking for Tau 2"<< std::endl;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  ) Hist( "Cuts_GenReco_mutau")->Fill(20) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==2 ) Hist( "Cuts_GenReco_mutau")->Fill(21) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==2 && goodTaus_forGen_match[0].byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5) Hist( "Cuts_GenReco_mutau")->Fill(22) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==1 ) Hist( "Cuts_GenReco_mutau")->Fill(23) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==1 && goodTaus_forGen_match[0].byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5) Hist( "Cuts_GenReco_mutau")->Fill(24) ;
    
      if (index_tau1!=-99  &&  goodTaus_forGen.size()>0  && (my_tau.tlv()).DeltaR(goodTaus_forGen[0].tlv())<0.05) Hist( "Cuts_GenReco_mutau")->Fill(25) ;
      if (index_tau1!=-99  &&  goodTaus_forGen.size()>0  && goodTaus_forGen_match.size()>0  &&( goodTaus_forGen[0].tlv()).DeltaR(goodTaus_forGen_match[0].tlv())<0.05 ) Hist( "Cuts_GenReco_mutau")->Fill(26) ;

      UZH::Muon goodMuon_leppt;
      UZH::Tau goodTau_mutau_leppt;
      UZH::Muon goodMuon_pairpt;
      UZH::Tau goodTau_mutau_pairpt;

      bool found_pair_leppt=false; 
      bool found_pair_pairpt=false;
    
      if (goodMuons_forGen.size() >0 && goodTaus_forGen.size() >0 ){ 
	TLorentzVector muon_tlv = goodMuons_forGen[0].tlv();
	TLorentzVector tau_tlv = goodTaus_forGen[0].tlv();

	if( muon_tlv.DeltaR(tau_tlv) >0.05 ){
	  goodMuon_leppt= goodMuons_forGen[0];
	  goodTau_mutau_leppt=goodTaus_forGen[0]; 
	  found_pair_leppt=true;
	}


      }
 
      float ZPtMin = -99.;
      if (goodMuons_forGen.size() >0 && goodTaus_forGen.size() >0 ){ 
	for(unsigned int mu=0; mu<goodMuons_forGen.size(); mu++){
	  for(int ta = (int)  goodTaus_forGen.size() - 1 ; ta>=0; --ta){

	
	    TLorentzVector muo_tlv = goodMuons_forGen[mu].tlv();
	    TLorentzVector tau_tlv = goodTaus_forGen[ta].tlv();
	    if(muo_tlv.DeltaR(tau_tlv)>0.05// 1 && muo_tlv.DeltaR(tau_tlv)<0.8
	       ){
	      if((muo_tlv+tau_tlv).Pt()>ZPtMin){ 
	    
		goodMuon_pairpt = goodMuons_forGen[mu];
		goodTau_mutau_pairpt = goodTaus_forGen[ta];
		
		ZPtMin = (muo_tlv+tau_tlv).Pt();
		found_pair_pairpt=true;
		
	      }
	    }
	  }
	
	}
      
      } 
      if (index_tau1!=-99  &&     found_pair_leppt )Hist( "Cuts_GenReco_mutau")->Fill(27) ;
      if (index_tau1!=-99  &&  goodMuons_forGen_match.size()>0  &&  found_pair_leppt  &&  goodMuons_forGen_match[0].tlv().DeltaR( goodMuon_leppt.tlv())  <0.05)Hist( "Cuts_GenReco_mutau")->Fill(28) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  found_pair_leppt  &&  (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_mutau_leppt.tlv())  <0.05 )Hist( "Cuts_GenReco_mutau")->Fill(29) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0 && goodMuons_forGen_match.size()>0  &&  found_pair_leppt  &&  goodMuons_forGen_match[0].tlv().DeltaR( goodMuon_leppt.tlv())  <0.05   &&  (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_mutau_leppt.tlv())  <0.05 )Hist( "Cuts_GenReco_mutau")->Fill(30) ;
      
      
      if (index_tau1!=-99  &&   found_pair_pairpt )Hist( "Cuts_GenReco_mutau")->Fill(31) ;
      if (index_tau1!=-99  &&  goodMuons_forGen_match.size()>0  &&  found_pair_pairpt  &&  goodMuons_forGen_match[0].tlv().DeltaR( goodMuon_pairpt.tlv())<0.05 )Hist( "Cuts_GenReco_mutau")->Fill(32) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  found_pair_pairpt && (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_mutau_pairpt.tlv())<0.05 )Hist( "Cuts_GenReco_mutau")->Fill(33) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodMuons_forGen_match.size()>0 &&  found_pair_pairpt  &&  goodMuons_forGen_match[0].tlv().DeltaR( goodMuon_pairpt.tlv())<0.05  && (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_mutau_pairpt.tlv())<0.05 )Hist( "Cuts_GenReco_mutau")->Fill(34) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodMuons_forGen_match.size()>0 &&  found_pair_pairpt  &&  goodMuons_forGen_match[0].tlv().DeltaR( goodMuon_pairpt.tlv())<0.05  && (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_mutau_pairpt.tlv())<0.05  &&( ((goodMuon_pairpt.tlv()).DeltaR( goodTau_mutau_pairpt.tlv() )<0.4  && ( goodMuon_pairpt.SemileptonicPFIso()-  goodTau_mutau_pairpt.pt() )/ goodMuon_pairpt.pt() <0.25 ) ||((goodMuon_pairpt.tlv()).DeltaR( goodTau_mutau_pairpt.tlv() )>0.4  && ( goodMuon_pairpt.SemileptonicPFIso() )/ goodMuon_pairpt.pt() <0.25 ))){Hist( "Cuts_GenReco_mutau")->Fill(35) ;
      

	//	std::cout << " ( goodMuon_pairpt.SemileptonicPFIso()-  goodTau_mutau_pairpt.pt() )/ goodMuon_pairpt.pt() " <<  ( goodMuon_pairpt.SemileptonicPFIso()-  goodTau_mutau_pairpt.pt() )/ goodMuon_pairpt.pt() <<"  ( goodMuon_pairpt.SemileptonicPFIso() )/ goodMuon_pairpt.pt() "<<  ( goodMuon_pairpt.SemileptonicPFIso())/ goodMuon_pairpt.pt()<<std::endl;}
      // if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  found_pair_pairpt  && fabs( goodTaus_forGen_match[0].pt()- goodTau_mutau_pairpt.pt())  <0.1)Hist( "Cuts_GenReco_mutau")->Fill(27) ;


      }
    }
    






    if (abs(my_lep.pdgId())==11){
      // std::cout  << "DeltaR jet -ele :" <<( myV_qq.tlv()).DeltaR( my_lep.tlv() )<< " DeltaR jet - tau "<<( myV_qq.tlv()).DeltaR( my_tau.tlv() )<< " DeltaR ele - tau " <<  (my_tau.tlv()).DeltaR( my_lep.tlv() )<< std::endl;

      if (index_V_qq!=-99)  Hist( "Cuts_GenReco_eletau")->Fill(0) ;
      if (index_V_qq!=-99  && FatJets_forGen.size()>0 &&   (myV_qq.tlv()).DeltaR( FatJets_forGen[0].tlv() )<0.3 ) Hist( "Cuts_GenReco_eletau")->Fill(1) ;
      if (index_V_qq!=-99  && goodFatJets_forGen.size()>0 &&  ( myV_qq.tlv()).DeltaR( goodFatJets_forGen[0].tlv() )<0.3 ) Hist( "Cuts_GenReco_eletau")->Fill(2) ;
      //// Maybe put b-tagging at 3 



      // std::cout << "checking for Eleon"<< std::endl;
      if (index_lep1!=-99)  Hist( "Cuts_GenReco_eletau")->Fill(4) ;

      bool foundElectron_forGen=false;
      std::vector<UZH::Electron> goodElectrons_forGen_match; std::vector<UZH::Electron> goodElectrons_forGen;
  
      for ( int i = 0; i <   (m_electron.N)
	      ; ++i ) {
	UZH::Electron myelectron( &m_electron, i );
	if (myelectron.pt() > m_electronPtCut) {
	  if (fabs(myelectron.eta()) < m_electronEtaCut) { 

	 
	    if ( myelectron.isVetoElectronWithoutIPandIsolation()==1 
		 // &&
		 // (( fabs(myelectron.eta()) <=  1.479 &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.0893) ||
		 //  ( fabs(myelectron.eta()) >   1.479  &&  (myelectron.SemileptonicCorrPFIso()/myelectron.pt()) < 0.121 ) )

		 ){
	      goodElectrons_forGen.push_back( myelectron);
	    }
	    if ( (my_lep.tlv()).DeltaR( myelectron.tlv())<0.05 ){
	    
	      //   std::cout<<" myelectron.SemileptonicPFIso() "<<  myelectron.SemileptonicPFIso() <<std::endl;
	       // std::cout<<" myelectron.pt() " << myelectron.pt() <<" myelectron.eta() " << myelectron.eta() <<" myelectron.isVetoElectron() "<<  myelectron.isVetoElectron() << " iso "  <<myelectron.SemileptonicCorrPFIso() /myelectron.pt() <<std::endl;
	    
	    
	    
	      // if (myelectron.SemileptonicCorrPFIso() /myelectron.pt() <0.25){
	      goodElectrons_forGen_match.push_back(myelectron);
	    
	      foundElectron_forGen=true;
	    }
	    // }
	  }
	}
      }
    
    


      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0  ) Hist( "Cuts_GenReco_eletau")->Fill(5) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isVetoElectronWithoutIPandIsolation()  ) Hist( "Cuts_GenReco_eletau")->Fill(6) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isVetoElectronWithoutIPandIsolation() &&  (goodElectrons_forGen_match[0].SemileptonicCorrPFIso() /goodElectrons_forGen_match[0].pt() <0.175) ) Hist( "Cuts_GenReco_eletau")->Fill(7) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isVetoElectronWithoutIPandIsolation() &&  (goodElectrons_forGen_match[0].SemileptonicCorrPFIso() /goodElectrons_forGen_match[0].pt() < 0.0354) ) Hist( "Cuts_GenReco_eletau")->Fill(8) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isVetoElectron()  ) Hist( "Cuts_GenReco_eletau")->Fill(9) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isVetoElectron()) Hist( "Cuts_GenReco_eletau")->Fill(10) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isTightElectronBoosted()  ) Hist( "Cuts_GenReco_eletau")->Fill(11) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isTightElectronBoosted() &&  (goodElectrons_forGen_match[0].SemileptonicCorrPFIso() /goodElectrons_forGen_match[0].pt() <0.175) ) Hist( "Cuts_GenReco_eletau")->Fill(12) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isTightElectronBoosted() &&  (goodElectrons_forGen_match[0].SemileptonicCorrPFIso() /goodElectrons_forGen_match[0].pt() < 0.0354) ) Hist( "Cuts_GenReco_eletau")->Fill(13) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isVetoElectronWithoutIPandIsolation()  ) Hist( "Cuts_GenReco_eletau")->Fill(14) ;
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0 && goodElectrons_forGen_match[0].isTightElectron()  ) Hist( "Cuts_GenReco_eletau")->Fill(15) ;


    
    

      if (index_lep1!=-99  &&  goodElectrons_forGen.size()>0  ) Hist( "Cuts_GenReco_eletau")->Fill(16); 
      if (index_lep1!=-99  &&  goodElectrons_forGen.size()>0  && (my_lep.tlv()).DeltaR( goodElectrons_forGen[0].tlv())<0.05) Hist( "Cuts_GenReco_eletau")->Fill(17);
      if (index_lep1!=-99  &&  goodElectrons_forGen_match.size()>0  && goodElectrons_forGen.size()>0  &&  (goodElectrons_forGen_match[0].tlv()).DeltaR(goodElectrons_forGen[0].tlv()) <0.05 ) Hist( "Cuts_GenReco_eletau")->Fill(18);
      //// Maybe leaving a blank one

     // std::cout << "checking for Tau 1"<< std::endl;
    
    
      bool foundTau=false;
    
      std::vector<UZH::Tau> goodTaus_forGen_match;	std::vector<UZH::Tau> goodTaus_forGen;
      for ( int i = 0; i <   (m_tau.N)
	      ; ++i ) {
	UZH::Tau mytau( &m_tau, i );
	if (mytau.pt() > m_tauPtCut && mytau.TauType() ==2 ){
	
	  if (fabs(mytau.eta()) < m_tauEtaCut){

	    if (mytau.byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5  )goodTaus_forGen.push_back(mytau);

	    if ((my_tau.tlv()).DeltaR( mytau.tlv())<0.05 ){
	      goodTaus_forGen_match.push_back(mytau);

	      foundTau=true;
	    }
	  }
	}
      }
  
  
      for ( int i = 0; i <   (m_tau.N); ++i ) {
	UZH::Tau mytau( &m_tau, i );
	if (mytau.pt() > m_tauPtCut && mytau.TauType()==1 ){
	
	  if (fabs(mytau.eta()) < m_tauEtaCut){
	    if (mytau.byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5 )goodTaus_forGen.push_back(mytau);
 

	    if ((my_tau.tlv()).DeltaR( mytau.tlv())<0.05 ){
	    
	      goodTaus_forGen_match.push_back(mytau);
	    
	      foundTau=true;
	    }
	  }
	}
      }

      // std::cout << "checking for Tau 2"<< std::endl;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  ) Hist( "Cuts_GenReco_eletau")->Fill(20) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==2 ) Hist( "Cuts_GenReco_eletau")->Fill(21) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==2 && goodTaus_forGen_match[0].byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5) Hist( "Cuts_GenReco_eletau")->Fill(22) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==1 ) Hist( "Cuts_GenReco_eletau")->Fill(23) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  && goodTaus_forGen_match[0].TauType()==1 && goodTaus_forGen_match[0].byVLooseIsolationMVArun2v1PWnewDMwLT()>0.5) Hist( "Cuts_GenReco_eletau")->Fill(24) ;
    
      if (index_tau1!=-99  &&  goodTaus_forGen.size()>0  && (my_tau.tlv()).DeltaR(goodTaus_forGen[0].tlv())<0.05) Hist( "Cuts_GenReco_eletau")->Fill(25) ;
      if (index_tau1!=-99  &&  goodTaus_forGen.size()>0  && goodTaus_forGen_match.size()>0  &&( goodTaus_forGen[0].tlv()).DeltaR(goodTaus_forGen_match[0].tlv())<0.05 ) Hist( "Cuts_GenReco_eletau")->Fill(26) ;

      UZH::Electron goodElectron_leppt;
      UZH::Tau goodTau_eletau_leppt;
      UZH::Electron goodElectron_pairpt;
      UZH::Tau goodTau_eletau_pairpt;

      bool found_pair_leppt=false; 
      bool found_pair_pairpt=false;
    
      if (goodElectrons_forGen.size() >0 && goodTaus_forGen.size() >0 ){ 
	TLorentzVector electron_tlv = goodElectrons_forGen[0].tlv();
	TLorentzVector tau_tlv = goodTaus_forGen[0].tlv();

	if( electron_tlv.DeltaR(tau_tlv) >0.05 ){
	  goodElectron_leppt= goodElectrons_forGen[0];
	  goodTau_eletau_leppt=goodTaus_forGen[0]; 
	  found_pair_leppt=true;
	}


      }
 
      float ZPtMin = -99.;
      if (goodElectrons_forGen.size() >0 && goodTaus_forGen.size() >0 ){ 
	for(unsigned int ele=0; ele<goodElectrons_forGen.size(); ele++){
	  for(int ta = (int)  goodTaus_forGen.size() - 1 ; ta>=0; --ta){

	
	    TLorentzVector eleo_tlv = goodElectrons_forGen[ele].tlv();
	    TLorentzVector tau_tlv = goodTaus_forGen[ta].tlv();
	    if(eleo_tlv.DeltaR(tau_tlv)>0.05// 1 && eleo_tlv.DeltaR(tau_tlv)<0.8
	       ){
	      if((eleo_tlv+tau_tlv).Pt()>ZPtMin){ 
	    
		goodElectron_pairpt = goodElectrons_forGen[ele];
		goodTau_eletau_pairpt = goodTaus_forGen[ta];
		
		ZPtMin = (eleo_tlv+tau_tlv).Pt();
		found_pair_pairpt=true;
		
	      }
	    }
	  }
	  
	}
      
      }
      if (index_tau1!=-99  &&     found_pair_leppt )Hist( "Cuts_GenReco_eletau")->Fill(27) ;
      if (index_tau1!=-99  &&  goodElectrons_forGen_match.size()>0  &&  found_pair_leppt  &&  goodElectrons_forGen_match[0].tlv().DeltaR( goodElectron_leppt.tlv())  <0.05)Hist( "Cuts_GenReco_eletau")->Fill(28) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  found_pair_leppt  &&  (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_eletau_leppt.tlv())  <0.05 )Hist( "Cuts_GenReco_eletau")->Fill(29) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  goodElectrons_forGen_match.size()>0  &&  found_pair_leppt  &&  goodElectrons_forGen_match[0].tlv().DeltaR( goodElectron_leppt.tlv())  <0.05   &&  (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_eletau_leppt.tlv())  <0.05 )Hist( "Cuts_GenReco_eletau")->Fill(30) ;
      
      
      if (index_tau1!=-99  &&   found_pair_pairpt )Hist( "Cuts_GenReco_eletau")->Fill(31) ;
      if (index_tau1!=-99  &&  goodElectrons_forGen_match.size()>0  &&  found_pair_pairpt  &&  goodElectrons_forGen_match[0].tlv().DeltaR( goodElectron_pairpt.tlv())<0.05 )Hist( "Cuts_GenReco_eletau")->Fill(32) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  found_pair_pairpt && (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_eletau_pairpt.tlv())<0.05 )Hist( "Cuts_GenReco_eletau")->Fill(33) ;
      if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  goodElectrons_forGen_match.size()>0  &&  found_pair_pairpt  &&  goodElectrons_forGen_match[0].tlv().DeltaR( goodElectron_pairpt.tlv())<0.05  && (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_eletau_pairpt.tlv())<0.05 )Hist( "Cuts_GenReco_eletau")->Fill(34) ;
      
      if ((index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  goodElectrons_forGen_match.size()>0  &&  found_pair_pairpt  &&  goodElectrons_forGen_match[0].tlv().DeltaR( goodElectron_pairpt.tlv())<0.05  && (goodTaus_forGen_match[0].tlv()).DeltaR(  goodTau_eletau_pairpt.tlv())<0.05 ) &&
	  ( ((goodElectron_pairpt.tlv()).DeltaR(goodTau_eletau_pairpt.tlv()) <= 0.3  &&
	     (	( fabs(goodElectron_pairpt.eta()) <=  1.479 &&  ((goodElectron_pairpt.SemileptonicPFIso() -  goodTau_eletau_pairpt.pt())/goodElectron_pairpt.pt())  < 0.175 ) ||
		( fabs(goodElectron_pairpt.eta()) >   1.479 &&  ((goodElectron_pairpt.SemileptonicPFIso() -  goodTau_eletau_pairpt.pt())/goodElectron_pairpt.pt())   < 0.159 ) ))
	    ||
	    ((goodElectron_pairpt.tlv()).DeltaR(goodTau_eletau_pairpt.tlv())> 0.3  && 
	     (      ( fabs(goodElectron_pairpt.eta()) <=  1.479 && goodElectron_pairpt.SemileptonicPFIso()/goodElectron_pairpt.pt()  < 0.175  ) ||
		    ( fabs(goodElectron_pairpt.eta()) >   1.479 && goodElectron_pairpt.SemileptonicPFIso()/goodElectron_pairpt.pt()  < 0.159 ) )   ))  
	  ){
	Hist( "Cuts_GenReco_eletau")->Fill(35) ; 
	// std:: cout << " (goodElectron_pairpt.SemileptonicPFIso() -  goodTau_eletau_pairpt.pt())/goodElectron_pairpt.pt() " << (goodElectron_pairpt.SemileptonicPFIso() -  goodTau_eletau_pairpt.pt())/goodElectron_pairpt.pt()<< " (goodElectron_pairpt.SemileptonicPFIso())/goodElectron_pairpt.pt() " << (goodElectron_pairpt.SemileptonicPFIso() )/goodElectron_pairpt.pt()<<std::endl;
	  
	  }
      // if (index_tau1!=-99  &&  goodTaus_forGen_match.size()>0  &&  found_pair_pairpt  && fabs( goodTaus_forGen_match[0].pt()- goodTau_eletau_pairpt.pt())  <0.1)Hist( "Cuts_GenReco_eletau")->Fill(27) ;
      
      
    }
    
    
    
    
    
  }
}



void VHTausAnalysis::initPuppiSD_ScaleAndRes_Weight(void  ){

 
  
  auto  filejmr = TFile::Open(((std::string) (std::getenv("SFRAME_DIR"))+ "/../VHTausResonances/weightSD/puppiSoftdropResol_Moriond17.root").c_str(),"READ");
  puppisd_resolution_cen = (TF1*)filejmr->Get("massResolution_0eta1v3");
  puppisd_resolution_for   = (TF1*)filejmr->Get("massResolution_1v3eta2v5");
  //puppisd_resolution_for->SetDirectory(0);
  //puppisd_resolution_cen->SetDirectory(0);
 
  trandom = new TRandom3();
  filejmr->Close();

   
}

float VHTausAnalysis::getPuppi_sd_mass_ScaleAndResolution(bool isMC, float oldmass, TLorentzVector puppijet_tlv, TLorentzVector AK8jet_tlv , float jerSigmaPt  , float sigmaMassScale, float sigmaMassRes  ) {

  //("oldmass" is the corrected PUPPI softdrop mass (corrected using the jet mass corrections described  in https://twiki.cern.ch/twiki/bin/view/Sandbox/PUPPIJetMassScaleAndResolution).
  // just for MC 

  float mass    = 0;   
  
  // Scale correction
  
  
  
  float jms = 1;// 0.999
  float jmsUnc = 0.0094; //0.004;
  mass = oldmass* ( jms + sigmaMassScale*jmsUnc);
  
  // Resolution correction

  float jmr       = 1;//1.079;
  float jmrUnc =  0.20;  //0.105;
  float massResolution   = 0;
  //Evaluate PUPPI softdropMC  mass resolution as a function of PUPPI jet pt and eta:
  
 
  if( fabs( puppijet_tlv.Eta()) <= 1.3 ){
    massResolution = puppisd_resolution_cen->Eval( puppijet_tlv.Pt() );

    //std::cout<< "mass resolution "<< massResolution  << std::endl;
  }
  else{
    massResolution = puppisd_resolution_for->Eval( puppijet_tlv.Pt() );
}
  
  //Used the mass already corrected for the Mass scale
  
  bool scaled = false;
  
  // First try scaling:
  
  for ( unsigned int i = 0; i < (*m_genjetAK8.genJetAK8_pt).size(); ++i ) {
    UZH::GenJet myjetGen( &m_genjetAK8, i );
    float genMass = myjetGen.genJetAK8_softdropmass();
    TLorentzVector genJet;
    genJet.SetPtEtaPhiE(  myjetGen.genJetAK8_pt(),  myjetGen.genJetAK8_eta(),  myjetGen.genJetAK8_phi(),  myjetGen.genJetAK8_e() );
    if( AK8jet_tlv.DeltaR(genJet) > 0.4 || ( fabs(AK8jet_tlv.Pt()-genJet.Pt()) > (3 * AK8jet_tlv.Pt() *jerSigmaPt))) continue;
    mass = std::max(float(0.), mass + ((jmr - sigmaMassRes*jmrUnc -1)*(mass-genMass)));
    scaled = true;
    break;
  }
  //Scaling failed, move to smearing:
  if( !scaled ){
    mass = trandom->Gaus( mass, TMath::Sqrt(jmr*jmr-1)*(massResolution-1)*mass);
  }


  return mass;
}


TLorentzVector VHTausAnalysis::doCollinearApprox(TLorentzVector met_tot,TLorentzVector lep1,TLorentzVector lep2){
  TLorentzVector Higgs;
  float a=0, b=0; 
  Higgs.SetPtEtaPhiE( 0,0,0,0);
  if (  (lep2.Px() * lep1.Py() -  lep2.Py() * lep1.Px()) !=0){
    a = (met_tot.Py() *lep2.Px() - met_tot.Px()*lep2.Py() ) / ( lep2.Px() * lep1.Py() -  lep2.Py() * lep1.Px());
    b = (met_tot.Px() *lep1.Py() - met_tot.Py()*lep1.Px() ) / ( lep2.Px() * lep1.Py() -  lep2.Py() * lep1.Px());
    Higgs= (a+1) * lep1  + (b+1) * lep2 ;
  }
  return Higgs;

}


//void VHTausAnalysis::fillHistogramsSyst(const std::string& channel){

//}
