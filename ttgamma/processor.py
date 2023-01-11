import time

import coffea.processor as processor
#from coffea import hist
#coffea hist deprecated in favor of
import hist
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod

NanoAODSchema.warn_missing_crossrefs = False

import pickle
import re

import awkward as ak
import numpy as np

from .scalefactors import (
    bJetScales,
    ele_id_err,
    ele_id_sf,
    ele_reco_err,
    ele_reco_sf,
    jet_factory,
    mu_id_err,
    mu_id_sf,
    mu_iso_err,
    mu_iso_sf,
    mu_trig_err,
    mu_trig_sf,
    puLookup,
    puLookup_Down,
    puLookup_Up,
    taggingEffLookup,
)
from .utils.crossSections import crossSections, lumis
from .utils.genParentage import maxHistoryPDGID

def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

def generatorOverlapRemoval(events, ptCut, etaCut, deltaRCut):
    """Filter generated events with overlapping phase space"""
    genmotherIdx = events.GenPart.genPartIdxMother
    genpdgid = events.GenPart.pdgId

    # potential overlap photons are only those passing the kinematic cuts
    # if the overlap photon is actually from a non prompt decay (maxParent > 37), it's not part of the phase space of the separate sample
    overlapPhoSelect = (
        (events.GenPart.pt >= ptCut)
        & (abs(events.GenPart.eta) < etaCut)
        & (events.GenPart.pdgId == 22)
        & (events.GenPart.status == 1)
        & (events.GenPart.maxParent < 37)
    )
    overlapPhotons = events.GenPart[overlapPhoSelect]

    # also require that photons are separate from all other gen particles
    # don't consider neutrinos and don't calculate the dR between the overlapPhoton and itself
    finalGen = events.GenPart[
        ((events.GenPart.status == 1) | (events.GenPart.status == 71))
        & (events.GenPart.pt > 0.01)
        & ~(
            (abs(events.GenPart.pdgId) == 12)
            | (abs(events.GenPart.pdgId) == 14)
            | (abs(events.GenPart.pdgId) == 16)
        )
        & ~overlapPhoSelect
    ]

    # calculate dR between overlap photons and each gen particle
    phoGenDR = overlapPhotons.metric_table(finalGen)
    # ensure none of them are within the deltaR cut
    phoGenMask = ak.all(phoGenDR > deltaRCut, axis=-1)

    # the event is overlapping with the separate sample if there is an overlap photon passing the dR cut, kinematic cuts, and not coming from hadronic activity
    isOverlap = ak.any(phoGenMask, axis=-1)
    return ~isOverlap


def selectMuons(events):
    """Select tight and loose muons

    Returns a tuple of (tight, loose) muons

    Tight muons should have a pt of at least 30 GeV, |eta| < 2.4, pass the tight muon ID cut
    (tightId variable), and have a relative isolation of less than 0.15

    Loose muon requirements are already coded
    """
    muonSelectTight = (
        (events.Muon.pt > 30)
        & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId)
        & (events.Muon.pfRelIso04_all < 0.15)
    )  # fixed


    muonSelectLoose = (
        (events.Muon.pt > 15)
        & (abs(events.Muon.eta) < 2.4)
        & ((events.Muon.isPFcand) & (events.Muon.isTracker | events.Muon.isGlobal))
        & (events.Muon.pfRelIso04_all < 0.25)
        & np.invert(muonSelectTight)
    )

    return events.Muon[muonSelectTight], events.Muon[muonSelectLoose]


def selectElectrons(events):
    """Select tight and loose electrons

    Returns a tuple of (tight, loose) electrons

    Tight electrons should have a pt of at least 35 GeV, |eta| < 2.1, pass the cut based electron
    id (cutBased variable in NanoAOD>=4), and pass the eta gap, DXY, and DZ cuts defined above

    Loose electron requirements are already coded
    """
    eleEtaGap = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
    elePassDXY = (abs(events.Electron.eta) < 1.479) & (
        abs(events.Electron.dxy) < 0.05
    ) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dxy) < 0.1)
    elePassDZ = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dz) < 0.1) | (
        abs(events.Electron.eta) > 1.479
    ) & (abs(events.Electron.dz) < 0.2)

    electronSelectTight = (
        (events.Electron.pt > 35)
        & (abs(events.Electron.eta) < 2.1)
        & events.Electron.cutBased>=4
        & eleEtaGap
        & elePassDXY
        & elePassDZ
    )  # FIXME 1a


    # select loose electrons
    electronSelectLoose = (
        (events.Electron.pt > 15)
        & (abs(events.Electron.eta) < 2.4)
        & eleEtaGap
        & (events.Electron.cutBased >= 1)
        & elePassDXY
        & elePassDZ
        & np.invert(electronSelectTight)
    )

    return events.Electron[electronSelectTight], events.Electron[electronSelectLoose]


def selectPhotons(photons):
    """Select tight and loose photons

    Returns a tuple of (tight, loose) photons

    The selection is implemented except for the last step of defining tight and loose
    """
    photonSelect = (
        (photons.pt > 20)
        & (abs(photons.eta) < 1.4442)
        & (photons.isScEtaEE | photons.isScEtaEB)
        & (photons.electronVeto)
        & np.invert(photons.pixelSeed)
    )

    # the whole cut-based ID is precomputed, here we ask for "medium"
    photonID = photons.cutBased >= 2

    # if we want to remove one component of the cut-based ID we can
    # split out the ID requirement using the vid (versioned ID) bitmap
    # this is enabling Iso to be inverted for control regions
    photon_MinPtCut = (photons.vidNestedWPBitmap >> 0 & 3) >= 2
    photon_PhoSCEtaMultiRangeCut = (photons.vidNestedWPBitmap >> 2 & 3) >= 2
    photon_PhoSingleTowerHadOverEmCut = (photons.vidNestedWPBitmap >> 4 & 3) >= 2
    photon_PhoFull5x5SigmaIEtaIEtaCut = (photons.vidNestedWPBitmap >> 6 & 3) >= 2
    photon_ChIsoCut = (photons.vidNestedWPBitmap >> 8 & 3) >= 2
    photon_NeuIsoCut = (photons.vidNestedWPBitmap >> 10 & 3) >= 2
    photon_PhoIsoCut = (photons.vidNestedWPBitmap >> 12 & 3) >= 2

    # photons passing all ID requirements, without the charged hadron isolation cut applied
    photonID_NoChIso = (
        photon_MinPtCut
        & photon_PhoSCEtaMultiRangeCut
        & photon_PhoSingleTowerHadOverEmCut
        & photon_PhoFull5x5SigmaIEtaIEtaCut
        & photon_NeuIsoCut
        & photon_PhoIsoCut
    )

    # select tightPhotons, the subset of photons passing the photonSelect cut and the photonID cut
    tightPhotons = photons[photonSelect & photonID]  # FIXME 1a
    # select loosePhotons, the subset of photons passing the photonSelect cut and all photonID cuts
    # except the charged hadron isolation cut applied (photonID_NoChIso)
    loosePhotons = photons[photonSelect & photonID & ~(photonID_NoChIso)]  # FIXME 1a

    return tightPhotons, loosePhotons


def categorizeGenPhoton(photon):
    """A helper function to categorize MC reconstructed photons

    Returns an integer array to label them as either a generated true photon (1),
    a mis-identified generated electron (2), a photon from a hadron decay (3),
    or a fake (e.g. from pileup) (4).
    """
    #### Photon categories, using pdgID of the matched gen particle for the leading photon in the event
    # reco photons matched to a generated photon
    # if matched_gen is None (i.e. no match), then we set the flag False
    matchedPho = ak.fill_none(photon.matched_gen.pdgId == 22, False)
    # reco photons really generated as electrons
    matchedEle = ak.fill_none(abs(photon.matched_gen.pdgId) == 11, False)
    # if the gen photon has a PDG ID > 25 in its history, it has a hadronic parent
    hadronicParent = ak.fill_none(photon.matched_gen.maxParent > 25, False)

    # define the photon categories for tight photon events
    # a genuine photon is a reconstructed photon which is matched to a generator level photon, and does not have a hadronic parent
    isGenPho = matchedPho  # FIXME 2b
    # a hadronic photon is a reconstructed photon which is matched to a generator level photon, but has a hadronic parent
    isHadPho = ~matchedPho # FIXME 2b
    # a misidentified electron is a reconstructed photon which is matched to a generator level electron
    isMisIDele = ~matchedPho # FIXME 2b matchedEle and matchedPho are exclusive
    # a hadronic/fake photon is a reconstructed photon that does not fall within any of the above categories
    isHadFake = ~matchedPho # FIXME 2b

    # integer definition for the photon category axis
    # since false = 0 , true = 1, this only leaves the integer value of the category it falls into
    return 1 * isGenPho + 2 * isMisIDele + 3 * isHadPho + 4 * isHadFake


# Look at ProcessorABC to see the expected methods and what they are supposed to do
class TTGammaProcessor(processor.ProcessorABC):
    def __init__(self, isMC=False):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################
        ak.behavior.update(nanoaod.behavior)

        self.isMC = isMC

        lep_axis = hist.axis.StrCategory([], name="lepFlavor", label="Lepton flavor", growth=True)
        
        #added to try and stop error
        #nominal_axis = hist.axis.StrCategory([], name="nominal", label="Nominal correction", growth=True)
        
        systematic_axis = hist.axis.StrCategory([], name="systematic", label="Systematic uncertainty", growth=True)
        

        m3_axis = hist.axis.Regular(200, 0.0, 1000, name="M3", label=r"$M_3$ [GeV]")
        mass_axis = hist.axis.Regular(400, 0.0, 400, name="mass", label=r"$m_{\ell\gamma}$ [GeV]")
        pt_axis = hist.axis.Regular(200, 0.0, 1000, name="pt", label=r"$p_{T}$ [GeV]")
        eta_axis = hist.axis.Regular(300, -1.5, 1.5, name="eta", label=r"$\eta_{\gamma}$")
        chIso_axis = hist.axis.Regular(400, -0.1, 20.001, name="chIso", label=r"Charged Hadron Isolation")

        ## Define axis to keep track of photon category
        phoCategory_axis = hist.axis.IntCategory([1, 2, 3, 4, 5], name="category", label=r"Photon Category")

        ## And the dataset
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="Dataset", growth=True)

        self.make_output = lambda: {
            # Test histogram; not needed for final analysis but useful to check things are working
            "all_photon_pt": hist.Hist(
                dataset_axis,
                pt_axis),
            ## book histograms for photon pt, eta, and charged hadron isolation
            "photon_pt": hist.Hist(
                dataset_axis,
                pt_axis,
                phoCategory_axis,
                lep_axis,
                systematic_axis,
            ),
            # "photon_eta": hist.Hist(...),  # FIXME 3
            "photon_chIso": hist.Hist(
                dataset_axis,
                chIso_axis,
                phoCategory_axis,
                lep_axis,
                systematic_axis,
            ),
            ## book histogram for photon/lepton mass in a 3j0t region
            "photon_lepton_mass_3j0t": hist.Hist(
                dataset_axis,
                mass_axis,
                phoCategory_axis,
                lep_axis,
                systematic_axis,
            ),
            ## book histogram for M3 variable
            "M3": hist.Hist(
                dataset_axis,
                m3_axis,
                #phoCategory_axis,
                #lep_axis,
                #systematic_axis,
            ),
            "EventCount": processor.value_accumulator(int),
        }

    def process(self, events):

        ## Here we pre-compute a few common variables that we will re-use later and add them to the events object

        # Temporary patch so we can add photon and lepton four vectors. Not needed for newer versions of NanoAOD
        events["Photon", "charge"] = 0
        # Calculate charged hadron isolation for photons
        events["Photon", "chIso"] = (events.Photon.pfRelIso03_chg) * (events.Photon.pt)

        # Calculate the maximum pdgID of any of the particles in the GenPart history
        if self.isMC:
            idx = ak.to_numpy(ak.flatten(abs(events.GenPart.pdgId)))
            par = ak.to_numpy(ak.flatten(events.GenPart.genPartIdxMother))
            num = ak.to_numpy(ak.num(events.GenPart.pdgId))
            maxParentFlatten = maxHistoryPDGID(idx, par, num)
            events["GenPart", "maxParent"] = ak.unflatten(maxParentFlatten, num)


        shift_systs = [None]
        if self.isMC:
            # Jet energy systematics
            shift_systs += ["JESUp", "JESDown", "JERUp", "JERDown"]

        return processor.accumulate(self.process_shift(events, name) for name in shift_systs)

    def process_shift(self, events, shift_syst=None):
        dataset = events.metadata["dataset"]
        output = self.make_output()

        # Fill temp hist for testing purposes
        # Feel free to comment this out and copy-paste it to later in the code to check histgrams
        # after each step (for example, after defining tightPhotons
        # Remember not to fill it if we are in a shift systematic though
        if shift_syst is None:
            output["all_photon_pt"].fill(
                dataset=dataset,
                pt=ak.flatten(events.Photon.pt)
            )

        #################
        # OVERLAP REMOVAL
        #################
        # Overlap removal between related samples
        # TTGamma and TTbar
        # WGamma and WJets
        # ZGamma and ZJets
        # We need to remove events from TTbar which are already counted in the phase space in which the TTGamma sample is produced
        # photon with pT> 10 GeV, eta<5, and at least dR>0.1 from other gen objects
        if "TTbar" in dataset:
            passGenOverlapRemoval = generatorOverlapRemoval(
                events, ptCut=10.0, etaCut=5.0, deltaRCut=0.1
            )
        elif re.search("^W[1234]jets$", dataset):
            passGenOverlapRemoval = generatorOverlapRemoval(
                events, ptCut=10.0, etaCut=2.5, deltaRCut=0.05
            )
        elif "DYjetsM" in dataset:
            passGenOverlapRemoval = generatorOverlapRemoval(
                events, ptCut=15.0, etaCut=2.6, deltaRCut=0.05
            )
        else:
            passGenOverlapRemoval = np.ones(len(events), dtype=bool)

        ##################
        # OBJECT SELECTION
        ##################

        # muon and electron selections are broken out into standalone functions
        tightMuons, looseMuons = selectMuons(events)
        tightElectrons, looseElectrons = selectElectrons(events)

        ## Cross-cleaning:

        # Remove photons that are within 0.4 of a lepton
        # phoMuDR is the delta R value for each photon-muon pair
        # (metric_table default metric is delta_r)
        phoMuDR = events.Photon.metric_table(tightMuons)
        # require all muons to be away from the photon
        phoMuMask = ak.all(phoMuDR > 0.4, axis=-1)

        # an alternate way to do this cross-cleaning is to find the
        # closest object and check it is far enough away
        phoEle, phoEleDR = events.Photon.nearest(tightElectrons, return_metric=True)
        # we have to guard against the possibility there are no electrons here
        phoEleMask = ak.fill_none(phoEleDR > 0.4, True)

        # we select from only those photons that are already cross-cleaned against our tight leptons
        tightPhotons, loosePhotons = selectPhotons(
            events.Photon[phoMuMask & phoEleMask]
        )

        ## Jet objects
        # update jet kinematics based on jet energy corrections
        # in data, the corrections are already applied
        jets = events.Jet
        if self.isMC:
            events["Jet", "pt_raw"] = (1 - events.Jet.rawFactor) * events.Jet.pt
            events["Jet", "mass_raw"] = (1 - events.Jet.rawFactor) * events.Jet.mass
            events["Jet", "pt_gen"] = ak.values_astype(
                ak.fill_none(events.Jet.matched_gen.pt, 0), np.float32
            )
            events["Jet", "rho"] = ak.broadcast_arrays(
                events.fixedGridRhoFastjetAll, events.Jet.pt
            )[0]

            events_cache = events.caches[0]
            corrected_jets = jet_factory.build(events.Jet, lazy_cache=events_cache)

            # If processing a jet systematic, we need to update the
            # jets to reflect the jet systematic uncertainty variations
            if shift_syst == "JERUp":
                jets = corrected_jets.JER.up
            elif shift_syst == "JERDown":
                jets = corrected_jets.JER.down
            elif shift_syst == "JESUp":
                jets = corrected_jets.JES.up  #fixed
            elif shift_syst == "JESDown":
                #print('help!')
                jets = corrected_jets.JES.down   #fixed 
            else:
                # either nominal or some shift systematic unrelated to jets
                jets = corrected_jets

        ## More cross-cleaning: check jet does not overlap with our selected leptons or photons
        jetMuMask = ak.all(jets.metric_table(tightMuons) > 0.4, axis=-1)
        jetEleMask = ak.all(jets.metric_table(tightElectrons) > 0.4, axis=-1)
        jetPhoMask = ak.all(jets.metric_table(tightPhotons) > 0.4, axis=-1)

        # 1. ADD SELECTION
        # select good jets
        # jets should have a pt of at least 30 GeV, |eta| < 2.4, pass the medium jet id
        # (bit-wise selected from the jetID variable), and pass the cross-cleaning cuts defined above
        mediumJetIDbit = 1

        tightJet = jets[
            (abs(jets.eta) < 2.4)
            & (jets.pt > 30)
            & ((jets.jetId >> mediumJetIDbit & 1) == 1)
            & jetMuMask
            & jetEleMask
            & jetPhoMask
        ]

        # label the subset of tightJet which pass the Deep CSV tagger
        bTagWP = 0.6321  # 2016 DeepCSV working point
        tightJet["btagged"] = tightJet.btagDeepB > bTagWP #fixed 


        #####################
        # EVENT SELECTION
        #####################

        # create a PackedSelection object
        # this will help us later in composing the boolean selections easily
        selection = PackedSelection()

        # add the generatorOverlapRemoval flag we computed earlier
        selection.add("passGenOverlapRemoval", passGenOverlapRemoval)

        ## Apply triggers
        # muon events should be triggered by either the HLT_IsoMu24 or HLT_IsoTkMu24 triggers
        # electron events should be triggered by HLT_Ele27_WPTight_Gsf trigger
        # HINT: trigger values can be accessed with the variable events.HLT.TRIGGERNAME,
        # the bitwise or operator can be used to select multiple triggers events.HLT.TRIGGER1 | events.HLT.TRIGGER2
        selection.add(
            "muTrigger", events.HLT.IsoMu24 | events.HLT.IsoTkMu24
        )  # FIXME 1b -- fixed  -Ethan
        selection.add("eleTrigger", events.HLT.Ele27_WPTight_Gsf)  # FIXME 1b -- fixed -Ethan
          

        # oneMuon should be true if there is exactly one tight muon in the event
        # (the ak.num() method returns the number of objects in each row of a jagged array)
        selection.add("oneMuon", ak.num(tightMuons) == 1)
        # zeroMuon should be true if there are no tight muons in the event
        selection.add("zeroMuon", ak.num(tightMuons) == 0)#, dtype=bool))  # fixed
        # we also need to know if there are any loose muons in each event
        selection.add("zeroLooseMuon", ak.num(looseMuons) == 0) # fixed

        # similar selections will be needed for electrons
        selection.add("oneEle", ak.num(tightElectrons) == 1)  # FIXME 1b
        selection.add("zeroEle", ak.num(tightElectrons) == 0)  # FIXME 1b
        selection.add("zeroLooseEle", ak.num(looseElectrons) == 0)  # FIXME 1b

        # our overall muon category is then those events that pass:
        muon_cat = {
            "muTrigger",
            "passGenOverlapRemoval",
            "oneMuon",
            "zeroLooseMuon",          
        }

        # similarly for electrons:
        ele_cat = {
            "eleTrigger",
            "passGenOverlapRemoval",
            "oneEle",
            "zeroLooseEle",
            "zeroEle",
            
        } # FIXME 1b

        selection.add("eleSel", selection.all(*ele_cat))
        selection.add("muSel", selection.all(*muon_cat))
        # add two jet selection criteria
        #   One which selects events with at least 4 tightJet and at least one b-tagged jet
        selection.add(
            "jetSel_4j1b",
            (ak.num(tightJet) >= 4) & (ak.sum(tightJet.btagged, axis=-1) >= 1),
        )
        #   And another which selects events with at least 3 tightJet and exactly zero b-tagged jet
        selection.add(
            "jetSel_3j0b",
            (ak.num(tightJet) >= 3) & (ak.sum(tightJet.btagged, axis=-1) == 0),
        )  # fixed


        # add selection for events with exactly 0 tight photons
        selection.add("zeroPho", ak.num(tightPhotons) == 0)  # FIXME 1b -- fixed -Ethan

        # add selection for events with exactly 1 tight photon
        selection.add("onePho", ak.num(tightPhotons) == 1)  # FIXME 1b -- fixed -Ethan

        # add selection for events with exactly 1 loose photon
        selection.add("loosePho", ak.num(loosePhotons) == 1)  # FIXME 1b -- fixed -Ethan

        # useful debugger for selection efficiency
        if False and shift_syst is None:
            print(dataset)
            for n in selection.names:
                print(
                    f"- Cut {n} pass {selection.all(n).sum()} of {len(events)} events"
                )

        ##################
        # EVENT VARIABLES
        ##################

        ## Define M3, mass of 3-jet pair with highest pT
        # Find all possible combinations of 3 tight jets in the events
        # Hint: using the ak.combinations(array,n) method chooses n unique items from array.
        # More hints are in the twiki
        triJet = ak.combinations(tightJet, 3, fields=["first","second","third"])  # FIXME 2a
        # Sum together jets from the triJet object and find its pt and mass
        triJetPt = (triJet.first + triJet.second + triJet.third).pt  # FIXME 2a
        triJetMass = (triJet.first + triJet.second + triJet.third).mass  # FIXME 2a
        # define the M3 variable, the triJetMass of the combination with the highest triJetPt value
        # (ak.argmax and ak.firsts will be helpful here)
        highPtIdx = ak.argmax(triJetPt, axis=-1, keepdims=True)
        M3 = triJetMass[highPtIdx]

        # For all the other event-level variables, we can form the variables from just
        # the leading (in pt) objects rather than form all combinations and arbitrate them
        # this is because all of our signal and control regions require exactly zero or one of them
        # so there is no ambiguity to resolve
        # (highest pt, muon is muon and antimuon. objects are sorted by pt, first is always highest)
        leadingMuon = ak.firsts(tightMuons)
        leadingElectron = ak.firsts(tightElectrons)
        leadingPhoton = ak.firsts(tightPhotons)
        leadingPhotonLoose = ak.firsts(loosePhotons)

        # define egammaMass, mass of leadingElectron and leadingPhoton system
        egammaMass  = (leadingElectron + leadingPhoton).mass
        # define mugammaMass analogously

        mugammaMass = (leadingMuon + leadingPhoton).mass  # FIXME 2a
        gammaMasses = {'electron': egammaMass, 'muon': mugammaMass }

        ###################
        # PHOTON CATEGORIES
        ###################

        if self.isMC:
            phoCategory = categorizeGenPhoton(leadingPhoton)
            phoCategoryLoose = categorizeGenPhoton(leadingPhotonLoose)
        else:
            phoCategory = np.ones(len(events))
            phoCategoryLoose = np.ones(len(events))

        ################
        # EVENT WEIGHTS
        ################

        # create a processor Weights object, with the same length as the number of events in the chunk
        weights = Weights(len(events))

        if self.isMC:
            ## Note:Lumi weighting is done in postprocessing in our workflow

            # calculate pileup weights and variations
            # use the puLookup, puLookup_Up, and puLookup_Down lookup functions to find the nominal and up/down systematic weights
            # the puLookup dictionary is called with the full dataset name (datasetFull) and the number of true interactions (Pileup.nTrueInt)
            datasetFull = dataset + "_2016"  # Name for pileup lookup includes the year
            if not datasetFull in puLookup:
                print(
                    "WARNING : Using TTGamma_SingleLept_2016 pileup distribution instead of {}".format(
                        datasetFull
                    )
                )
                datasetFull = "TTGamma_SingleLept_2016"

            puWeight = puLookup[datasetFull](events.Pileup.nTrueInt)
            puWeight_Up = puLookup[datasetFull](events.Pileup.nTrueInt)  # FIXME 4
            puWeight_Down = puLookup[datasetFull](events.Pileup.nTrueInt)  # FIXME 4

            # add the puWeight and it's uncertainties to the weights container
            weights.add(
                "puWeight",
                weight=puWeight,
                weightUp=puWeight_Up,
                weightDown=puWeight_Down,
            )

            # btag key name
            # name / working Point / type / systematic / jetType
            #  ... / 0-loose 1-medium 2-tight / comb,mujets,iterativefit / central,up,down / 0-b 1-c 2-udcsg

            bJetSF = bJetScales(
                "central", tightJet.hadronFlavour, abs(tightJet.eta), tightJet.pt
            )
            bJetSF_up = bJetScales(
                "up", tightJet.hadronFlavour, abs(tightJet.eta), tightJet.pt
            )
            bJetSF_down = bJetScales(
                "down", tightJet.hadronFlavour, abs(tightJet.eta), tightJet.pt
            )

            ## mc efficiency lookup, data efficiency is eff* scale factor
            taggingName = "TTGamma_SingleLept_2016"
            if datasetFull in taggingEffLookup:
                taggingName = datasetFull
            btagEfficiencies = taggingEffLookup[taggingName](
                tightJet.hadronFlavour, tightJet.pt, abs(tightJet.eta)
            )
            btagEfficienciesData = btagEfficiencies * bJetSF
            btagEfficienciesData_up = btagEfficiencies * bJetSF_up
            btagEfficienciesData_down = btagEfficiencies * bJetSF_down

            ##probability is the product of all efficiencies of tagged jets, times product of 1-eff for all untagged jets
            ## https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
            pMC = ak.prod(btagEfficiencies[tightJet.btagged], axis=-1) * ak.prod(
                (1.0 - btagEfficiencies[np.invert(tightJet.btagged)]), axis=-1
            )
            pData = ak.prod(btagEfficienciesData[tightJet.btagged], axis=-1) * ak.prod(
                (1.0 - btagEfficienciesData[np.invert(tightJet.btagged)]), axis=-1
            )
            pData_up = ak.prod(
                btagEfficienciesData_up[tightJet.btagged], axis=-1
            ) * ak.prod(
                (1.0 - btagEfficienciesData_up[np.invert(tightJet.btagged)]), axis=-1
            )
            pData_down = ak.prod(
                btagEfficienciesData_down[tightJet.btagged], axis=-1
            ) * ak.prod(
                (1.0 - btagEfficienciesData_down[np.invert(tightJet.btagged)]), axis=-1
            )

            pMC = ak.where(pMC == 0, 1, pMC)
            btagWeight = pData / pMC
            btagWeight_up = pData_up / pMC
            btagWeight_down = pData_down / pMC

            weights.add(
                "btagWeight",
                weight=btagWeight,
                weightUp=btagWeight_up,
                weightDown=btagWeight_down,
            )

            eleID = ele_id_sf(tightElectrons.eta, tightElectrons.pt)
            eleIDerr = ele_id_err(tightElectrons.eta, tightElectrons.pt)
            eleRECO = ele_reco_sf(tightElectrons.eta, tightElectrons.pt)
            eleRECOerr = ele_reco_err(tightElectrons.eta, tightElectrons.pt)

            eleSF = ak.prod((eleID * eleRECO), axis=-1)
            eleSF_up = ak.prod(((eleID + eleIDerr) * (eleRECO + eleRECOerr)), axis=-1)
            eleSF_down = ak.prod( (eleID * eleRECO), axis=-1)  # FIXME 4
            weights.add("eleEffWeight", weight=eleSF)  # FIXME 4

            muID = mu_id_sf(tightMuons.eta, tightMuons.pt)
            muIDerr = mu_id_err(tightMuons.eta, tightMuons.pt)
            muIso = mu_iso_sf(tightMuons.eta, tightMuons.pt)
            muIsoerr = mu_iso_err(tightMuons.eta, tightMuons.pt)
            muTrig = mu_trig_sf(abs(tightMuons.eta), tightMuons.pt)
            muTrigerr = mu_trig_err(abs(tightMuons.eta), tightMuons.pt)

            muSF = ak.prod(muID * muIso * muTrig, axis=-1)
            muSF_up = ak.prod(
                (muID + muIDerr) * (muIso + muIsoerr) * (muTrig + muTrigerr), axis=-1
            )
            muSF_down = ak.prod(muID * muIso * muTrig, axis=-1)  # FIXME 4
            weights.add("muEffWeight", weight=muSF)  # FIXME 4

            # This section sets up some of the weight shifts related to theory uncertainties
            # in some samples, generator systematics are not available, in those case the systematic weights of 1. are used
            if ak.mean(ak.num(events.PSWeight)) == 1:
                weights.add(
                    "ISR",
                    weight=np.ones(len(events)),
                    weightUp=np.ones(len(events)),
                    weightDown=np.ones(len(events)),
                )
                weights.add(
                    "FSR",
                    weight=np.ones(len(events)),
                    weightUp=np.ones(len(events)),
                    weightDown=np.ones(len(events)),
                )
                weights.add(
                    "PDF",
                    weight=np.ones(len(events)),
                    weightUp=np.ones(len(events)),
                    weightDown=np.ones(len(events)),
                )
                weights.add(
                    "Q2Scale",
                    weight=np.ones(len(events)),
                    weightUp=np.ones(len(events)),
                    weightDown=np.ones(len(events)),
                )

            # Otherwise, calculate the weights and systematic variations
            else:
                # PDF Uncertainty weights
                # avoid errors from 0/0 division
                LHEPdfWeight_0 = ak.where(
                    events.LHEPdfWeight[:, 0] == 0, 1, events.LHEPdfWeight[:, 0]
                )
                LHEPdfVariation = events.LHEPdfWeight / LHEPdfWeight_0
                weights.add(
                    "PDF",
                    weight=np.ones(len(events)),
                    weightUp=ak.max(LHEPdfVariation, axis=1),
                    weightDown=ak.min(LHEPdfVariation, axis=1),
                )

                # Q2 Uncertainty weights
                if ak.mean(ak.num(events.LHEScaleWeight)) == 9:
                    scaleWeightSelector = [0, 1, 3, 5, 7, 8]
                elif ak.mean(ak.num(events.LHEScaleWeight)) == 44:
                    scaleWeightSelector = [0, 5, 15, 24, 34, 39]
                else:
                    scaleWeightSelector = []
                LHEScaleVariation = events.LHEScaleWeight[:, scaleWeightSelector]
                weights.add(
                    "Q2Scale",
                    weight=np.ones(len(events)),
                    weightUp=ak.max(LHEScaleVariation, axis=1),
                    weightDown=ak.min(LHEScaleVariation, axis=1),
                )

                # ISR / FSR uncertainty weights
                if not ak.all(
                    events.Generator.weight == events.LHEWeight.originalXWGTUP
                ):
                    psWeights = (
                        events.PSWeight
                        * events.LHEWeight.originalXWGTUP
                        / events.Generator.weight
                    )
                else:
                    psWeights = events.PSWeight

                weights.add(
                    "ISR",
                    weight=np.ones(len(events)),
                    weightUp=psWeights[:, 2],
                    weightDown=psWeights[:, 0],
                )
                weights.add(
                    "FSR",
                    weight=np.ones(len(events)),
                    weightUp=psWeights[:, 3],
                    weightDown=psWeights[:, 1],
                )

        ###################
        # FILL HISTOGRAMS
        ###################

        systList = []
        if self.isMC:
            if shift_syst is None:
                systList = [
                    "nominal",
                    # "muEffWeightUp",
                    # "muEffWeightDown",
                    # "eleEffWeightUp",
                    # "eleEffWeightDown",  # FIXME 4
                    "ISRUp",
                    "ISRDown",
                    "FSRUp",
                    "FSRDown",
                    "PDFUp",
                    "PDFDown",
                    "Q2ScaleUp",
                    "Q2ScaleDown",
                    "puWeightUp",
                    "puWeightDown",
                    "btagWeightUp",
                    "btagWeightDown",
                ]
            else:
                # if we are currently processing a shift systematic, we don't need to process any of the weight systematics
                # since those are handled in the "nominal" run
                systList = [shift_syst]
        else:
            systList = ["noweight"]

        for syst in systList:

            # find the event weight to be used when filling the histograms
            weightSyst = syst

            # in the case of 'nominal', or the jet energy systematics, no weight systematic variation is used (weightSyst=None)
            if syst in ["nominal", "JERUp", "JERDown", "JESUp", "JESDown"]:
                weightSyst = None

            if syst == "noweight":
                evtWeight = np.ones(len(events))
            else:
                # call weights.weight() with the name of the systematic to be varied
                evtWeight = weights.weight(weightSyst)

            # loop over both electron and muon selections
            for lepton in ["electron", "muon"]:
                if lepton == "electron":
                    lepSel = "eleSel"
                elif lepton == "muon":
                    lepSel = "muSel"

                # use the selection.all() method to select events passing
                # the lepton selection, 4-jet 1-tag jet selection, and either the one-photon or loose-photon selections
                phosel = selection.all(lepSel, "jetSel_4j1b", "onePho")
                phoselLoose = selection.all(
                    lepSel, "jetSel_4j1b", "zeroPho"
                )  # FIXME 3

                # fill photon_pt and photon_eta, using the leadingPhoton array, from events passing the phosel selection
                # Make sure to apply the correct mask to the category, weight, and photon pt or eta
                # Although some elements of leadingPhoton or phoCategory may be 'None' for events without a good photon,
                # in principle there should be no None elements after selection since the leadingPhoton should always be defined when phosel is True
                # Because of this, we have to use np.asarray(...) since coffea.hist can only accept numpy-compatible arrays

                output["photon_pt"].fill(
                    dataset=dataset,
                    pt=np.asarray(leadingPhoton.pt[phosel]),
                    category=np.asarray(phoCategory[phosel]),
                    lepFlavor=lepton,
                    systematic=syst,
                    weight=evtWeight[phosel],
                )

                output["photon_eta"].fill(
                    dataset=dataset,
                    eta=np.asarray(leadingPhoton.eta[phosel]),
                    category=np.asarray(phoCategory[phosel]),
                    lepFlavor=lepton,
                    systematic=syst,
                    weight=evtWeight[phosel],
                )

                # fill photon_chIso histogram, using the loosePhotons array (photons passing all cuts, except the charged hadron isolation cuts)
                output["photon_chIso"].fill(
                    dataset=dataset,
                    chIso=np.asarray(leadingPhotonLoose.chIso[phoselLoose]),
                    category=np.asarray(phoCategoryLoose[phoselLoose]),
                    lepFlavor=lepton,
                    systematic=syst,
                    weight=evtWeight[phoselLoose],
                )

                # fill M3 histogram, for events passing the phosel selection

                output["M3"].fill(
                    dataset=dataset,
                    M3=np.asarray(ak.flatten(M3[phosel])),
                #    category=np.asarray(phoCategory[phosel]),
                #    lepFlavor=lepton,
                #    systematic=syst,
                #    weight=evtWeight[phosel],
                )

            # use the selection.all() method to select events passing the eleSel or muSel selection,
            # and the 3-jet 0-btag selection, and have exactly one photon

            #phosel_3j0t = { 'electron': selection.all("eleSel", "jetSel_3j0b", "onePho"),
            #                'muon': selection.all("muSel", "jetSel_3j0b", "onePho")
            #               }

            #for lepton in phosel_3j0t.keys():
                # output["photon_lepton_mass_3j0t"].fill()  # FIXME 3

            output["EventCount"] = len(events)

        return {dataset:output}

    def postprocess(self, accumulator):
        return accumulator
