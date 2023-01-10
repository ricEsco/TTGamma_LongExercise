#!/usr/bin/env python3

import hist
from coffea import util
from coffea.processor import accumulate
import numpy as np
import uproot
import os

from ttgamma.utils.plotting import RebinHist, SetRangeHist, GroupBy

# NOTE: your timestamps will differ!
outputMC = accumulate(
    [
        util.load("Outputs/output_MCOther_run20230110_162217.coffea"),      
        util.load("Outputs/output_MCSingleTop_run20230110_170719.coffea"),  
        util.load("Outputs/output_MCTTbar1l_run20230110_164315.coffea"),  
        util.load("Outputs/output_MCTTbar2l_run20230110_170012.coffea"), 
        util.load("Outputs/output_MCTTGamma_run20230110_171615.coffea"),
        util.load("Outputs/output_MCWJets_run20230110_161311.coffea"),
        util.load("Outputs/output_MCZJets_run20230110_155614.coffea"),
    ]
)

outputData = util.load("Outputs/output_Data_run20230110_172734.coffea")

groupingCategory = {
    "NonPrompt": [3j,4j],
    "MisID": [2j],
    "Prompt": [1j],
}

groupingMCDatasets = {
    "ZG": [
        "ZGamma_01J_5f_lowMass",
    ],
    "WG": [
        "WGamma",
    ],

    "other": [
        "TTbarPowheg_Dilepton",
        "TTbarPowheg_Semilept",
        "TTbarPowheg_Hadronic",
        "W2jets",
        "W3jets",
        "W4jets",
        "DYjetsM50",
        "ST_s_channel",
        "ST_tW_channel",
        "ST_tbarW_channel",
        "ST_tbar_channel",
        "ST_t_channel",
        "TTWtoLNu",
        "TTWtoQQ",
        "TTZtoLL",
        "GJets_HT200To400",
        "GJets_HT400To600",
        "GJets_HT600ToInf",
        "ZZ",
        "WZ",
        "WW",
        "TGJets"
    ],
    "ttgamma": [
        "TTGamma_Dilepton",
        "TTGamma_SingleLept",
        "TTGamma_Hadronic",
    ],
}
    
s = hist.tag.Slicer()

if __name__ == "__main__":
    
    # Group MC histograms
    histList = []
    for samp, sampList in groupingMCDatasets.items():
        histList += [outputMC[s] for s in sampList]

    outputMCHist = accumulate(histList)
    for key, histo in outputMCHist.items():
        if isinstance(histo, hist.Hist):
            outputMCHist[key] = GroupBy(histo, 'dataset', 'dataset', groupingMCDatasets)

    # Group data histograms
    outputDataHist = accumulate([histo for key, histo in outputData.items()])

    h = outputMCHist['M3']
    h = h[{'lepFlavor':sum}]
    h = GroupBy(h, "category", "category", groupingCategory)
    h = h[{'M3':s[::hist.rebin(10)]}]
    h = SetRangeHist(h, 'M3', 50, 550)

    hData = outputDataHist['M3'][{'lepFlavor':sum,'category':sum,'systematic':sum,'dataset':sum}]
    hData = hData[{'M3':s[::hist.rebin(10)]}]
    hData = SetRangeHist(hData, 'M3', 50, 550)

    outdir = "RootFiles"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outputFile = uproot.recreate(os.path.join(outdir, "M3_Output.root"))

    outputFile["data_obs"] = hData

    systematics = h.axes["systematic"]

    for _category in ["MisID", "NonPrompt"]:
        for _systematic in systematics:
            histname = f"{_category}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_category}"
            outputFile[histname] = h[{'dataset':sum,'category':_category,'systematic':_systematic}]

    for _dataset in ["ttgamma", "WG", "ZG", "other"]:
        for _systematic in systematics:
            histname = f"{_dataset}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_dataset}"
            outputFile[histname] = h[{'dataset':_dataset,'category':'Prompt','systematic':_systematic}]

    outputFile.close()

    '''
    nonprompt control region
    '''

    # regroup for the different photon categories, summing over all data sets.

    h = outputMCHist['photon_chIso']
    h = h[{"lepFlavor":sum}]
    h = GroupBy(h, "category", "category", groupingCategory)

    new_bins = np.array([1.15, 2.5, 4.9, 9, 14.9, 20])  # 1.14 is in the cutbased medium ID.   
    chIso_axis = hist.axis.Variable(new_bins, name='chIso', label=r"Charged Hadron Isolation");

    hData = outputDataHist['photon_chIso'][{'lepFlavor':sum}]
    hData = hData[{'category':sum,'systematic':sum,'dataset':sum}]

    outputFile = uproot.recreate(os.path.join(outdir, "Isolation_Output.root"))
    outputFile["data_obs"] = hData

    for _category in ["MisID", "NonPrompt"]:
        for _systematic in systematics:
            histname = f"{_category}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_category}"
            outputFile[histname] = RebinHist(h[{'dataset':sum,'category':_category,'systematic':_systematic}],chIso=chIso_axis)
            
    for _dataset in ["ttgamma", "WG", "ZG", "other"]:
        for _systematic in systematics:
            histname = f"{_dataset}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_dataset}"
            outputFile[histname] = RebinHist(h[{'dataset':_dataset,'category':'Prompt','systematic':_systematic}],chIso=chIso_axis)

    outputFile.close()

    '''
    Mis-ID control region
    '''

    h = outputMCHist['photon_lepton_mass_3j0t']
    h = GroupBy(h, 'category', 'category', groupingCategory)
    h = h[{'mass':s[::hist.rebin(20)]}]
    h = SetRangeHist(h,'mass',40,200)

    hData = outputDataHist['photon_lepton_mass_3j0t']
    hData = hData[{'category':sum,'systematic':sum,'dataset':sum}]
    hData = hData[{'mass':s[::hist.rebin(20)]}]
    hData = SetRangeHist(hData,'mass',40,200)

    for _lepton in ["electron", "muon"]:
        outputFile = uproot.recreate(os.path.join(outdir, f"MisID_Output_{_lepton}.root"))

        outputFile["data_obs"] = hData[{"lepFlavor":_lepton}]

        systematics = h.axes["systematic"]
        for _category in ["MisID", "NonPrompt"]:
            for _systematic in systematics:
                histname = f"{_category}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_category}"
                outputFile[histname] = h[{'category':_category,'systematic':_systematic,'lepFlavor':_lepton,'dataset':sum}]

        for _dataset in ["ttgamma", "WG", "ZG", "other"]:
            for _systematic in systematics:
                histname = f"{_dataset}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_dataset}"
                outputFile[histname] = h[{'category':'Prompt','systematic':_systematic,'lepFlavor':_lepton,'dataset':_dataset}]

        outputFile.close()
