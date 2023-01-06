#!/usr/bin/env python3
import uproot
import datetime
import logging
from coffea import util, processor, hist
from coffea.nanoevents import NanoAODSchema
from ttgamma import TTGammaProcessor
from ttgamma.utils.fileset2021 import fileset
from ttgamma.utils.crossSections import lumis, crossSections

import time
import sys
import os

import argparse

# Define mapping for running on condor
mc_group_mapping = {
    "MCTTGamma": [key for key in fileset if "TTGamma" in key],
    "MCTTbar1l": ["TTbarPowheg_Semilept", "TTbarPowheg_Hadronic"],
    "MCTTbar2l": ["TTbarPowheg_Dilepton"],
    "MCSingleTop": [key for key in fileset if "ST" in key],
    "MCZJets": [key for key in fileset if "DY" in key],
    "MCWJets": [
        key
        for key in fileset
        if "W1" in key or "W2" in key or "W3" in key or "W4" in key
    ],
}
mc_nonother = {key for group in mc_group_mapping.values() for key in group}
mc_group_mapping["MCOther"] = [
    key for key in fileset if (not key in mc_nonother) and (not "Data" in key)
]
mc_group_mapping["MCAll"] = [
    key for group in mc_group_mapping.values() for key in group
]


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
        level=logging.WARNING,
    )

    parser = argparse.ArgumentParser(
        description="Batch processing script for ttgamma analysis"
    )
    parser.add_argument(
        "mcGroup",
        choices=list(mc_group_mapping) + ["Data"],
        help="Name of process to run",
    )
    parser.add_argument("--chunksize", type=int, default=10000, help="Chunk size")
    parser.add_argument("--maxchunks", type=int, default=None, help="Max chunks")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--outdir", type=str, default="Outputs", help="Where to put the output files")
    parser.add_argument(
        "--batch", action="store_true", help="Batch mode (no progress bar)"
    )
    parser.add_argument(
        "-e",
        "--executor",
        choices=["local", "lpcjq", "debug"],
        default="local",
        help="How to run the processing",
    )
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    tstart = time.time()

    print("Running mcGroup {}".format(args.mcGroup))

    if args.executor == "local":
        if args.workers > 4:
            raise RuntimeError("You probably shouldn't run more than 4 cores locally at LPC")
        executor = processor.FuturesExecutor(
            workers=args.workers, status=not args.batch
        )
    elif args.executor == "debug":
        executor = processor.IterativeExecutor(status=not args.batch)
    elif args.executor == "lpcjq":
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        if args.workers == 1:
            print("Are you sure you want to use only one worker?")
        cluster = LPCCondorCluster(
            transfer_input_files="ttgamma",
#            log_directory="/uscms/home/ncsmith/dask_logs",
        )
        cluster.adapt(minimum=1, maximum=args.workers)
        executor = processor.DaskExecutor(client=Client(cluster), status=not args.batch)

    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=args.maxchunks,
    )

    if args.mcGroup == "Data":
        job_fileset = {key: fileset[key] for key in fileset if "Data" in key}
        output = runner(
            job_fileset,
            treename="Events",
            processor_instance=TTGammaProcessor(isMC=False),
        )
    else:
        job_fileset = {key: fileset[key] for key in mc_group_mapping[args.mcGroup]}
        output = runner(
            job_fileset,
            treename="Events",
            processor_instance=TTGammaProcessor(isMC=True),
        )

        # Compute original number of events for normalization
#        output["InputEventCount"] = processor.defaultdict_accumulator(int)
        lumi_sfs = {}
        for dataset_name, dataset_files in job_fileset.items():
            output[dataset_name]["InputEventCount"] = processor.value_accumulator(int)
            for filename in dataset_files:
                with uproot.open(filename) as fhandle:
                    output[dataset_name]["InputEventCount"] += fhandle["hEvents"].values()[2] - fhandle["hEvents"].values()[0]

            # Calculate luminosity scale factor
            lumi_sf = (
                crossSections[dataset_name]
                * lumis[2016]
                / output[dataset_name]["InputEventCount"].value
            )

            for key, obj in output[dataset_name].items():
                if isinstance(obj, hist.Hist):
                    obj.scale(lumi_sf)

    elapsed = time.time() - tstart
    print(f"Total time: {elapsed:.1f} seconds")
#    print("Total rate: %.1f events / second" % (output["EventCount"].value / elapsed))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(args.outdir, f"output_{args.mcGroup}_run{timestamp}.coffea")
    util.save(output, outfile)
    print(f"Saved output to {outfile}")
