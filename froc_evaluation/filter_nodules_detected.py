import os
import math
import sys
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, StrMethodFormatter, FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np
from tqdm import tqdm

from NoduleFinding import NoduleFinding

from tools import csvTools
import csv
# Evaluation settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 500
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
diameter_mm_label2 = 'radius'
diameter_mm_label3 = 'diameter'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
FROC_maxX = 8  # Maximum value of x-axis of FROC curve
bLogPlot = True


def evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False, numberOfBootstrapSamples=1000, confidence=0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_filename: file with results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''

    results = csvTools.readCSV(results_filename)

    allCandsCAD = {}
    for seriesuid in tqdm(seriesUIDs):

        # collect candidates from result file
        nodules = {}
        header = results[0]

        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.iteritems():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.iteritems():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2

        # print 'adding candidates: ' + seriesuid
        allCandsCAD[seriesuid] = nodules

    outputs = []
    totalNumberOfCands = 0

    # -- loop over the cases
    for seriesuid in tqdm(seriesUIDs):
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(candidates.keys())

        for key, candidate in candidates.iteritems():
            # x2 = float(candidate.coordX)
            # y2 = float(candidate.coordY)
            # z2 = float(candidate.coordZ)
            outputs.append([seriesuid, candidate.coordX, candidate.coordY, candidate.coordZ, candidate.diameter_mm, candidate.CADprobability])
    print ("totalNumberOfCands {}".format(totalNumberOfCands))

    csvTools.writeCSV("./annotations/{}_filter.csv".format(os.path.splitext(os.path.basename(results_filename))[0]), outputs)


def getNodule(annotation, header, state=""):
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]

    if diameter_mm_label in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]

    if diameter_mm_label2 in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label2)]

    if diameter_mm_label3 in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label3)]

    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]

    if not state == "":
        nodule.state = state

    return nodule


def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0

    for seriesuid in seriesUIDs:
        # print 'adding nodule annotations: ' + seriesuid

        nodules = []
        numberOfIncludedNodules = 0

        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1

        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Excluded")
                nodules.append(nodule)

        allNodules[seriesuid] = nodules
        noduleCount += numberOfIncludedNodules
        noduleCountTotal += len(nodules)

    print("Total number of included nodule annotations: {}".format(noduleCount))
    print("Total number of nodule annotations: {}".format(noduleCountTotal))
    return allNodules


def collect(annotations_filename, annotations_excluded_filename, seriesuids_filename):
    annotations = csvTools.readCSV(annotations_filename)
    annotations_excluded = csvTools.readCSV(annotations_excluded_filename)
    seriesUIDs_csv = csvTools.readCSV(seriesuids_filename)

    seriesUIDs = []
    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])

    allNodules = collectNoduleAnnotations(
        annotations, annotations_excluded, seriesUIDs)

    return (allNodules, seriesUIDs)


def noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename, results_filename, outputDir):
    """
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    """

    (allNodules, seriesUIDs) = collect(annotations_filename,
                                       annotations_excluded_filename, seriesuids_filename)

    evaluateCAD(
        seriesUIDs, results_filename, outputDir, allNodules, os.path.splitext(os.path.basename(results_filename))[0],
        maxNumberOfCADMarks=200, performBootstrapping=bPerformBootstrapping,
        numberOfBootstrapSamples=bNumberOfBootstrapSamples,
        confidence=bConfidence)


if __name__ == '__main__':

    annotations_filename = sys.argv[1]
    annotations_excluded_filename = sys.argv[2]
    seriesuids_filename = sys.argv[3]
    results_filename = sys.argv[4]
    outputDir = sys.argv[5]

    print(60 * ">")
    print("processing {0}".format(results_filename))
    # execute only if run as a script
    noduleCADEvaluation(annotations_filename, annotations_excluded_filename,
                        seriesuids_filename, results_filename, outputDir)
    print("Finished!")
    print(60 * "<")
