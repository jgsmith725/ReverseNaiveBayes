#!/usr/bin/python3.6
import sys
import json
import numpy as np
import math
import os
import random
import statistics
import collections
import matplotlib.pyplot as plt
import copy

def standardizeNumeric(val, mean, stdDev):
    return (val - mean) / stdDev

def oneHotEncode(featureIdx, featureAry, trainInst):
    outArr = np.zeros(len(featureAry[featureIdx][1]))
    idx = featureAry[featureIdx][1].index(trainInst[featureIdx])
    outArr[idx]=1
    return outArr

def getInputLayer(featureAry, trainInst, meanAry, stdDevAry):
    inputLayer = []
    inputLayer.append(1) #set bias to 1
    for featureIdx in range(len(featureAry) - 1):
        if featureAry[featureIdx][1] == 'numeric':
            inputLayer = np.append(inputLayer, standardizeNumeric(trainInst[featureIdx], meanAry[featureIdx], stdDevAry[featureIdx]))
        else:
            inputLayer = np.append(inputLayer, oneHotEncode(featureIdx, featureAry, trainInst))
    return inputLayer
        
def calcMeanAry(featureAry, trainingDataAry):
    meanAry = []
    numInsts = len(trainingDataAry)
    for featureIdx in range(len(featureAry)-1):
        if featureAry[featureIdx][1] == 'numeric':
            col =  list(map(float, np.array(trainingDataAry)[:, featureIdx]))
            meanAry.append(np.divide(np.sum(col),numInsts))
        else:
            meanAry.append(0) #we don't care about the mean of categorical features, but append it to keep the right index
            
    return meanAry

def calcStdDevAry(featureAry, trainingDataAry, meanAry):
    stdDevAry = []
    numInsts = len(trainingDataAry)
    for featureIdx in range(len(featureAry)-1):
        if featureAry[featureIdx][1] == 'numeric':
            col =  list(map(float, np.array(trainingDataAry)[:, featureIdx]))    
            stdDevAry.append(math.sqrt(np.sum(np.square(np.subtract(col, meanAry[featureIdx]))) / numInsts))
        else:
            stdDevAry.append(0) #we don't care about the mean of categorical features, but append it to keep the right index    
    return stdDevAry
    
def calcOutputNode(inputLayer, weightLayer):
    netd = np.sum(np.multiply(inputLayer, weightLayer))
    return 1 / (1 + math.exp(-netd)) #apply sigmoid
    
def recalcWeigts(weightLayer, inputLayer, od, yd, learnRate):
    for i in range(len(weightLayer)):
        weightChange = -learnRate*(od - yd)*inputLayer[i]
        weightLayer[i] += weightChange
    return weightLayer
    
def calcCrossEntropyError(trainingDataAry, odLayer, ydLayer):
    error = 0
    for d in range(len(trainingDataAry)):
        error += -ydLayer[d]*math.log(odLayer[d]) - (1 - ydLayer[d])*math.log(1 - odLayer[d])
    return error

def getPred(row, featureAry, dataAry, meanAry, stdDevAry, weightLayer):
    inputLayer = getInputLayer(featureAry, dataAry[row], meanAry, stdDevAry)
    activation = calcOutputNode(inputLayer, weightLayer)
    if activation >= .5:         
        predClass = 1
    else:
        predClass = 0
    return [predClass, activation]

def calcAccuracy(featureAry, dataAry, nets, hidePrint):
    numCorrect = 0
    meanAry = calcMeanAry(featureAry, dataAry)
    stdDevAry = calcStdDevAry(featureAry, dataAry, meanAry)
    for row in range(len(dataAry)):
        #use the mean and stdDev from the training data
        ensembleActivations = []
        classLabel = featureAry[-1][1].index(dataAry[row][-1])
        for weightLayer in nets:
            [predClass, activation] = getPred(row, featureAry, dataAry, meanAry, stdDevAry, weightLayer)
            ensembleActivations.append(activation)
                    
            if not hidePrint:
                print('activation = ', format(activation,'.12f'), predClass, classLabel)
        
        activationMean = np.mean(ensembleActivations)
        if activationMean >= .5:         
            predClass = 1
        else:
            predClass = 0

        if predClass == classLabel:
            numCorrect += 1
    accuracy = numCorrect / len(dataAry)
    if not hidePrint:
        print(numCorrect, len(dataAry)-numCorrect)
    return accuracy

def getCategoryMode(col, featureAry, featureIdx):
    cntr = collections.Counter(np.array(col))
    mode = cntr.most_common(1)[0][0]
    return mode

def imputeAverage(trainingDataAry, featureIdx, classLabel, classIdx, featureAry, probDict):
    if (featureIdx, classLabel) in probDict:
        if featureAry[featureIdx][1] == 'numeric':
            mean = probDict[(featureIdx, classLabel)]
            return [mean, probDict]
        else:
            catMode = probDict[(featureIdx, classLabel)]
            return [catMode, probDict]
    else:
        npTrainingDataAry = np.array(trainingDataAry)
        rowsWithSameLabel = npTrainingDataAry[npTrainingDataAry[:,classIdx] == classLabel]
        rowsWithSameLabelAndNonnull = rowsWithSameLabel[rowsWithSameLabel[:,featureIdx] != 'null']
        if featureAry[featureIdx][1] == 'numeric':
            col =  list(map(float, np.array(rowsWithSameLabelAndNonnull)[:, featureIdx]))
            mean = np.divide(np.sum(col), len(col))
            probDict[(featureIdx, classLabel)] = round(mean)
            return [round(mean), probDict]
        else:
            col =  list(np.array(rowsWithSameLabelAndNonnull)[:, featureIdx])
            catMode = getCategoryMode(col, featureAry, featureIdx)
            probDict[(featureIdx, classLabel)] = catMode
            return [catMode, probDict]

def imputeSimple(trainingDataAry, featureIdx, featureAry, probDict):
    if featureIdx in probDict:
        if featureAry[featureIdx][1] == 'numeric':
            mean = probDict[featureIdx]
            return [mean, probDict]
        else:
            catMode = probDict[featureIdx]
            return [catMode, probDict]
    else:
        npTrainingDataAry = np.array(trainingDataAry)
        rowsNonnull = npTrainingDataAry[npTrainingDataAry[:,featureIdx] != 'null']

        if featureAry[featureIdx][1] == 'numeric':
            col =  list(map(float, np.array(rowsNonnull)[:, featureIdx]))
            mean = np.divide(np.sum(col), len(col))
            probDict[featureIdx] = mean
            return [round(mean), probDict]
        else:
            col =  list(np.array(rowsNonnull)[:, featureIdx])
            catMode = getCategoryMode(col, featureAry, featureIdx)
            probDict[featureIdx] = catMode
            return [catMode, probDict]

def imputeRandom(trainingDataAry, featureIdx, featureAry, probDict):
    if featureIdx in probDict:
        if featureAry[featureIdx][1] == 'numeric':
            minVal = probDict[featureIdx][0]
            maxVal = probDict[featureIdx][1]
            ret = random.choice(range(minVal, maxVal+1))
            return [ret, probDict]
    else:
        npTrainingDataAry = np.array(trainingDataAry)
        rowsNonnull = npTrainingDataAry[npTrainingDataAry[:,featureIdx] != 'null']

        if featureAry[featureIdx][1] == 'numeric':
            col =  list(map(float, np.array(rowsNonnull)[:, featureIdx]))
            maxVal = int(col[np.argmax(col)])
            minVal = int(col[np.argmin(col)])
            probDict[featureIdx] = [minVal, maxVal]
            ret = random.choice(range(minVal, maxVal+1))
            return [ret, probDict]
        else:
            ret = random.choice(featureAry[featureIdx][1])
            return [ret, probDict]

def imputeRNB(trainingDataAry, featureIdx, classLabel, classIdx, featureAry, probDict):
    if (featureIdx, classLabel) in probDict:
        if featureAry[featureIdx][1] == 'numeric':
            [mean, stdDev] = probDict[(featureIdx, classLabel)]
            return [round(np.random.normal(mean, stdDev, 1)[0]), probDict]
        else:
            col = probDict[(featureIdx, classLabel)]
            return [random.choice(col), probDict]
    else:
        npTrainingDataAry = np.array(trainingDataAry)
        rowsWithSameLabel = npTrainingDataAry[npTrainingDataAry[:,classIdx] == classLabel]
        rowsWithSameLabelAndNonnull = rowsWithSameLabel[rowsWithSameLabel[:,featureIdx] != 'null']  
        #filteredCols = trainingDataAry[np.ix_(rowsWithSameLabel,(col, classIdx))]
        if featureAry[featureIdx][1] == 'numeric':
            col =  list(map(float, np.array(rowsWithSameLabelAndNonnull)[:, featureIdx]))
            mean = np.divide(np.sum(col), len(col))
            stdDev = math.sqrt(np.sum(np.square(np.subtract(col, mean))) / len(col))
            probDict[(featureIdx, classLabel)] = [mean, stdDev]
            ret = np.random.normal(mean, stdDev, 1)
            #print('ret = ', ret, 'mean = ', mean, 'stddev = ', stdDev)
            return [round(ret[0]), probDict]
        else:
            col =  list(np.array(rowsWithSameLabelAndNonnull)[:, featureIdx])
            col = addLeplaceValues(col, featureAry, featureIdx)
            probDict[(featureIdx, classLabel)] = col
            ret = random.choice(col)
            #print('cat ret = ', ret)
            return [ret, probDict]

def addLeplaceValues(col, featureAry, featureIdx):
    for categoryVal in featureAry[featureIdx][1]:
        if categoryVal not in col:
            col.append(categoryVal)
        count = (np.array(col) == categoryVal).sum()
        #print('count of ', categoryVal, ' = ', count)
    return col

def imputeMissingVals(trainingDataAry, featureAry, method):
    numFeatures = len(trainingDataAry[0]) - 1 #-1 because we don't include the class label
    classIdx = numFeatures
    numRows = len(trainingDataAry)
    probDict = {}

    for row in range(numRows):
        for featureIdx in range(numFeatures):
            if trainingDataAry[row][featureIdx] == 'null':
                classLabel = trainingDataAry[row][classIdx]
                if method == 'rnb':
                    [imputedValue, probDict] = imputeRNB(trainingDataAry, featureIdx, classLabel, classIdx, featureAry, probDict)
                elif method == 'avg':
                    #this is the average based on the class label
                    [imputedValue, probDict] = imputeAverage(trainingDataAry, featureIdx, classLabel, classIdx, featureAry, probDict)
                elif method == 'simple':
                    #this method does not take the class label into account and simply replaces all null values with the most common
                    [imputedValue, probDict] = imputeSimple(trainingDataAry, featureIdx, featureAry, probDict)
                elif method == 'random':
                    #this method does not take the class label into account and simply replaces all null values with the most common
                    [imputedValue, probDict] = imputeRandom(trainingDataAry, featureIdx, featureAry, probDict)

                if featureAry[featureIdx][1] == 'numeric':
                     trainingDataAry[row][featureIdx] = imputedValue
                else:
                    trainingDataAry[row][featureIdx] = str(imputedValue)
    return trainingDataAry

def deleteRandomFeatureVals(trainingDataAry, percent):
    numFeatures = len(trainingDataAry[0]) - 1 #-1 because we don't include the class label
    numRows = len(trainingDataAry)

    totalFeatureVals = numFeatures * numRows
    numToDelete = round(totalFeatureVals * float(percent))
    if numToDelete == 0 or numToDelete > totalFeatureVals:
        return trainingDataAry

    indexesToDelete = random.sample(range(0,totalFeatureVals), numToDelete)

    for row in range(numRows):
        for col in range(numFeatures):
            if row * numFeatures + col in indexesToDelete:
                trainingDataAry[row][col] = 'null'

    return trainingDataAry

def getWeightsForEpoch(epoch, weightLayer, trainingDataAry, featureAry, learnRate, meanAry, stdDevAry, hidePrint):
    odLayer = []
    ydLayer = []
    numCorrect = 0
    for row in range(len(trainingDataAry)):
        inputLayer = getInputLayer(featureAry, trainingDataAry[row], meanAry, stdDevAry)
        
        #initialize the weight layer once we know the length of the input layer
        if weightLayer is None:
            weightLayer = np.random.uniform(low=-0.01, high=0.01, size=(1, len(inputLayer)))[0]
            
        od = calcOutputNode(inputLayer, weightLayer)
        yd = featureAry[-1][1].index(trainingDataAry[row][-1])
        odLayer.append(od)
        ydLayer.append(yd)
        
        if od >= .5:         
            predClass = 1
        else:
            predClass = 0
            
        if predClass == yd:
            numCorrect += 1
        
        #online learning so recalculate weights after each training instance
        weightLayer = recalcWeigts(weightLayer, inputLayer, od, yd, learnRate)      
                
    crossEntropyError = calcCrossEntropyError(trainingDataAry, odLayer, ydLayer)
    if not hidePrint:
        print(epoch+1, format(crossEntropyError,'.12f'), numCorrect, len(trainingDataAry) - numCorrect)
    return weightLayer

def getEnsembleAccuracy(numEpochs, numNets, learnRate, missingTrainingDataAry, featureAry, testDataAry, hidePrint):
    nets = []
    for idx in range(numNets):
        imputedTrainingDataAry = imputeMissingVals(copy.deepcopy(missingTrainingDataAry), featureAry, 'rnb')
        
        meanAry = calcMeanAry(featureAry, imputedTrainingDataAry)
        stdDevAry = calcStdDevAry(featureAry, imputedTrainingDataAry, meanAry)
        #train the data
        weightLayer = None
        for epoch in range(numEpochs):
            weightLayer = getWeightsForEpoch(epoch, weightLayer, imputedTrainingDataAry, featureAry, learnRate, meanAry, stdDevAry, hidePrint)
        
        nets.append([weightLayer])
    #test the data
    accuracy = calcAccuracy(featureAry, testDataAry, nets, hidePrint)
    return accuracy

def getAverageAccuracy(numEpochs, learnRate, missingTrainingDataAry, featureAry, testDataAry, hidePrint):
    imputedTrainingDataAry = imputeMissingVals(missingTrainingDataAry, featureAry,'avg')
    
    meanAry = calcMeanAry(featureAry, imputedTrainingDataAry)
    stdDevAry = calcStdDevAry(featureAry, imputedTrainingDataAry, meanAry)
    #train the data
    weightLayer = None
    for epoch in range(numEpochs):
        weightLayer = getWeightsForEpoch(epoch, weightLayer, imputedTrainingDataAry, featureAry, learnRate, meanAry, stdDevAry, hidePrint)
        
    #test the data
    accuracy = calcAccuracy(featureAry, testDataAry, [weightLayer], hidePrint)
    return accuracy

def getSimpleAccuracy(numEpochs, learnRate, missingTrainingDataAry, featureAry, testDataAry, hidePrint):
    imputedTrainingDataAry = imputeMissingVals(missingTrainingDataAry, featureAry,'simple')
    
    meanAry = calcMeanAry(featureAry, imputedTrainingDataAry)
    stdDevAry = calcStdDevAry(featureAry, imputedTrainingDataAry, meanAry)
    #train the data
    weightLayer = None
    for epoch in range(numEpochs):
        weightLayer = getWeightsForEpoch(epoch, weightLayer, imputedTrainingDataAry, featureAry, learnRate, meanAry, stdDevAry, hidePrint)
        
    #test the data
    accuracy = calcAccuracy(featureAry, testDataAry, [weightLayer], hidePrint)
    return accuracy

def getRandomAccuracy(numEpochs, learnRate, missingTrainingDataAry, featureAry, testDataAry, hidePrint):
    imputedTrainingDataAry = imputeMissingVals(missingTrainingDataAry, featureAry,'random')
    
    meanAry = calcMeanAry(featureAry, imputedTrainingDataAry)
    stdDevAry = calcStdDevAry(featureAry, imputedTrainingDataAry, meanAry)
    #train the data
    weightLayer = None
    for epoch in range(numEpochs):
        weightLayer = getWeightsForEpoch(epoch, weightLayer, imputedTrainingDataAry, featureAry, learnRate, meanAry, stdDevAry, hidePrint)
        
    #test the data
    accuracy = calcAccuracy(featureAry, testDataAry, [weightLayer], hidePrint)
    return accuracy

def showPlot(percentages, rnbAccuracies, ensembleAccuracies, averageAccuracies, simpleAccuracies, randomAccuracies):
    plt.plot(percentages, rnbAccuracies, label = 'RNB Imputation')
    plt.plot(percentages, ensembleAccuracies, label = 'RNB Ensemble Imputation')
    plt.plot(percentages, averageAccuracies, label = 'Average Imputation')
    plt.plot(percentages, simpleAccuracies, label = 'Simple Imputation')
    plt.plot(percentages, randomAccuracies, label = 'Random Imputation')
    plt.xlabel('Percent data missing')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of imputation methods over different missing data percentages')
    plt.xlim(0, 1)
    plt.ylim(.5, 1)
    plt.legend()
    plt.show()

def showPlotError(percentages, rnbAccuraciesOuter, ensembleAccuraciesOuter, averageAccuraciesOuter, simpleAccuraciesOuter, randomAccuraciesOuter):
    
    #mean =  np.mean(rnbAccuraciesOuter, axis=0)[0]
    #stdDev =  np.std(rnbAccuraciesOuter, axis=0)[0]

    rnbMean = np.mean(rnbAccuraciesOuter, axis=0)[0]
    rnbStd = np.std(rnbAccuraciesOuter, axis=0)[0]
    print("RNB")
    print(rnbMean)
    print(rnbStd)

    ensembleMean = np.mean(ensembleAccuraciesOuter, axis=0)[0]
    ensembleStd = np.std(ensembleAccuraciesOuter, axis=0)[0]
    print("Ensemble")
    print(ensembleMean)
    print(ensembleStd)

    averageMean = np.mean(averageAccuraciesOuter, axis=0)[0]
    averageStd = np.std(averageAccuraciesOuter, axis=0)[0]
    print("Average")
    print(averageMean)
    print(averageStd)

    #rnbMean = np.mean(simpleAccuraciesOuter, axis=0)[0]
    #rnbStd = np.std(simpleAccuraciesOuter, axis=0)[0]

    randomMean = np.mean(randomAccuraciesOuter, axis=0)[0]
    randomStd = np.std(randomAccuraciesOuter, axis=0)[0]
    print("Random")
    print(randomMean)
    print(randomStd)

    plt.figure(1)
    plt.plot(percentages, rnbMean, label="RNB Imputation")
    plt.plot(percentages, ensembleMean, label = 'RNB Ensemble Imputation')
    plt.plot(percentages, averageMean, label = 'Average Imputation')
    #plt.plot(percentages, np.mean(simpleAccuraciesOuter, axis=0)[0], label = 'Simple Imputation')
    plt.plot(percentages, randomMean, label = 'Random Imputation')
    plt.xlabel('Percent data missing')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of imputation methods over different missing data percentages')
    plt.xlim(0, 1)
    plt.ylim(.5, 1)
    plt.legend()

    plt.figure(2)
    plt.errorbar(percentages, rnbMean, rnbStd, capsize=1, label="RNB Imputation")
    plt.errorbar(percentages, ensembleMean,  ensembleStd, capsize=1, label = 'RNB Ensemble Imputation')
    plt.errorbar(percentages, averageMean, averageStd, capsize=1, label = 'Average Imputation')
    plt.errorbar(percentages, randomMean, randomStd, capsize=1, label = 'Random Imputation')
    plt.xlabel('Percent data missing')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of imputation methods over different missing data percentages w/ err')
    plt.xlim(0, 1)
    plt.ylim(.5, 1)
    plt.legend()

    plt.show()

def logisticMain(argv):
    learnRate = float(argv[1])
    trainingFileName = argv[4]
    testFileName = argv[5]
    hidePrint = False
    trainF1 = False
    numEpochs = 5
    numNets = 10
    numTrials = 50
    numPercentages = 10

    
    if len(argv) > 5:
        hidePrint = (argv[6] == 'True')
    
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    trainingFile = os.path.join(THIS_FOLDER, trainingFileName)
    testFile = os.path.join(THIS_FOLDER, testFileName)

    trainingDict = json.load(open(trainingFile))
    featureAry = trainingDict['metadata']['features']
    trainingDataAryBase = trainingDict['data']
    testDataAry = json.load(open(testFile))['data']

    rnbAccuraciesOuter = []
    ensembleAccuraciesOuter = []
    averageAccuraciesOuter = []
    simpleAccuraciesOuter = []
    randomAccuraciesOuter = []
    for trial in range(numTrials):
        allAccuracies =[]
        rnbAccuracies = []
        ensembleAccuracies = []
        averageAccuracies = []
        simpleAccuracies = []
        randomAccuracies = []
        percentages = np.linspace(0, .9, numPercentages)
        for percentDataMissing in percentages:
            missingTrainingDataAry = deleteRandomFeatureVals(copy.deepcopy(trainingDataAryBase), percentDataMissing)

            rnbAccuracy = getEnsembleAccuracy(numEpochs, 1, learnRate, copy.deepcopy(missingTrainingDataAry), featureAry, testDataAry, hidePrint)
            rnbAccuracies.append(rnbAccuracy)

            ensembleAccuracy = getEnsembleAccuracy(numEpochs, numNets, learnRate, copy.deepcopy(missingTrainingDataAry), featureAry, testDataAry, hidePrint)
            ensembleAccuracies.append(ensembleAccuracy)

            averageAccuracy = getAverageAccuracy(numEpochs, learnRate, copy.deepcopy(missingTrainingDataAry), featureAry, testDataAry, hidePrint)
            averageAccuracies.append(averageAccuracy)

            #simpleAccuracy = getSimpleAccuracy(numEpochs, learnRate, copy.deepcopy(missingTrainingDataAry), featureAry, testDataAry, hidePrint)
            #simpleAccuracies.append(simpleAccuracy)

            randomAccuracy = getRandomAccuracy(numEpochs, learnRate, copy.deepcopy(missingTrainingDataAry), featureAry, testDataAry, hidePrint)
            randomAccuracies.append(randomAccuracy)

            print('percent data missing: ', percentDataMissing)
            print('rnb accuracy = ' + str(rnbAccuracy))
            print('ensemble accuracy = ' + str(ensembleAccuracy))
            print('average accuracy = ' + str(averageAccuracy))
            #print('simple accuracy = ' + str(simpleAccuracy))
            print('random accuracy = ' + str(randomAccuracy))

        rnbAccuraciesOuter.append([rnbAccuracies])
        ensembleAccuraciesOuter.append([ensembleAccuracies])
        averageAccuraciesOuter.append([averageAccuracies])
        simpleAccuraciesOuter.append([simpleAccuracies])
        randomAccuraciesOuter.append([randomAccuracies])
    

    #showPlot(percentages, rnbAccuracies, ensembleAccuracies, averageAccuracies, simpleAccuracies, randomAccuracies)
    showPlotError(percentages, np.array(rnbAccuraciesOuter), np.array(ensembleAccuraciesOuter), np.array(averageAccuraciesOuter), np.array(simpleAccuraciesOuter), np.array(randomAccuraciesOuter))
        
if __name__ == "__main__":
    np.random.seed(0)
    logisticMain(sys.argv)

    