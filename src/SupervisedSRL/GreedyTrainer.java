/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package SupervisedSRL;

import SupervisedSRL.Strcutures.NNIndexMaps;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPTrainer;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.AveragingOption;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class GreedyTrainer {
    Options options;
    Random random;
    NNIndexMaps maps;

    public GreedyTrainer(Options options, NNIndexMaps maps) {
        this.options = options;
        random = new Random();
        this.maps = maps;
    }

    public static void trainWithNN(Options options, NNIndexMaps maps, int numOutputs,
                                   ArrayList<String> trainSentencesInCONLLFormat,
                                   ArrayList<String> devSentencesInCONLLFormat) throws Exception {
        if (options.trainingOptions.trainFile.equals("") || options.generalProperties.modelDir.equals("")) {
            Options.showHelp();
        } else {
            if (options.trainingOptions.pretrainLayers && options.networkProperties.hiddenLayer2Size != 0) {
                trainMultiLayerNetwork(options, trainSentencesInCONLLFormat, devSentencesInCONLLFormat,maps, numOutputs);
            } else {
                train(options, trainSentencesInCONLLFormat, devSentencesInCONLLFormat,maps, numOutputs);
            }
        }
    }

    private static void trainMultiLayerNetwork(Options options,ArrayList<String> trainSentencesInCONLLFormat,
                                               ArrayList<String> devSentencesInCONLLFormat, NNIndexMaps maps, int numOutputs) throws Exception {
        Options oneLayerOption = options.clone();
        oneLayerOption.networkProperties.hiddenLayer2Size = 0;
        oneLayerOption.generalProperties.beamWidth = 1;
        oneLayerOption.trainingOptions.trainingIter = options.trainingOptions.preTrainingIter;
        String modelFile = options.trainingOptions.preTrainedModelPath.equals("") ? options.generalProperties.modelDir : options.trainingOptions
                .preTrainedModelPath;

        if (options.trainingOptions.preTrainedModelPath.equals("")) {
            System.out.println("First training with one hidden layer!");
            train(oneLayerOption,trainSentencesInCONLLFormat, devSentencesInCONLLFormat, maps, numOutputs);
        }

        System.out.println("Loading model with one hidden layer!");
        FileInputStream fos = new FileInputStream(modelFile);
        GZIPInputStream gz = new GZIPInputStream(fos);
        ObjectInput reader = new ObjectInputStream(gz);
        MLPNetwork mlpNetwork = (MLPNetwork) reader.readObject();
        reader.close();

        System.out.println("Now Training with two layers!");
        Options twoLayerOptions = options.clone();
        twoLayerOptions.generalProperties.beamWidth = 1;
        MLPNetwork net = constructMlpNetwork(twoLayerOptions, maps, numOutputs);
        // Putting the first layer into it!
        net.layer(0).setLayer(mlpNetwork.layer(0));
        trainNetwork(twoLayerOptions, maps, trainSentencesInCONLLFormat, devSentencesInCONLLFormat, net);
    }

    private static void train(Options options,ArrayList<String> trainSentencesInCONLLFormat,
                              ArrayList<String> devSentencesInCONLLFormat,
                              NNIndexMaps maps, int numOutputs) throws Exception {
        Options greedyOptions = options.clone();
        greedyOptions.generalProperties.beamWidth = 1;
        MLPNetwork mlpNetwork = constructMlpNetwork(greedyOptions, maps, numOutputs);
        trainNetwork(greedyOptions, maps, trainSentencesInCONLLFormat, devSentencesInCONLLFormat,mlpNetwork);
    }

    private static void trainNetwork(Options options, NNIndexMaps maps, ArrayList<String> trainSentencesInCONLLFormat,
                                     ArrayList<String> devSentencesInCONLLFormat, MLPNetwork mlpNetwork) throws Exception {
        MLPNetwork avgMlpNetwork = mlpNetwork.clone(true, true);

        GreedyTrainer trainer = new GreedyTrainer(options, maps);

        ArrayList<String> dataSet =trainSentencesInCONLLFormat;

        boolean binary = mlpNetwork.getNumOutputs() ==2 ? true: false;
        ArrayList<NeuralTrainingInstance> allInstances = Train.getNextInstances(dataSet, 0, dataSet.size(), mlpNetwork.maps, binary);
        mlpNetwork.maps.constructPreComputeMap(allInstances, mlpNetwork.getNumWordLayers(), 10000);
        mlpNetwork.resetPreComputeMap();
        avgMlpNetwork.resetPreComputeMap();
        mlpNetwork.maps.emptyEmbeddings();

        MLPTrainer neuralTrainer = new MLPTrainer(mlpNetwork, options);

        double bestModelUAS = 0;
        Random random = new Random();
        System.out.println("Data has " + allInstances.size() + " instances");
        System.out.println("Decay after every " + options.trainingOptions.decayStep + " batches");
        int step;
        for (step = 0; step < options.trainingOptions.trainingIter; step++) {
            List<NeuralTrainingInstance> instances = Utils.getRandomSubset(allInstances, random, options.networkProperties.batchSize);
            try {
                neuralTrainer.fit(instances, step, step % (Math.max(1, options.trainingOptions.UASEvalPerStep / 10)) == 0);
            } catch (Exception ex) {
                System.err.println("Exception occurred: " + ex.getMessage());
                ex.printStackTrace();
                System.exit(1);
            }
            if (options.updaterProperties.updaterType == UpdaterType.SGD) {
                if ((step + 1) % options.trainingOptions.decayStep == 0) {
                    neuralTrainer.setLearningRate(0.96 * neuralTrainer.getLearningRate());
                    System.out.println("The new learning rate: " + neuralTrainer.getLearningRate());
                }
            }

            if (options.trainingOptions.averagingOption != AveragingOption.NO) {
                // averaging
                double ratio = Math.min(0.9999, (double) step / (9 + step));
                mlpNetwork.averageNetworks(avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);
            }

            if (step % options.trainingOptions.UASEvalPerStep == 0) {
                // todo evaluate and save if needed
            }
        }

        if (options.trainingOptions.averagingOption != AveragingOption.NO) {
            // averaging
            double ratio = Math.min(0.9999, (double) step / (9 + step));
            mlpNetwork.averageNetworks(avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);
        }

        if (options.trainingOptions.averagingOption != AveragingOption.ONLY) {
            //todo evaluate and save if needed
        }
        if (options.trainingOptions.averagingOption != AveragingOption.NO) {
            avgMlpNetwork.preCompute();
            // todo evaluate and save if needed
        }
        neuralTrainer.shutDownLiveThreads();
    }

    private static MLPNetwork constructMlpNetwork(Options options, NNIndexMaps maps, int numOutputs) throws Exception {
        int wDim = options.networkProperties.wDim;
        if (options.trainingOptions.wordEmbeddingFile.length() > 0)
            wDim = maps.readEmbeddings(options.trainingOptions.wordEmbeddingFile);

        System.out.println("Embedding dimension " + wDim);
        return new MLPNetwork(maps, options, wDim, options.networkProperties.posDim,
                options.networkProperties.depDim,options.networkProperties.subcatDim, options.networkProperties.depPathDim,
                options.networkProperties.posPathDim, options.networkProperties.positionDim, numOutputs);
    }

}