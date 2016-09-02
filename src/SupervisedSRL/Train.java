package SupervisedSRL;

import SentenceStructures.Argument;
import SentenceStructures.PA;
import SentenceStructures.Sentence;
import SupervisedSRL.Features.BaseFeatures;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import ml.AveragedPerceptron;
import util.IO;

import java.awt.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {
    public static ArrayList<NeuralTrainingInstance> getNextInstances(ArrayList<String> trainData, int start, int end, NNIndexMaps maps,
                                                                     boolean binary)
            throws Exception {
        ArrayList<NeuralTrainingInstance> instances = new ArrayList<>();
        for (int i = start; i < end; i++) {
            Sentence sentence = new Sentence(trainData.get(i));
            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            String[] sentenceWords = sentence.getWords();

            for (PA pa : pas) {
                int pIdx = pa.getPredicateIndex();
                ArrayList<Argument> currentArgs = pa.getArguments();

                for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                    String arg = isArgument(wordIdx, currentArgs);
                    int isArg = (arg.equals("")) ? 0 : 1;

                    if (binary) {
                        double[] label = new double[2];
                        label[isArg] = 1;
                        BaseFeatures baseFeatures = new BaseFeatures(pIdx, wordIdx, sentence);
                        NeuralTrainingInstance instance = new NeuralTrainingInstance(maps.features(baseFeatures), label);
                        instances.add(instance);
                    } else if (isArg == 1) {
                        double[] label = new double[maps.labelMap.size()];
                        label[maps.labelMap.get(arg)] = 1;
                        BaseFeatures baseFeatures = new BaseFeatures(pIdx, wordIdx, sentence);
                        NeuralTrainingInstance instance = new NeuralTrainingInstance(maps.features(baseFeatures), label);
                        instances.add(instance);
                    }
                }
            }
        }
        return instances;
    }

    public static NNIndexMaps createIndicesForNN(List<String> trainSentencesInCONLLFormat) throws Exception {
        NNIndexMaps nnIndexMaps = new NNIndexMaps();
        for (String sentenceInCONLLFormat : trainSentencesInCONLLFormat) {
            Sentence sentence = new Sentence(sentenceInCONLLFormat);
            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            String[] sentenceWords = sentence.getWords();

            for (PA pa : pas) {
                int pIdx = pa.getPredicateIndex();
                ArrayList<Argument> currentArgs = pa.getArguments();

                for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                    BaseFeatures baseFeatures = new BaseFeatures(pIdx, wordIdx, sentence);
                    String arg = isArgument(wordIdx, currentArgs);
                    nnIndexMaps.addToMap(baseFeatures);
                    nnIndexMaps.addLabel(arg);
                }
            }
        }
        return nnIndexMaps;
    }

    //this function is used to train stacked ai-ac models
    public static String[] train(Options options, String trainData,
                                 String devData,
                                 String modelDir,
                                 int numOfPDFeatures) throws Exception {

        ArrayList<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        ArrayList<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        String aiModelPath = "";
        String acModelPath = "";
        String aiMappingDictsPath = "";
        String acMappingDictsPath = "";

        //training PD module
        PD.train(trainSentencesInCONLLFormat, Pipeline.numOfPDTrainingIterations, modelDir, numOfPDFeatures);
        NNIndexMaps nnIndexMaps = createIndicesForNN(trainSentencesInCONLLFormat);

        aiModelPath = trainAI(options, trainSentencesInCONLLFormat, devSentencesInCONLLFormat, nnIndexMaps, modelDir);
        acModelPath = trainAC(options, trainSentencesInCONLLFormat, devSentencesInCONLLFormat, nnIndexMaps, modelDir);
        return new String[]{aiModelPath, aiMappingDictsPath, acModelPath, acMappingDictsPath};
    }

    public static String trainAI(Options options, ArrayList<String> trainSentencesInCONLLFormat,
                                 ArrayList<String> devSentencesInCONLLFormat,
                                 NNIndexMaps maps, String modelDir)
            throws Exception {
        GreedyTrainer.trainWithNN(options, maps, 2, trainSentencesInCONLLFormat, devSentencesInCONLLFormat);
        String modelPath = modelDir + "/AI.model";
        return modelPath;
    }


    public static String trainAC(Options options, ArrayList<String> trainSentencesInCONLLFormat,
                                 ArrayList<String> devSentencesInCONLLFormat,
                                 NNIndexMaps maps, String modelDir)
            throws Exception {

        GreedyTrainer.trainWithNN(options, maps, maps.labelMap.size(), trainSentencesInCONLLFormat, devSentencesInCONLLFormat);
        String modelPath = modelDir + "/AC.model";
        return modelPath;
    }

    public static String isArgument(int wordIdx, ArrayList<Argument> currentArgs) {
        for (Argument arg : currentArgs)
            if (arg.getIndex() == wordIdx)
                return arg.getType();
        return "";
    }

}
