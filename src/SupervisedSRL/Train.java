package SupervisedSRL;

import Sentence.Argument;
import Sentence.PA;
import Sentence.Sentence;
import SupervisedSRL.Features.BaseFeatures;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {

    //this function is used to train stacked ai-ac models
    public static String[] train(String trainData,
                          String devData,
                          String clusterFile,
                          int numberOfTrainingIterations,
                          String modelDir,
                          int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                          int aiMaxBeamSize, int acMaxBeamSize, int adamBatchSize, double learningRate,
                          ClassifierType classifierType, boolean greedy, int numOfThreads) throws Exception {

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);

        final IndexMap indexMap = new IndexMap(trainData, clusterFile);
        String aiModelPath="";
        String acModelPath="";
        String aiMappingDictsPath ="";
        String acMappingDictsPath="";

        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, Pipeline.numOfPDTrainingIterations, modelDir, numOfPDFeatures);
        if(classifierType == ClassifierType.NN) {
            NNIndexMaps nnIndexMaps = createAIIndicesForNN(trainSentencesInCONLLFormat, indexMap);
            aiModelPath = trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize, greedy);
            acModelPath = trainAC(trainSentencesInCONLLFormat, devData, argLabels, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    aiMaxBeamSize, acMaxBeamSize, greedy);
        } else if (classifierType == ClassifierType.AveragedPerceptron)
        {
            aiModelPath = trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize, greedy);
            acModelPath = trainAC(trainSentencesInCONLLFormat, devData, argLabels, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    aiMaxBeamSize, acMaxBeamSize, greedy);
        }
        return new String[]{aiModelPath, aiMappingDictsPath, acModelPath, acMappingDictsPath};
    }


    public static Pair<HashMap<Object, Integer>[], Pair<HashMap<String, Integer>, Pair<Integer, Integer>>>
    constructFeatureDict4LibLinear(List<String> trainSentencesInCONLLFormat,
                                   IndexMap indexMap, int numOfFeatures, String taskType) throws Exception {
        HashMap<Object, Integer>[] featureDic = new HashMap[numOfFeatures];
        HashSet<Object>[] featuresSeen = new HashSet[numOfFeatures];

        for (int i = 0; i < numOfFeatures; i++) {
            featureDic[i] = new HashMap<Object, Integer>();
            featuresSeen[i] = new HashSet<Object>();
        }
        HashMap<String, Integer> labelDic = new HashMap<String, Integer>();

        System.out.print("Extracting mapping dictionary...");
        DecimalFormat format = new DecimalFormat("##.00");
        long startTime = System.currentTimeMillis();
        int numOfTrainInstances = 0;
        for (String sentence : trainSentencesInCONLLFormat) {
            Object[] instances = null;
            if (taskType.equals("AI")) instances = obtainTrainInstance4AI(sentence, indexMap, numOfFeatures);
            else if (taskType.equals("AC")) instances = obtainTrainInstance4AC(sentence, indexMap, numOfFeatures);
            else if (taskType.equalsIgnoreCase("JOINT"))
                instances = obtainTrainInstance4JointModel(sentence, indexMap, numOfFeatures);
            else if (taskType.equals("PD")) throw new Exception("task not supported");

            ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0]; //in the format averaged perceptron supports
            ArrayList<String> labels = (ArrayList<String>) instances[1];

            numOfTrainInstances += labels.size();
            //getting set of all possible values for each slot
            for (int instance = 0; instance < labels.size(); instance++) {
                for (int dim = 0; dim < numOfFeatures; dim++) {
                    featuresSeen[dim].add(featVectors.get(instance)[dim]);
                }
                if (!labelDic.containsKey(labels.get(instance)))
                    labelDic.put(labels.get(instance), labelDic.size());
            }
        }
        //constructing featureDic
        int featureIndex = 1;
        //for each feature slot
        for (int i = 0; i < numOfFeatures; i++) {
            //adding seen feature indices
            for (Object feat : featuresSeen[i]) {
                featureDic[i].put(feat, featureIndex++);
            }
            //unseen feature index
            featureDic[i].put(Pipeline.unseenSymbol, featureIndex++);
            assert !featuresSeen[i].contains(Pipeline.unseenSymbol);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for extraction" + format.format( ((endTime - startTime)/1000.0)/ 60.0));
        System.out.println("Done!");
        return new Pair<HashMap<Object, Integer>[], Pair<HashMap<String, Integer>, Pair<Integer, Integer>>>(featureDic,
                new Pair<HashMap<String, Integer>, Pair<Integer, Integer>>(labelDic, new Pair<Integer, Integer>(featureIndex, numOfTrainInstances)));
    }


    public static NNIndexMaps createAIIndicesForNN(List<String> trainSentencesInCONLLFormat,  IndexMap indexMap) throws Exception {
        NNIndexMaps nnIndexMaps = new NNIndexMaps();
        for (String sentenceInCONLLFormat : trainSentencesInCONLLFormat) {
            Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            int[] sentenceWords = sentence.getWords();

            for (PA pa : pas) {
                int pIdx = pa.getPredicateIndex();
                ArrayList<Argument> currentArgs = pa.getArguments();

                for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                 BaseFeatures baseFeatures = new BaseFeatures(pIdx, wordIdx, sentence, indexMap);
                 nnIndexMaps.addToMap(baseFeatures);
                }
            }
        }
        return nnIndexMaps;
    }

    public static String trainAI(List<String> trainSentencesInCONLLFormat,
                           List<String> devSentencesInCONLLFormat,
                           IndexMap indexMap,
                           int numberOfTrainingIterations,
                           String modelDir, int numOfFeatures, int numOfPDFeatures, int aiMaxBeamSize, boolean greedy)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime = 0;
        double bestFScore = 0;
        String modelPath = modelDir + "/AI.model";
        int noImprovement = 0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            ap.correct = 0;
            for (String sentence : trainSentencesInCONLLFormat) {

                Object[] instances = obtainTrainInstance4AI(sentence, indexMap, numOfFeatures);
                ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];

                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                    if (labels.get(d).equals("0"))
                        negInstances++;
                    dataSize++;
                }
                s++;
                if (s % 1000 == 0)
                    System.out.print(s + "...");
            }

            System.out.print(s + "\n");

            double ac = 100. * (double) ap.correct / dataSize;

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++)
                    System.out.print(ap.confusionMatrix[i][j] + "\t");
                System.out.println("");
            }
            ap.confusionMatrix = new int[2][2];

            System.out.println("data size:" + dataSize + " neg_instances: " + negInstances + " accuracy: " + ac);
            endTime = System.currentTimeMillis();
            System.out.println("Total time for this iteration " + format.format(((endTime - startTime) / 1000.0) / 60.0));

            //making prediction over dev sentences
            System.out.println("****** DEV RESULTS ******");
            //instead of loading model from file, we just calculate the average weights
            Decoder argumentDecoder = new Decoder(ap.calculateAvgWeights(), "AI");
            boolean decode = true;

            //ai confusion matrix
            int[][] aiConfusionMatrix = new int[2][2];
            aiConfusionMatrix[0][0] = 0;
            aiConfusionMatrix[0][1] = 0;
            aiConfusionMatrix[1][0] = 0;
            aiConfusionMatrix[1][1] = 0;


            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap);
                HashMap<Integer, Prediction> prediction = argumentDecoder.predictAI(sentence, indexMap, aiMaxBeamSize,
                        numOfFeatures, modelDir, numOfPDFeatures, null, ClassifierType.AveragedPerceptron, greedy);

                //we do evaluation for each sentence and update confusion matrix right here
                aiConfusionMatrix = Evaluation.evaluateAI4ThisSentence(sentence, prediction, aiConfusionMatrix);
            }
            double f1 = Evaluation.computePrecisionRecall(aiConfusionMatrix);
            if (f1 > bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving the new model...");
                ModelInfo.saveModel(ap, indexMap, modelPath);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 5) {
                    System.out.print("\nEarly stopping...");
                    break;
                }
            }
        }

        return modelPath;
    }


    public static String trainAC(List<String> trainSentencesInCONLLFormat,
                           String devData,
                           HashSet<String> labelSet, IndexMap indexMap,
                           int numberOfTrainingIterations,
                           String modelDir, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                           int aiMaxBeamSize, int acMaxBeamSize, boolean greedy)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        //building trainJoint instances
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfACFeatures);

        //training average perceptron
        long startTime = 0;
        long endTime = 0;
        double bestFScore = 0;
        int noImprovement = 0;
        String modelPath = modelDir + "/AC.model";
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int dataSize = 0;
            int s = 0;
            for (String sentence : trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4AC(sentence, indexMap, numOfACFeatures);
                s++;
                ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];
                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                    dataSize++;
                }
                if (s % 1000 == 0)
                    System.out.print(s + "...");
            }
            System.out.print(s + "\n");
            double ac = 100. * (double) ap.correct / dataSize;
            System.out.println("accuracy: " + ac);
            ap.correct = 0;
            endTime = System.currentTimeMillis();
            System.out.println("Total time of this iteration: " + format.format(((endTime - startTime) / 1000.0) / 60.0));

            System.out.println("****** DEV RESULTS ******");
            //instead of loading model from file, we just calculate the average weights
            String aiModelPath = modelDir + "/AI.model";
            String outputFile = modelDir + "/dev_output_" + iter;

            Decoder argumentDecoder = new Decoder(AveragedPerceptron.loadModel(aiModelPath), ap.calculateAvgWeights());
            Decoder.decode(argumentDecoder, indexMap, devData, ap.getLabelMap(),
                    aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

            HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(ap.getReverseLabelMap());
            reverseLabelMap.put("0", reverseLabelMap.size());

            double f1 = Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
            if (f1 > bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving final model...");
                ModelInfo.saveModel(ap, modelPath);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 5) {
                    System.out.print("\nEarly stopping...");
                    break;
                }
            }
        }
        return modelPath;
    }


    public static Object[] obtainTrainInstance4AI(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx,
                        sentence, numOfFeatures, indexMap, false, 0);

                String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }


    public static Object[] obtainTrainInstance4AC(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();
            //extract features for arguments (not all words)
            for (Argument arg : currentArgs) {
                int argIdx = arg.getIndex();
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, argIdx, sentence, numOfFeatures, indexMap, false, 0);

                String label = arg.getType();
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }


    //function is used for joint ai-ac training/decoding
    public static Object[] obtainTrainInstance4JointModel(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                Object[] featVector = FeatureExtractor.extractJointFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap, false, 0);

                String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : isArgument(wordIdx, currentArgs);
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }


    /////////////////////////////////////////////////////////////////////////////
    //////////////////////////////  SUPPORT FUNCTIONS  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////


    public static String isArgument(int wordIdx, ArrayList<Argument> currentArgs) {
        for (Argument arg : currentArgs)
            if (arg.getIndex() == wordIdx)
                return arg.getType();
        return "";
    }

    private List<Integer> obtainSampleIndices(ArrayList<String> labels) {
        ArrayList<Integer> posIndices = new ArrayList<Integer>();
        ArrayList<Integer> negIndices = new ArrayList<Integer>();

        for (int k = 0; k < labels.size(); k++) {
            if (labels.get(k).equals("1"))
                posIndices.add(k);
            else
                negIndices.add(k);
        }

        Collections.shuffle(posIndices);
        Collections.shuffle(negIndices);

        //int sampleSize= Math.min(posIndices.size(), negIndices.size());
        int posSampleSize = posIndices.size();
        int negSampleSize = negIndices.size() / 8;
        List<Integer> sampledPosIndices = posIndices.subList(0, posSampleSize);
        List<Integer> sampledNegIndices = negIndices.subList(0, negSampleSize);

        sampledPosIndices.addAll(sampledNegIndices);
        return sampledPosIndices;
    }

    private Object[] sample(ArrayList<List<String>> featVectors,
                            ArrayList<String> labels,
                            List<Integer> sampleIndices) {
        ArrayList<List<String>> sampledFeatVectors = new ArrayList<List<String>>();
        ArrayList<String> sampledLabels = new ArrayList<String>();

        Collections.shuffle(sampleIndices);
        for (int idx : sampleIndices) {
            sampledFeatVectors.add(featVectors.get(idx));
            sampledLabels.add(labels.get(idx));
        }

        return new Object[]{sampledFeatVectors, sampledLabels};
    }


    private Object[] overSample(ArrayList<Object[]> senFeatVecs, ArrayList<String> senLabels) {
        ArrayList<Integer> negIndices = new ArrayList<Integer>();
        ArrayList<Integer> posIndices = new ArrayList<Integer>();

        ArrayList<Object[]> overSampledFeatVecs = senFeatVecs;
        ArrayList<String> overSampledLabels = senLabels;

        for (int idx = 0; idx < senLabels.size(); idx++) {
            if (senLabels.get(idx).equals("0"))
                negIndices.add(idx);
            else
                posIndices.add(idx);
        }
        int numOfSamples = negIndices.size() - posIndices.size();
        for (int k = 0; k < numOfSamples; k++) {
            if (posIndices.size() != 0) {
                int ranIdx = new Random().nextInt(posIndices.size());
                overSampledFeatVecs.add(senFeatVecs.get(posIndices.get(ranIdx)));
                overSampledLabels.add(senLabels.get(posIndices.get(ranIdx)));
            }
        }
        return new Object[]{overSampledFeatVecs, overSampledLabels};
    }


    private Object[] downSample(ArrayList<Object[]> senFeatVecs, ArrayList<String> senLabels) {
        ArrayList<Integer> negIndices = new ArrayList<Integer>();
        ArrayList<Integer> posIndices = new ArrayList<Integer>();

        ArrayList<Object[]> downSampledFeatVecs = new ArrayList<Object[]>();
        ArrayList<String> downSampledLabels = new ArrayList<String>();

        for (int idx = 0; idx < senLabels.size(); idx++) {
            if (senLabels.get(idx).equals("0"))
                negIndices.add(idx);
            else
                posIndices.add(idx);
        }
        int numOfSamples = posIndices.size();
        ArrayList<Integer> downSampledIndices = new ArrayList<Integer>();
        //adding down sampled neg indices
        for (int k = 0; k < numOfSamples; k++) {
            if (negIndices.size() != 0) {
                int ranIdx = new Random().nextInt(negIndices.size());
                downSampledIndices.add(negIndices.get(ranIdx));
            }
        }
        //adding all pos indices
        downSampledIndices.addAll(posIndices);
        Collections.shuffle(downSampledIndices);

        for (int idx : downSampledIndices) {
            downSampledFeatVecs.add(senFeatVecs.get(idx));
            downSampledLabels.add(senLabels.get(idx));
        }

        return new Object[]{downSampledFeatVecs, downSampledLabels};
    }
}
