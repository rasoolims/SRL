package SupervisedSRL;

import SupervisedSRL.Strcutures.ClassifierType;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;

import java.util.HashMap;

/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {

    //single features 25 + 3 (predicate cluster features) + 5(argument cluster features)
    //p-p features 55
    //a-a feature 91
    //p-a features 154
    //p-a-a features 91
    //some msc tri-gram feature 6
    //joined features based on original paper (ai) 13
    //joined features based on original paper (ac) 15
    //predicate cluster features 3
    //argument cluster features 5

    public static int numOfAIFeatures = 25 + 3 + 5 + 154 + 91 + 6;
    public static int numOfACFeatures = 25 + 3 + 5 + 154 + 91 + 6;
    public static int numOfPDFeatures = 9;
    public static int numOfPDTrainingIterations = 10;
    public static String unseenSymbol = ";;?;;";


    public static void main(String[] args) throws Exception {
        //getting trainJoint/test sentences
        String trainData = args[0];
        String devData = args[1];
        String clusterFile = args[2];
        String modelDir = args[3];
        String outputFile = args[4];
        int aiMaxBeamSize = Integer.parseInt(args[5]);
        int acMaxBeamSize = Integer.parseInt(args[6]);
        int numOfTrainingIterations = Integer.parseInt(args[7]);
        int adamBatchSize = Integer.parseInt(args[8]);
        int learnerType = Integer.parseInt(args[9]); //1: ap 2: ll 3:adam
        double adamLearningRate = Double.parseDouble(args[10]);
        boolean decodeOnly = Boolean.parseBoolean(args[12]);
        boolean greedy = Boolean.parseBoolean(args[13]);
        int numOfThreads = Integer.parseInt(args[14]);
        ClassifierType classifierType = ClassifierType.AveragedPerceptron;
        switch (learnerType) {
            case (0):
                classifierType = ClassifierType.NN;
                break;
            case (1):
                classifierType = ClassifierType.AveragedPerceptron;
                break;
        }

        if (!decodeOnly) {
            String[] modelPaths = new String[4];

            //stacked decoding
            if (classifierType == ClassifierType.NN) {
                modelPaths = Train.train(trainData, devData, clusterFile, numOfTrainingIterations, modelDir,
                        numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize, adamBatchSize, adamLearningRate,
                        ClassifierType.NN, greedy, numOfThreads);

            } else if (classifierType == ClassifierType.AveragedPerceptron) {
                modelPaths = Train.train(trainData, devData, clusterFile, numOfTrainingIterations, modelDir,
                        numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize, adamBatchSize, adamLearningRate,
                        ClassifierType.AveragedPerceptron, greedy, numOfThreads);

                ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);
                IndexMap indexMap = aiModelInfo.getIndexMap();
                AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[2]);
                Decoder.decode(new Decoder(aiClassifier, acClassifier),
                        aiModelInfo.getIndexMap(),
                        devData, acClassifier.getLabelMap(),
                        aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                        modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

                HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                reverseLabelMap.put("0", reverseLabelMap.size());
                Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
            }
        } else {
            //stacked decoding
            if (classifierType == ClassifierType.AveragedPerceptron) {
                ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI.model");
                IndexMap indexMap = aiModelInfo.getIndexMap();
                AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelDir + "/AC.model");
                Decoder.decode(new Decoder(aiClassifier, acClassifier),
                        aiModelInfo.getIndexMap(),
                        devData, acClassifier.getLabelMap(),
                        aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                        modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

                HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                reverseLabelMap.put("0", reverseLabelMap.size());
                Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
            }
        }
    }
}
