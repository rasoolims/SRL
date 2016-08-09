package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;

import java.util.HashMap;


/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {

    public static int numOfAIFeatures = 25 + 154 + 91 + 6; // 154 + 55 + 91 + 91 + 6; //25 + 13;
    public static int numOfACFeatures = 25 + 154 + 91 + 6;// + 55 + 91 + 91 + 6;
    public static int numOfPDFeatures = 9;

    public static void main(String[] args) throws Exception {

        //getting trainJoint/test sentences
        String trainData = args[0];
        String devData = args[1];
        String modelDir = args[2];
        String outputFile = args[3];
        int aiMaxBeamSize = Integer.parseInt(args[4]);
        int acMaxBeamSize = Integer.parseInt(args[5]);
        int numOfTrainingIterations = Integer.parseInt(args[6]);
        boolean decodeJoint = Boolean.parseBoolean(args[7]);
        boolean decodeOnly = Boolean.parseBoolean(args[8]);
        boolean evalOnly = Boolean.parseBoolean(args[9]);

        //single features 25
        //p-p features 55
        //a-a feature 91
        //p-a features 154
        //p-a-a features 91
        //some msc tri-gram feature 6
        //joined features based on original paper (ai) 13
        //joined features based on original paper (ac) 15

        if (evalOnly) {

        } else {
            if (!decodeOnly) {
                Train train = new Train();
                String[] modelPaths = new String[2];
                if (decodeJoint) {
                    //joint decoding
                    modelPaths[0] = train.trainJoint(trainData, devData, numOfTrainingIterations, modelDir, outputFile, numOfACFeatures, numOfPDFeatures, acMaxBeamSize);
                    ModelInfo modelInfo = new ModelInfo(modelPaths[0]);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    AveragedPerceptron classifier = modelInfo.getClassifier();
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, classifier.getLabelMap(),
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile);

                    Evaluation.evaluate(outputFile, devData, indexMap, modelInfo.getClassifier().getReverseLabelMap());

                } else {
                    //stacked decoding
                    modelPaths = train.train(trainData, devData, numOfTrainingIterations, modelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize);
                    ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                    AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[1]);
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            aiModelInfo.getIndexMap(),
                            devData, acClassifier.getLabelMap(),
                            aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }
            } else {
                if (decodeJoint) {
                    //joint decoding
                    ModelInfo modelInfo = new ModelInfo(modelDir + "/joint.model");
                    IndexMap indexMap = modelInfo.getIndexMap();
                    AveragedPerceptron classifier = modelInfo.getClassifier();
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, classifier.getLabelMap(),
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile);

                    Evaluation.evaluate(outputFile, devData, indexMap, classifier.getReverseLabelMap());

                } else {
                    //stacked decoding
                    ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI.model");
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                    AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelDir + "/AC.model");
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            aiModelInfo.getIndexMap(),
                            devData, acClassifier.getLabelMap(),
                            aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }
            }
        }
    }
}
