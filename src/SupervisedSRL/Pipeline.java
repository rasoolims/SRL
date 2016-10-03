package SupervisedSRL;

import SupervisedSRL.Strcutures.Properties;

/**
 * Created by Maryam Aminian on 9/12/16.
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

    public final static int numOfPDFeatures = 289;
    public final static int numOfAIFeatures = 25 + 3 + 5 + 154;
    public final static int numOfACFeatures = 25 + 3 + 5 + 154;
    public final static int numOfGlobalFeatures = 1;

    public static void main(String[] args) {
        String trainFile = args[0];
        String devFile = args[1];
        String clusterFile = args[2];
        String modelDir = args[3];
        String outputDir = args[4];
        String steps = args[5];
        String modelsToBeTrained = args[6];
        int numOfPartitions = Integer.parseInt(args[7]);
        int maxNumOfPDTrainingIterations = Integer.parseInt(args[8]);
        int maxNumOfAITrainingIterations = Integer.parseInt(args[9]);
        int maxNumOfACTrainingIterations = Integer.parseInt(args[10]);
        int maxNumOfRerankerTrainingIterations = Integer.parseInt(args[11]);
        int numOfAIBeamSize = Integer.parseInt(args[12]);
        int numOfACBeamSize = Integer.parseInt(args[13]);
        double aiCoefficient = Double.parseDouble(args[14]);
        boolean reranker = Boolean.parseBoolean(args[15]);

        Properties properties = new Properties(trainFile, devFile, clusterFile, modelDir, outputDir, numOfPartitions,
                maxNumOfPDTrainingIterations,maxNumOfAITrainingIterations,maxNumOfACTrainingIterations, maxNumOfRerankerTrainingIterations,
                numOfAIBeamSize, numOfACBeamSize, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures,
                reranker, steps, modelsToBeTrained, aiCoefficient);
        try {
            Step1.buildIndexMap(properties);
            Step2.buildTrainDataPartitions(properties);
            Step3.trainPDModel(properties);
            Step4.predictPDLabels(properties);
            Step5.trainAIAICModels(properties);
            Step6.buildRerankerFeatureMap(properties);
            Step7.generateRerankerInstances(properties);
            Step8.trainRerankerModel(properties);
            Step9.decode(properties);
            Step10.evaluate(properties);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
