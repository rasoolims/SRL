package SupervisedSRL;

import SupervisedSRL.Strcutures.ClassifierType;
import SupervisedSRL.Strcutures.ModelInfo;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import ml.AveragedPerceptron;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;

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
        int learnerType = Integer.parseInt(args[9]);
        double adamLearningRate = Double.parseDouble(args[10]);
        boolean decodeOnly = Boolean.parseBoolean(args[11]);
        boolean greedy = Boolean.parseBoolean(args[12]);
        int numOfThreads = Integer.parseInt(args[13]);
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
            String[] modelPaths = Train.train(trainData, devData, modelDir, numOfPDFeatures);
        } else {
            //stacked decoding

            FileInputStream fos1 = new FileInputStream(modelDir + "/AI.model");
            GZIPInputStream gz1 = new GZIPInputStream(fos1);
            ObjectInput reader1 = new ObjectInputStream(gz1);
            MLPNetwork aiClassifier = (MLPNetwork) reader1.readObject();
            Options infoptions1 = (Options) reader1.readObject();


            FileInputStream fos2 = new FileInputStream(modelDir + "/AI.model");
            GZIPInputStream gz2 = new GZIPInputStream(fos2);
            ObjectInput reader2 = new ObjectInputStream(gz2);
            MLPNetwork acClassifier = (MLPNetwork) reader2.readObject();
            Options infoptions2 = (Options) reader2.readObject();

            // todo sure not working
            Decoder.decode(new Decoder(aiClassifier, acClassifier),
                    devData, acClassifier.maps.revLabel,
                    aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

            // todo sure not working
            Evaluation.evaluate(outputFile, devData, acClassifier.maps.labelMap);
        }
    }
}
