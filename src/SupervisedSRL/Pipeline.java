package SupervisedSRL;

import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.Random;
import java.util.zip.GZIPInputStream;

/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {
    public static int numOfPDFeatures = 9;
    public static int numOfPDTrainingIterations = 10;

    public static void main(String[] args) throws Exception {
        //getting trainJoint/test sentences
       Options options =  Options.processArgs(args);

        if (options.generalProperties.train) {
            String[] modelPaths = Train.train(new Random(),options, options.trainingOptions.trainFile, options.trainingOptions.devPath,
                    options.generalProperties.modelDir, numOfPDFeatures);
        } else if(options.generalProperties.parseConllFile){
            //stacked decoding
            FileInputStream fos1 = new FileInputStream(options.generalProperties.modelDir + "/AI.model");
            GZIPInputStream gz1 = new GZIPInputStream(fos1);
            ObjectInput reader1 = new ObjectInputStream(gz1);
            MLPNetwork aiClassifier = (MLPNetwork) reader1.readObject();

            FileInputStream fos2 = new FileInputStream(options.generalProperties.modelDir + "/AC.model");
            GZIPInputStream gz2 = new GZIPInputStream(fos2);
            ObjectInput reader2 = new ObjectInputStream(gz2);
            MLPNetwork acClassifier = (MLPNetwork) reader2.readObject();

            // todo sure not working
            Decoder.decode(new Decoder(aiClassifier, acClassifier),
                    options.generalProperties.inputFile, acClassifier.maps.revLabel,
                    options.generalProperties.beamWidth, options.generalProperties.beamWidth, numOfPDFeatures,
                    options.generalProperties.modelDir, options.generalProperties.outputFile,
                    options.generalProperties.beamWidth == 1 ? true : false, acClassifier.maps);

            // todo sure not working
            Evaluation.evaluate(options.generalProperties.outputFile, options.generalProperties.inputFile, acClassifier.maps.labelMap);
        } else{
            Options.showHelp();
        }
    }
}
