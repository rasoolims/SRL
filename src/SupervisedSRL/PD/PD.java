package SupervisedSRL.PD;

import SentenceStructures.PA;
import SentenceStructures.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import ml.AveragedPerceptron;
import util.IO;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * Created by Maryam Aminian on 5/19/16.
 * Predicate disambiguation modules
 */
public class PD {

    public static int unseenPreds = 0;
    public static int totalPreds = 0;

    public static void main(String[] args) throws Exception {

        String inputFile = args[0];
        String modelDir = args[1];
        String clusterFile = args[2];
        int numOfPDFeatures = 9;

        //read trainJoint and test sentences
        ArrayList<String> sentencesInCONLLFormat = IO.readCoNLLFile(inputFile);

        int totalNumOfSentences = sentencesInCONLLFormat.size();
        int trainSize = (int) Math.floor(0.8 * totalNumOfSentences);

        List<String> train = sentencesInCONLLFormat.subList(0, trainSize);
        List<String> test = sentencesInCONLLFormat.subList(trainSize, totalNumOfSentences);

        //training
        train(train, 10, modelDir, numOfPDFeatures);

        //prediction
        HashMap<Integer, String>[] predictions = new HashMap[test.size()];
        System.out.println("Prediction started...");
        for (int senIdx = 0; senIdx < test.size(); senIdx++) {
            boolean decode = true;
            Sentence sentence = new Sentence(test.get(senIdx));
            predictions[senIdx] = predict(sentence,  modelDir, numOfPDFeatures);
        }

    }


    public static void train(List<String> trainSentencesInCONLLFormat,  int numberOfTrainingIterations, String modelDir, int numOfPDFeaturs)
            throws Exception {
        //creates lexicon of all predicates in the trainJoint set
        HashMap<String, HashMap<String, HashSet<PredicateLexiconEntry>>> trainPLexicon =
                buildPredicateLexicon(trainSentencesInCONLLFormat,  numOfPDFeaturs);

        System.out.println("Training Started...");

        for (String plem : trainPLexicon.keySet()) {
            //extracting feature vector for each training example
            for (String ppos : trainPLexicon.get(plem).keySet()) {
                HashSet<PredicateLexiconEntry> featVectors = trainPLexicon.get(plem).get(ppos);
                HashSet<String> labelSet = getLabels(featVectors);

                AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfPDFeaturs);

                //System.out.print("training model for predicate/pos -->"+ plem+"|"+ppos+"\n");
                for (int i = 0; i < numberOfTrainingIterations; i++) {
                    //System.out.print("iteration:" + i + "...");
                    for (PredicateLexiconEntry ple : trainPLexicon.get(plem).get(ppos)) {
                        //trainJoint average perceptron
                        String plabel = ple.getPlabel();
                        ap.learnInstance(ple.getPdfeats(), plabel);
                    }
                }
                ap.saveModel(modelDir + "/" + plem + "_" + ppos);
            }
        }
        System.out.println("Done!");
    }

    public static HashMap<Integer, String> predict(Sentence sentence, String modelDir, int numOfPDFeatures) throws Exception {
        File f1;
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        String[] sentenceLemmas = sentence.getLemmas();
        String[] sentencePOSTags = sentence.getPosTags();
        String[] sentenceCPOSTags = sentence.getCPosTags();
        String[] sentenceLemmas_str = sentence.getLemmas_str();

        HashMap<Integer, String> predictions = new HashMap<Integer, String>();
        //given gold predicate ids, we just disambiguate them
        for (PA pa : pas) {
            totalPreds++;
            int pIdx = pa.getPredicateIndex();
            String plem = sentenceLemmas[pIdx];
            //we use coarse POS tags instead of original POS tags
            String ppos = sentenceCPOSTags[pIdx]; //sentencePOSTags[pIdx];
            Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures);
            f1 = new File(modelDir + "/" + plem + "_" + ppos);
            if (f1.exists() && !f1.isDirectory()) {
                //seen predicates
                AveragedPerceptron classifier = AveragedPerceptron.loadModel(modelDir + "/" + plem + "_" + ppos);
                String prediction = classifier.predict(pdfeats);
                predictions.put(pIdx, prediction);
            } else {
                //unseen predicate --> assign lemma.01 (default sense) as predicate label instead of null
                unseenPreds++;
                if (!plem.equals("_UNK_"))
                    predictions.put(pIdx, plem + ".01"); //seen pLem
                else
                    predictions.put(pIdx, sentenceLemmas_str[pIdx] + ".01"); //unseen pLem
            }
        }
        return predictions;
    }


    public static HashMap<String, HashMap<String, HashSet<PredicateLexiconEntry>>> buildPredicateLexicon
            (List<String> sentencesInCONLLFormat, int numOfPDFeatures) throws Exception {
        HashMap<String, HashMap<String, HashSet<PredicateLexiconEntry>>> pLexicon = new HashMap<>();

        boolean decode = false;
        for (int senID = 0; senID < sentencesInCONLLFormat.size(); senID++) {
            Sentence sentence = new Sentence(sentencesInCONLLFormat.get(senID));

            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            String[] sentenceLemmas = sentence.getLemmas();
            String[] sentenceCPOSTags = sentence.getCPosTags();

            for (PA pa : pas) {
                int pIdx = pa.getPredicateIndex();
                String plem = sentenceLemmas[pIdx];
                String plabel = pa.getPredicateLabel();
                //instead of original POS tags, we use coarse POS tags
                String ppos = sentenceCPOSTags[pIdx];

                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures);
                PredicateLexiconEntry ple = new PredicateLexiconEntry(plabel, pdfeats);

                if (!pLexicon.containsKey(plem)) {
                    HashMap<String, HashSet<PredicateLexiconEntry>> posDic = new HashMap<>();
                    HashSet<PredicateLexiconEntry> featVectors = new HashSet<PredicateLexiconEntry>();
                    featVectors.add(ple);
                    posDic.put(ppos, featVectors);
                    pLexicon.put(plem, posDic);

                } else if (!pLexicon.get(plem).containsKey(ppos)) {
                    HashSet<PredicateLexiconEntry> featVectors = new HashSet<PredicateLexiconEntry>();
                    featVectors.add(ple);
                    pLexicon.get(plem).put(ppos, featVectors);
                } else {
                    pLexicon.get(plem).get(ppos).add(ple);
                }

            }
        }
        return pLexicon;
    }


    public static HashSet<String> getLabels(HashSet<PredicateLexiconEntry> featVectors) {
        HashSet<String> labelSet = new HashSet<String>();
        for (PredicateLexiconEntry ple : featVectors)
            labelSet.add(ple.getPlabel());

        return labelSet;
    }

}
