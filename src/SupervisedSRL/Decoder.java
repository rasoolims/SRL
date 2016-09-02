package SupervisedSRL;

import SentenceStructures.Sentence;
import SupervisedSRL.Features.BaseFeatures;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import util.IO;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class Decoder {

    MLPNetwork aiClassifier; //argument identification (binary classifier)
    MLPNetwork acClassifier; //argument classification (multi-class classifier)

    public Decoder(MLPNetwork classifier, String state) {
        if (state.equals("AI")) {
            this.aiClassifier = classifier;
        } else if (state.equals("AC") || state.equals("joint")) {
            this.acClassifier = classifier;
        }
    }

    public Decoder(MLPNetwork aiClassifier, MLPNetwork acClassifier) {
        this.aiClassifier = aiClassifier;
        this.acClassifier = acClassifier;
    }

    //stacked decoding
    public static void decode(Decoder decoder, String devDataPath, ArrayList<String> labelMap,
                              int aiMaxBeamSize, int acMaxBeamSize,
                              int numOfPDFeatures, String modelDir, String outputFile, boolean greedy, NNIndexMaps maps) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        boolean decode = true;
        ArrayList<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devDataPath);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence);

            predictions[d] = (TreeMap<Integer, Prediction>) decoder.predict(sentence, aiMaxBeamSize, acMaxBeamSize,
                    numOfPDFeatures, modelDir, greedy, maps);

            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));
        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }


    public HashMap<Integer, Prediction> predictAI(Sentence sentence, int aiMaxBeamSize,
                                                  int numOfFeatures, String modelDir, int numOfPDFeatures,
                                                  HashMap<Object, Integer>[] featDict,
                                                  ClassifierType classifierType, boolean greedy)
            throws Exception {
        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, modelDir, numOfPDFeatures);
        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();
            ArrayList<Integer> aiCandidatesGreedy= new ArrayList<Integer>();
            HashMap<Integer, String> highestScorePrediction = new HashMap<>();

            if (!greedy)
            {
                aiCandidates = getBestAICandidates(sentence, pIdx, aiMaxBeamSize);
                highestScorePrediction= getHighestScorePredication(aiCandidates);

            }else {
                aiCandidatesGreedy = getBestAICandidatesGreedy(sentence, pIdx);
                for (int idx=0; idx< aiCandidatesGreedy.size(); idx++) {
                    int wordIdx = aiCandidatesGreedy.get(idx);
                    highestScorePrediction.put(wordIdx, "1");
                }
            }

            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }



    public Object predict(Sentence sentence, int aiMaxBeamSize,
                          int acMaxBeamSize,
                          int numOfPDFeatures, String modelDir,
                          boolean greedy,
                          NNIndexMaps maps) throws Exception {

        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, modelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();
        TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates = new TreeMap<Integer, Prediction4Reranker>();
        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = new ArrayList();

            ArrayList<Integer> aiCandidatesGreedy = new ArrayList<Integer>();
            ArrayList<Integer> acCandidatesGreedy = new ArrayList<Integer>();
            HashMap<Integer, String> highestScorePrediction = new HashMap<>();


            if (!greedy) {
                aiCandidates = getBestAICandidates(sentence, pIdx, aiMaxBeamSize);
                acCandidates = getBestACCandidates(sentence, pIdx, aiCandidates, acMaxBeamSize);
                highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates, maps);
            } else {
                aiCandidatesGreedy = getBestAICandidatesGreedy(sentence, pIdx);
                acCandidatesGreedy = getBestACCandidatesGreedy(sentence, pIdx, aiCandidatesGreedy);

                for (int idx = 0; idx < acCandidatesGreedy.size(); idx++) {
                    int wordIdx = aiCandidatesGreedy.get(idx);
                    String label = maps.revLabel.get(acCandidatesGreedy.get(idx));
                    highestScorePrediction.put(wordIdx, label);
                }
            }
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }

        return predictedPAs;
    }

    ////////////////////////////////// GET BEST CANDIDATES ///////////////////////////////////////////////
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, int pIdx,  int maxBeamSize) throws Exception {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        String[] sentenceWords = sentence.getWords();
        double[] labels = new double[2];
        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            double[] featVector = aiClassifier.maps.features(new BaseFeatures(pIdx, wordIdx, sentence),0);
            double score0 = Double.POSITIVE_INFINITY;
            double score1 = Double.NEGATIVE_INFINITY;

            double[] scores = aiClassifier.output(featVector, labels);
            score0 = scores[0];
            score1 = scores[1];

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;
                BeamElement be0 = new BeamElement(index, currentScore + score0, 0);
                BeamElement be1 = new BeamElement(index, currentScore + score1, 1);

                newBeamHeap.add(be0);
                if (newBeamHeap.size() > maxBeamSize)
                    newBeamHeap.pollFirst();

                newBeamHeap.add(be1);
                if (newBeamHeap.size() > maxBeamSize)
                    newBeamHeap.pollFirst();
            }

            ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);
            for (BeamElement beamElement : newBeamHeap) {
                ArrayList<Integer> newArrayList = new ArrayList<Integer>();
                for (int b : currentBeam.get(beamElement.index).second)
                    newArrayList.add(b);
                if (beamElement.label == 1)
                    newArrayList.add(wordIdx);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }

            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }

    //getting highest score AI candidate (AP/LL/Adam) without Beam Search
    private ArrayList<Integer> getBestAICandidatesGreedy
            (Sentence sentence, int pIdx) throws Exception {
        String[] sentenceWords = sentence.getWords();
        double[] labels = new double[2];
        ArrayList<Integer> aiCandids = new ArrayList<>();
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            double[] featVector = aiClassifier.maps.features(new BaseFeatures(pIdx, wordIdx, sentence),0);
            double[] scores = aiClassifier.output(featVector, labels);
            if (scores[0] > scores[1])
                aiCandids.add(wordIdx);
        }
        return aiCandids;
    }


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, int pIdx, 
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates, int maxBeamSize) throws Exception {

        int numOfLabels = acClassifier.getNumOutputs();
        ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> finalACCandidates = new ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>>();
        for (Pair<Double, ArrayList<Integer>> aiCandidate : aiCandidates) {
            //for each AI candidate generated by aiClassifier
            double aiScore = aiCandidate.first;
            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(aiScore, new ArrayList<Integer>()));

            // Gradual building of the beam for the words identified as an argument by AI classifier
            for (int wordIdx : aiCandidate.second) {
                // retrieve candidates for the current word
                double[] featVector = acClassifier.maps.features(new BaseFeatures(pIdx, wordIdx, sentence), 0);
                double[] labelScores = new double[numOfLabels];
                labelScores = acClassifier.output(featVector, labelScores);

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    for (int labelIdx = 0; labelIdx < numOfLabels; labelIdx++) {
                        newBeamHeap.add(new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx));
                        if (newBeamHeap.size() > maxBeamSize)
                            newBeamHeap.pollFirst();
                    }
                }

                ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);

                for (BeamElement beamElement : newBeamHeap) {
                    ArrayList<Integer> newArrayList = new ArrayList<Integer>();
                    for(int b:currentBeam.get(beamElement.index).second)
                        newArrayList.add(b);
                    newArrayList.add(beamElement.label);
                    newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
                }

                // replace the old beam with the intermediate beam
                currentBeam = newBeam;
            }

            //current beam for this ai candidates is built
            finalACCandidates.add(currentBeam);
        }

        return finalACCandidates;
    }


    //getting highest score AC candidate (AP/LL/Adam) without Beam Search
    private ArrayList<Integer> getBestACCandidatesGreedy
            (Sentence sentence, int pIdx,  ArrayList<Integer> aiCandidates) throws Exception {

        int numOfLabels = acClassifier.getNumOutputs();
        ArrayList<Integer> acCandids = new ArrayList<Integer>();
        for (int aiCandidIdx = 0; aiCandidIdx < aiCandidates.size(); aiCandidIdx++) {
            int wordIdx = aiCandidates.get(aiCandidIdx);
            double[] featVector = acClassifier.maps.features(new BaseFeatures(pIdx, wordIdx, sentence), 0);
            double[] labelScores = new double[numOfLabels];
            labelScores = acClassifier.output(featVector, labelScores);
            int predictedLabel = argmax(labelScores);
            acCandids.add(predictedLabel);
        }
        assert aiCandidates.size() == acCandids.size();
        return acCandids;
    }

    private HashMap<Integer, String> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates,
             NNIndexMaps maps) {

        double highestScore = Double.NEGATIVE_INFINITY;
        ArrayList<Integer> highestScoreACSeq = new ArrayList<Integer>();
        int highestScoreSeqAIIndex = -1;
        int bestAIIndex = 0;
        int bestACIndex = 0;

        for (int aiIndex = 0; aiIndex < aiCandidates.size(); aiIndex++) {
            for (int acIndex = 0; acIndex < acCandidates.get(aiIndex).size(); acIndex++) {
                Pair<Double, ArrayList<Integer>> ar = acCandidates.get(aiIndex).get(acIndex);
                double score = ar.first;
                if (score > highestScore) {
                    highestScore = score;
                    bestAIIndex = aiIndex;
                    bestACIndex = acIndex;
                }
            }
        }

        //after finding highest score sequence in the list of AC candidates
        HashMap<Integer, String> wordIndexLabelMap = new HashMap<>();

        ArrayList<Integer> acResult = acCandidates.get(bestAIIndex).get(bestACIndex).second;
        ArrayList<Integer> aiResult = aiCandidates.get(bestAIIndex).second;
        assert acResult.size() == aiResult.size();

        for (int i = 0; i < acResult.size(); i++)
            wordIndexLabelMap.put(aiResult.get(i), maps.revLabel.get(acResult.get(i)));
        return wordIndexLabelMap;
    }

    private HashMap<Integer, String> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates) {

        TreeSet<Pair<Double, ArrayList<Integer>>> sortedCandidates = new TreeSet<Pair<Double, ArrayList<Integer>>>(aiCandidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = sortedCandidates.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, String> wordIndexLabelMap = new HashMap<Integer, String>();
        ArrayList<Integer> highestScoreSeq = highestScorePair.second;

        for (int index : highestScoreSeq) {
            wordIndexLabelMap.put(index, "1");
        }

        return wordIndexLabelMap;
    }

    private int argmax(double[] scores) {
        int argmax = -1;
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > max) {
                argmax = i;
                max = scores[i];
            }
        }
        return argmax;
    }

}
