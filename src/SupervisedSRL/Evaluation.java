package SupervisedSRL;

import SentenceStructures.Argument;
import SentenceStructures.PA;
import SentenceStructures.Sentence;
import SupervisedSRL.PD.PD;
import util.IO;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by monadiab on 7/13/16.
 */
public class Evaluation {


    public static double evaluate(String systemOutput, String goldOutput, 
                                  HashMap<String, Integer> reverseLabelMap) throws IOException {
        DecimalFormat format = new DecimalFormat("##.00");

        List<String> systemOutputInCONLLFormat = IO.readCoNLLFile(systemOutput);
        List<String> goldOutputInCONLLFormat = IO.readCoNLLFile(goldOutput);
        Set<String> argLabels = reverseLabelMap.keySet();

        int correctPLabel = 0;
        int wrongPLabel = 0;

        int[][] aiConfusionMatrix = new int[2][2];
        aiConfusionMatrix[0][0] = 0;
        aiConfusionMatrix[0][1] = 0;
        aiConfusionMatrix[1][0] = 0;
        aiConfusionMatrix[1][1] = 0;

        HashMap<String, int[]> acConfusionMatrix = new HashMap<>();
        for (String value : argLabels) {
            int[] acGoldLabels = new int[argLabels.size() + 1];
            acConfusionMatrix.put(value, acGoldLabels);
        }
        int[] acGoldLabels = new int[argLabels.size() + 1];
        acConfusionMatrix.put("", acGoldLabels);

        if (systemOutputInCONLLFormat.size() != goldOutputInCONLLFormat.size()) {
            System.out.print("WARNING --> Number of sentences in System output does not match with number of sentences in the Gold data");
            return -1;
        }

        boolean decode = true;
        for (int senIdx = 0; senIdx < systemOutputInCONLLFormat.size(); senIdx++) {
            //System.out.println("sen: "+senIdx);
            Sentence sysOutSen = new Sentence(systemOutputInCONLLFormat.get(senIdx));
            Sentence goldSen = new Sentence(goldOutputInCONLLFormat.get(senIdx));

            ArrayList<PA> sysOutPAs = sysOutSen.getPredicateArguments().getPredicateArgumentsAsArray();
            ArrayList<PA> goldPAs = goldSen.getPredicateArguments().getPredicateArgumentsAsArray();

            for (PA goldPA : goldPAs) {
                int goldPIdx = goldPA.getPredicateIndex();
                String goldPLabel = goldPA.getPredicateLabel();
                for (PA sysOutPA : sysOutPAs) {
                    int sysOutPIdx = sysOutPA.getPredicateIndex();
                    if (goldPIdx == sysOutPIdx) {
                        //same predicate index (predicate indices are supposed to be given)
                        String sysOutPLabel = sysOutPA.getPredicateLabel();
                        if (goldPLabel.equals(sysOutPLabel)) {
                            //same predicate labels
                            correctPLabel++;
                            //discover argument precision/recall
                            HashMap<Integer, String> sysOutPrediction = convertPredictionToMap(sysOutPA);
                            Object[] confusionMatrices = compareWithGold(goldPA, sysOutPrediction,
                                    aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
                            aiConfusionMatrix = (int[][]) confusionMatrices[0];
                            acConfusionMatrix = (HashMap<String, int[]>) confusionMatrices[1];
                        } else {
                            //different predicate labels
                            wrongPLabel++;
                            //discover argument precision/recall
                            HashMap<Integer, String> sysOutPrediction = convertPredictionToMap(sysOutPA);
                            Object[] confusionMatrices = compareWithGold(goldPA, sysOutPrediction,
                                    aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
                            aiConfusionMatrix = (int[][]) confusionMatrices[0];
                            acConfusionMatrix = (HashMap<String, int[]>) confusionMatrices[1];
                        }
                        break;
                    }
                }
            }
        }
        System.out.println("*********************************************");
        System.out.println("Total Predicate Disambiguation Accuracy " + format.format((double) correctPLabel / (correctPLabel + wrongPLabel)));
        System.out.println("Total Number of Predicate Tokens in dev data: " + PD.totalPreds);
        System.out.println("Total Number of Unseen Predicate Tokens in dev data: " + PD.unseenPreds);
        System.out.println("*********************************************");
        return computePrecisionRecall(aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
    }


    private static Object[] compareWithGold(PA pa, HashMap<Integer, String> highestScorePrediction,
                                            int[][] aiConfusionMatrix, HashMap<String, int[]> acConfusionMatrix,
                                            HashMap<String, Integer> reverseLabelMap) {

        ArrayList<Argument> goldArgs = pa.getArguments();
        HashMap<Integer, String> goldArgMap = getGoldArgMap(goldArgs);
        Set<Integer> goldArgsIndices = goldArgMap.keySet();
        Set<Integer> sysOutArgIndices = getNonZeroArgs(highestScorePrediction);


        HashSet<Integer> exclusiveGoldArgIndices = new HashSet(goldArgsIndices);
        HashSet<Integer> commonGoldPredictedArgIndices = new HashSet(sysOutArgIndices);
        HashSet<Integer> exclusivePredicatedArgIndices = new HashSet(sysOutArgIndices);

        exclusivePredicatedArgIndices.removeAll(goldArgsIndices); //contains argument indices only identified by AI module
        commonGoldPredictedArgIndices.retainAll(goldArgsIndices);
        exclusiveGoldArgIndices.removeAll(sysOutArgIndices);

        aiConfusionMatrix[1][1] += commonGoldPredictedArgIndices.size();
        aiConfusionMatrix[1][0] += exclusivePredicatedArgIndices.size();
        aiConfusionMatrix[0][1] += exclusiveGoldArgIndices.size();

        for (int predictedArgIdx : sysOutArgIndices) {
            String predictedLabel = highestScorePrediction.get(predictedArgIdx);
            if (goldArgMap.containsKey(predictedArgIdx)) {
                //System.out.print("predictedArgIdx: "+predictedArgIdx + "\tGoldLabel: " + goldArgMap.get(predictedArgIdx) +"\n\n");
                String goldLabel = goldArgMap.get(predictedArgIdx);
                int goldLabelIdx = -1;
                if (reverseLabelMap.containsKey(goldLabel)) {
                    //seen gold label in train data
                    goldLabelIdx = reverseLabelMap.get(goldLabel);
                } else
                    goldLabelIdx = acConfusionMatrix.size() - 1;
                acConfusionMatrix.get(predictedLabel)[goldLabelIdx]++;

            } else {
                acConfusionMatrix.get(predictedLabel)[acConfusionMatrix.get(predictedLabel).length - 1]++;
            }
        }

        //update acConfusionMatrix for false negatives
        for (int goldArgIdx : goldArgMap.keySet()) {
            if (!sysOutArgIndices.contains(goldArgIdx)) {
                //ai_fn --> ac_fn
                //System.out.println(goldArgMap.get(goldArgIdx));
                String goldLabel = goldArgMap.get(goldArgIdx);
                int goldLabelIdx = -1;
                //we might see an unseen gold label at this step
                if (reverseLabelMap.containsKey(goldLabel))
                    goldLabelIdx = reverseLabelMap.get(goldArgMap.get(goldArgIdx));
                acConfusionMatrix.get(acConfusionMatrix.get(0).length - 1)[goldLabelIdx]++;
            }
        }
        return new Object[]{aiConfusionMatrix, acConfusionMatrix};
    }


    public static double computePrecisionRecall(int[][] aiConfusionMatrix,
                                                HashMap<String, int[]> acConfusionMatrix, HashMap<String, Integer> reverseLabelMap) {
        DecimalFormat format = new DecimalFormat("##.00");
        //binary classification
        int aiTP = aiConfusionMatrix[1][1];
        int aiFP = aiConfusionMatrix[1][0];
        int aiFN = aiConfusionMatrix[0][1];
        int total_ai_predictions = aiTP + aiFP;

        System.out.println("Total AI prediction " + total_ai_predictions);
        System.out.println("AI Precision: " + format.format((double) aiTP / (aiTP + aiFP)));
        System.out.println("AI Recall: " + format.format((double) aiTP / (aiTP + aiFN)));
        System.out.println("*********************************************");

        int total_ac_predictions = 0;
        int total_tp = 0;
        int total_gold = 0;

        //multi-class classification
        for (String predicatedLabel : acConfusionMatrix.keySet()) {
            if (!predicatedLabel.equals("")) {
                //for real arguments
                int tp = acConfusionMatrix.get(predicatedLabel)[reverseLabelMap.get(predicatedLabel)]; //element on the diagonal
                total_tp += tp;

                int total_prediction_4_this_label = 0;
                for (int element : acConfusionMatrix.get(predicatedLabel))
                    total_prediction_4_this_label += element;

                total_ac_predictions += total_prediction_4_this_label;

                int total_gold_4_this_label = 0;

                for (String pLabel : acConfusionMatrix.keySet())
                    total_gold_4_this_label += acConfusionMatrix.get(pLabel)[reverseLabelMap.get(predicatedLabel)];

                total_gold += total_gold_4_this_label;

                double precision = 100. * (double) tp / total_prediction_4_this_label;
                double recall = 100. * (double) tp / total_gold_4_this_label;
            }
        }

        System.out.println("*********************************************");
        System.out.println("Total AC prediction " + format.format(total_ac_predictions));
        System.out.println("Total number of tp: " + format.format(total_tp));

        double micro_precision = 100. * (double) total_tp / total_ac_predictions;
        double micro_recall = 100. * (double) total_tp / total_gold;
        double FScore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall);

        System.out.println("Micro Precision: " + format.format(micro_precision));
        System.out.println("Micro Recall: " + format.format(micro_recall));
        System.out.println("Averaged F1-score: " + format.format(FScore));
        return FScore;
    }

    //////////// SUPPORTING FUNCTIONS /////////////////////////////////////////////////
    private static HashMap<Integer, String> getGoldArgMap(ArrayList<Argument> args) {
        HashMap<Integer, String> goldArgMap = new HashMap<Integer, String>();
        for (Argument arg : args)
            goldArgMap.put(arg.getIndex(), arg.getType());
        return goldArgMap;
    }


    private static HashSet<Integer> getNonZeroArgs(HashMap<Integer, String> prediction) {
        HashSet<Integer> nonZeroArgs = new HashSet();
        for (int key : prediction.keySet())
            if (!prediction.get(key).equals(""))
                nonZeroArgs.add(key);

        return nonZeroArgs;
    }

    private static HashMap<Integer, String> convertPredictionToMap(PA pa) {
        HashMap<Integer, String> highestScorePrediction = new HashMap<>();
        ArrayList<Argument> args = pa.getArguments();
        for (Argument arg : args)
            highestScorePrediction.put(arg.getIndex(), arg.getType());
        return highestScorePrediction;
    }
}
