package SupervisedSRL.Strcutures;

import ml.AveragedPerceptron;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Maryam Aminian on 6/22/16.
 */
public class ModelInfo implements Serializable {

    AveragedPerceptron classifier;
    HashMap<Object, Integer>[] featDict;
    HashMap<String, Integer> labelDict;
    IndexMap indexMap;

    public ModelInfo(AveragedPerceptron classifier, IndexMap indexMap) throws IOException {
        HashMap<Object, CompactArray>[] weights = classifier.getWeights();
        HashMap<Object, CompactArray>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap = classifier.getReverseLabelMap();
        int iteration = classifier.getIteration();

        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<>();
            for (Object feat : weights[f].keySet()) {
                HashMap<Integer, Double> w = weights[f].get(feat).getArray();
                HashMap<Integer, Double> aw = avgWeights[f].get(feat).getArray();
                HashMap<Integer, Double> naw = new HashMap<>();
                for (int i : w.keySet()) {
                    naw.put(i, w.get(i) - (aw.get(i) / iteration));
                }
                CompactArray nawCompact = new CompactArray(naw);
                newAvgMap[f].put(feat, nawCompact);
            }
        }
        this.classifier = new AveragedPerceptron(newAvgMap, labelMap, reverseLabelMap);
        this.indexMap = indexMap;
    }

    public ModelInfo(String modelPath) throws Exception {
        FileInputStream fis = new FileInputStream(modelPath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        HashMap<Object, CompactArray>[] newAvgWeight =
                (HashMap<Object, CompactArray>[]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();
        IndexMap indexMap = (IndexMap) reader.readObject();
        fis.close();
        gz.close();
        reader.close();

        this.classifier = new AveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);
        this.indexMap = indexMap;
    }

    public static void saveModel(AveragedPerceptron classifier, String filePath) throws Exception {
        HashMap<Object, CompactArray>[] weights = classifier.getWeights();
        HashMap<Object, CompactArray>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap = classifier.getReverseLabelMap();
        int iteration = classifier.getIteration();

        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<>();
            for (Object feat : weights[f].keySet()) {
                HashMap<Integer, Double> w = weights[f].get(feat).getArray();
                HashMap<Integer, Double> aw = avgWeights[f].get(feat).getArray();
                HashMap<Integer, Double> naw = new HashMap<>();
                for (int i : w.keySet()) {
                    naw.put(i, w.get(i) - (aw.get(i) / iteration));
                }
                CompactArray nawCompact = new CompactArray(naw);
                newAvgMap[f].put(feat, nawCompact);
            }
        }

        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(newAvgMap);
        writer.writeObject(labelMap);
        writer.writeObject(reverseLabelMap);
        writer.close();
    }

    public static void saveDataPartition(ArrayList<String> part, String filePath) throws IOException {
        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(part);
        writer.close();
    }

    public static Pair<AveragedPerceptron, AveragedPerceptron> loadTrainedModels(String aiModelPath, String acModelPath) throws Exception {
        AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(aiModelPath);
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(acModelPath);
        return new Pair<>(aiClassifier, acClassifier);
    }

    public IndexMap getIndexMap() {
        return indexMap;
    }
}