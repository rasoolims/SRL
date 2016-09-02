package SupervisedSRL.Strcutures;

import SupervisedSRL.Features.BaseFeatures;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.*;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/2/16
 * Time: 11:46 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class NNIndexMaps implements Serializable {
    public HashMap<String, Integer> wordFeatMap;
    public HashMap<String, Integer> posFeatMap;
    public HashMap<String, Integer> depFeatMap;
    public HashMap<String, Integer> subcatFeatMap;
    public HashMap<String, Integer> depPathFeatMap;
    public HashMap<String, Integer> posPathFeatMap;
    public HashMap<Integer, Integer>[] preComputeMap;
    public HashMap<String, Integer> labelMap;
    public ArrayList<String> revLabel;
    Random random;

    public int featDim;
    public int numWordFeatures;
    public int numDepFeatures;
    public int numPosFeatures;
    public int numSubcatFeatures;
    public int numDepPathFeatures;
    public int numPosPathFeatures;
    public int numPositionFeatures;
    private HashMap<Integer, double[]> embeddingsDictionary;

    public static final int UnknownIndex = 1;
    public static final int NullIndex = 0;

    
    public NNIndexMaps(Random random) {
        wordFeatMap = new HashMap<>();
        posFeatMap = new HashMap<>();
        depFeatMap = new HashMap<>();
        subcatFeatMap = new HashMap<>();
        depPathFeatMap = new HashMap<>();
        posPathFeatMap = new HashMap<>();
        labelMap = new HashMap<>();
        revLabel = new ArrayList<>();
        this.random = random;
    }

    public void addToMap(BaseFeatures baseFeatures){
        numWordFeatures = baseFeatures.wordFeatures.features.size();
        numPosFeatures = baseFeatures.posFeatures.features.size();
        numDepFeatures = baseFeatures.dependencyFeatures.features.size();
        numSubcatFeatures = baseFeatures.subcatFeatures.features.size();
        numDepPathFeatures = baseFeatures.depRelPathFeatures.features.size();
        numPosPathFeatures = baseFeatures.posPathFeatures.features.size();
        numPositionFeatures = 1;

        featDim = 1;
        for(String wordFeat: baseFeatures.wordFeatures.features){
            if(!wordFeatMap.containsKey(wordFeat)){
                int index = wordFeatMap.size()+2;
                wordFeatMap.put(wordFeat, index);
            }
            featDim++;
        }

        for(String posFeat: baseFeatures.posFeatures.features){
            if(!posFeatMap.containsKey(posFeat)){
                int index = posFeatMap.size()+2;
                posFeatMap.put(posFeat, index);
            }
            featDim++;
        }

        for(String depFeat: baseFeatures.dependencyFeatures.features){
            if(!depFeatMap.containsKey(depFeat)){
                int index = depFeatMap.size()+2;
                depFeatMap.put(depFeat, index);
            }
            featDim++;
        }

        for(String subcatFeat: baseFeatures.subcatFeatures.features){
            if(!subcatFeatMap.containsKey(subcatFeat)){
                int index = subcatFeatMap.size()+2;
                subcatFeatMap.put(subcatFeat, index);
            }
            featDim++;
        }

        for(String depPath: baseFeatures.depRelPathFeatures.features){
            if(!depPathFeatMap.containsKey(depPath)){
                int index = depPathFeatMap.size();
                depPathFeatMap.put(depPath, index);
            }
            featDim++;
        }

        for(String posPath: baseFeatures.posPathFeatures.features){
            if(!posPathFeatMap.containsKey(posPath)){
                int index = posPathFeatMap.size()+2;
                posPathFeatMap.put(posPath, index);
            }
            featDim++;
        }
    }

    public double[] features(BaseFeatures baseFeatures, double dropout) {
        double[] feats = new double[featDim];
        int i = 0;
        for (String wordFeat : baseFeatures.wordFeatures.features) {
            if (random.nextDouble() < dropout)
                feats[i++] = UnknownIndex;
            else
                feats[i++] = word2int(wordFeat);
        }

        for (String posFeat : baseFeatures.posFeatures.features) {
           feats[i++] = pos2int(posFeat);
        }

        for (String depFeat : baseFeatures.dependencyFeatures.features) {
           feats[i++] = dep2int(depFeat);
        }

        for (String subcatFeat : baseFeatures.subcatFeatures.features) {
            if (random.nextDouble() < dropout)
                feats[i++] = UnknownIndex;
            else
                feats[i++] = subcat2int(subcatFeat);
        }

        for (String depPath : baseFeatures.depRelPathFeatures.features) {
            if (random.nextDouble() < dropout)
                feats[i++] = UnknownIndex;
            else
                feats[i++] = depPath2int(depPath);
        }

        for (String posPath : baseFeatures.posPathFeatures.features) {
            if (random.nextDouble() < dropout)
                feats[i++] = UnknownIndex;
            else
                feats[i++] = posPath2int(posPath);
        }
        feats[i++] = baseFeatures.positionFeatures.features.get(0);

        return feats;
    }
    
    public int word2int(String word){
        if(wordFeatMap.containsKey(word))
            return wordFeatMap.get(word);
        return UnknownIndex;
    }

    public int pos2int(String pos){
        if(posFeatMap.containsKey(pos))
            return posFeatMap.get(pos);
        return UnknownIndex;
    }

    public int dep2int(String dep){
        if(depFeatMap.containsKey(dep))
            return depFeatMap.get(dep);
        return UnknownIndex;
    }

    public int subcat2int(String subcat){
        if(subcatFeatMap.containsKey(subcat))
            return subcatFeatMap.get(subcat);
        return UnknownIndex;
    }

    public int depPath2int(String depPath){
        if(depPathFeatMap.containsKey(depPath))
            return depPathFeatMap.get(depPath);
        return UnknownIndex;
    }

    public int posPath2int(String posPath){
        if(posPathFeatMap.containsKey(posPath))
            return posPathFeatMap.get(posPath);
        return UnknownIndex;
    }

    public void constructPreComputeMap(List<NeuralTrainingInstance> instances, int numWordLayer, int maxNumber) {
        HashMap<Integer, Integer>[] counts = new HashMap[numWordLayer];
        preComputeMap = new HashMap[numWordLayer];

        for (int i = 0; i < counts.length; i++) {
            counts[i] = new HashMap<>();
            preComputeMap[i] = new HashMap<>();
        }

        for (NeuralTrainingInstance instance : instances) {
            double[] feats = instance.getFeatures();
            for (int i = 0; i < numWordLayer; i++) {
                int f = (int) feats[i];
                if (counts[i].containsKey(f))
                    counts[i].put(f, counts[i].get(f) + 1);
                else
                    counts[i].put(f, 1);
            }
        }

        TreeMap<Integer, HashSet<edu.columbia.cs.nlp.CuraParser.Structures.Pair<Integer, Integer>>> sortedCounts = new TreeMap<>();
        for (int i = 0; i < counts.length; i++) {
            for (int f : counts[i].keySet()) {
                int count = counts[i].get(f);
                if (!sortedCounts.containsKey(count))
                    sortedCounts.put(count, new HashSet<edu.columbia.cs.nlp.CuraParser.Structures.Pair<Integer, Integer>>());
                sortedCounts.get(count).add(new edu.columbia.cs.nlp.CuraParser.Structures.Pair<>(i, f));
            }
        }

        int c = 0;
        int[] slotcounter = new int[preComputeMap.length];

        for (int count : sortedCounts.descendingKeySet()) {
            for (edu.columbia.cs.nlp.CuraParser.Structures.Pair<Integer, Integer> p : sortedCounts.get(count)) {
                c++;
                preComputeMap[p.first].put(p.second, slotcounter[p.first]);
                slotcounter[p.first] += 1;
            }
            if (c >= maxNumber)
                break;
        }
    }

    public int readEmbeddings(String path) throws Exception {
        embeddingsDictionary = new HashMap<>();
        int eDim = 64;

        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split(" ");
            int wordIndex = -1;
            if (wordFeatMap.containsKey(spl[0]))
                wordIndex = wordFeatMap.get(spl[0]);
            if (wordFeatMap.containsKey(spl[0].toLowerCase()))
                wordIndex = wordFeatMap.get(spl[0].toLowerCase());
            else if (spl[0].equals("_unk_"))
                wordIndex = UnknownIndex;
            if (wordIndex != -1) {
                double[] e = new double[spl.length - 1];
                eDim = e.length;
                for (int i = 0; i < e.length; i++) {
                    e[i] = Double.parseDouble(spl[i + 1]);
                }
                Utils.normalize(e);
                embeddingsDictionary.put(wordIndex, e);
            }
        }
        return eDim;
    }

    public HashMap<Integer, double[]> getEmbeddingsDictionary() {
        return embeddingsDictionary;
    }

    public void emptyEmbeddings() {
        embeddingsDictionary = null;
    }

    public void addLabel(String label) {
        if (!labelMap.containsKey(label)) {
            int index = labelMap.size();
            labelMap.put(label, index);
            revLabel.add(label);
        }
    }
}
