package SupervisedSRL.Strcutures;

import SupervisedSRL.Features.BaseFeatures;
import com.sun.xml.internal.rngom.parse.host.Base;

import java.util.HashMap;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/2/16
 * Time: 11:46 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class NNIndexMaps {
    public HashMap<Integer, Integer> wordFeatMap;
    public HashMap<Integer, Integer> posFeatMap;
    public HashMap<Integer, Integer> depFeatMap;
    public HashMap<String, Integer> subcatFeatMap;
    public HashMap<String, Integer> depPathFeatMap;
    public HashMap<String, Integer> posPathFeatMap;
    public int featDim;
    public int numWordFeatures;
    public int numDepFeatures;
    public int numPosFeatures;
    public int numSubcatFeatures;
    public int numDepPathFeatures;
    public int numPosPathFeatures;
    public int numPositionFeatures;

    public static final int UnknownIndex = 1;
    public static final int NullIndex = 0;

    
    public NNIndexMaps() {
        wordFeatMap = new HashMap<Integer, Integer>();
        posFeatMap = new HashMap<Integer, Integer>();
        depFeatMap = new HashMap<Integer, Integer>();
        subcatFeatMap = new HashMap<String, Integer>();
        depPathFeatMap = new HashMap<String, Integer>();
        posPathFeatMap = new HashMap<String, Integer>();
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
        for(int wordFeat: baseFeatures.wordFeatures.features){
            if(!wordFeatMap.containsKey(wordFeat)){
                int index = wordFeatMap.size()+2;
                wordFeatMap.put(wordFeat, index);
            }
            featDim++;
        }

        for(int posFeat: baseFeatures.posFeatures.features){
            if(!posFeatMap.containsKey(posFeat)){
                int index = posFeatMap.size()+2;
                posFeatMap.put(posFeat, index);
            }
            featDim++;
        }

        for(int depFeat: baseFeatures.dependencyFeatures.features){
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

    public double[] features(BaseFeatures baseFeatures){
       double[] feats = new double[featDim];
        int i=0;
        feats[i++] = baseFeatures.posFeatures.features.get(0);
        for(int wordFeat: baseFeatures.wordFeatures.features){
              feats[i++] = word2int(wordFeat);
        }

        for(int posFeat: baseFeatures.posFeatures.features){
            feats[i++] = pos2int(posFeat);
        }

        for(int depFeat: baseFeatures.dependencyFeatures.features){
            feats[i++] = dep2int(depFeat);
        }

        for(String subcatFeat: baseFeatures.subcatFeatures.features){
            feats[i++] = subcat2int(subcatFeat);
        }

        for(String depPath: baseFeatures.depRelPathFeatures.features){
            feats[i++] = depPath2int(depPath);
        }

        for(String posPath: baseFeatures.posPathFeatures.features){
            feats[i++] = posPath2int(posPath);
        }
        return feats;
    }
    
    public int word2int(int word){
        if(wordFeatMap.containsKey(word))
            return wordFeatMap.get(word);
        return UnknownIndex;
    }

    public int pos2int(int pos){
        if(posFeatMap.containsKey(pos))
            return posFeatMap.get(pos);
        return UnknownIndex;
    }

    public int dep2int(int dep){
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
}
