package SupervisedSRL.Features;

import Sentence.Sentence;
import SupervisedSRL.Strcutures.IndexMap;
import util.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/2/16
 * Time: 10:59 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class BaseFeatures {
    public class Features<T>{
        public ArrayList<T> features;
        public Features(T... feats){
            features= new ArrayList<T>();
            for(T f:feats)
                features.add(f);
        }
    }

    public BaseFeatures(int pIdx, int aIdx, Sentence sentence, IndexMap indexMap) throws Exception {
        this.pIdx = pIdx;
        this.aIdx = aIdx;
        this.sentence = sentence;
        this.indexMap = indexMap;
        extract();
    }

    public int pIdx;
    public int aIdx;
    public Sentence sentence;
    public IndexMap indexMap;
    public Features<Integer> wordFeatures;
    public Features<Integer> posFeatures;
    public Features<Integer> dependencyFeatures;
    public  Features<String> senseFeatures;
    public Features<String> subcatFeatures;
    public Features<String> depRelPathFeatures;
    public Features<String> posPathFeatures;
    public Features<Integer> positionFeatures;

    // Lemma features.
    public int plem;

    // Depset features.
    public String pchilddepset;

    // Pos set features
    public String pchildposset;

    // Word set features.
    public String pchildwset;

    //word cluster features
    public int pw_cluster;
    public int plem_cluster;
    public int pprw_cluster;
    public int aw_cluster;
    public int leftw_cluster;
    public int rightw_cluster;
    public int rightsiblingw_cluster;
    public int leftsiblingw_cluster;

    private void extract() throws Exception {
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceLemmas = sentence.getLemmas();
        int[] sentenceWordsClusterIds = sentence.getWordClusterIds();
        int[] sentenceLemmaClusterIds = sentence.getLemmaClusterIds();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();
        HashMap<Integer, String> sentencePredicatesInfo = sentence.getPredicatesInfo();
        int leftMostDependentIndex = FeatureExtractor.getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int rightMostDependentIndex = FeatureExtractor.getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int parIndex = sentenceDepHeads[aIdx];
        int lefSiblingIndex = FeatureExtractor.getLeftSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);
        int rightSiblingIndex = FeatureExtractor.getRightSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);

        int pw = sentenceWords[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int aw = sentenceWords[aIdx];
        int leftw = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[leftMostDependentIndex];
        int rightw = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[rightMostDependentIndex];
        int rightsiblingw = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[rightSiblingIndex];
        int leftsiblingw = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[lefSiblingIndex];

        wordFeatures = new Features<Integer>(pw, pprw, aw, leftw, rightw, rightsiblingw, leftsiblingw);

        int ppos = sentencePOSTags[pIdx];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        int apos = sentencePOSTags[aIdx];
        int leftpos = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[leftMostDependentIndex];
        int rightpos = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[rightMostDependentIndex];
        int rightsiblingpos = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[rightSiblingIndex];
        int leftsiblingpos = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[lefSiblingIndex];

        posFeatures = new Features<Integer>(ppos, pprpos, apos, leftpos, rightpos, leftsiblingpos, rightsiblingpos);

        int pdeprel = sentenceDepLabels[pIdx];
        int adeprel = sentenceDepLabels[aIdx];
        dependencyFeatures = new Features<Integer>(adeprel, pdeprel);

        String pSense = sentencePredicatesInfo.get(pIdx);
        senseFeatures = new Features<String>(pSense);

        String pdepsubcat = FeatureExtractor.getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        subcatFeatures = new Features<String>(pdepsubcat);

        String deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
        depRelPathFeatures = new Features<String>(deprelpath);

        String pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));
        posPathFeatures = new Features<String>(pospath);

        int position = 0;
        if (pIdx < aIdx)
            position = 2; //after
        else if (pIdx > aIdx)
            position = 1; //before
        positionFeatures = new Features<Integer>(position);

        // other features.
        pw_cluster = sentenceWordsClusterIds[pIdx];
        plem = sentenceLemmas[pIdx];
        plem_cluster = sentenceLemmaClusterIds[pIdx];
        pprw_cluster = sentenceWordsClusterIds[sentenceDepHeads[pIdx]];
        pchilddepset = FeatureExtractor.getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        pchildposset = FeatureExtractor.getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        pchildwset = FeatureExtractor.getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);
        aw_cluster = sentenceWordsClusterIds[aIdx];
        leftw_cluster = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsClusterIds[leftMostDependentIndex];
        rightw_cluster = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsClusterIds[rightMostDependentIndex];
        rightsiblingw_cluster = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsClusterIds[rightSiblingIndex];
        leftsiblingw_cluster = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsClusterIds[lefSiblingIndex];
    }
}
