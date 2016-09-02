package SupervisedSRL.Features;

import SentenceStructures.Sentence;
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

    public BaseFeatures(int pIdx, int aIdx, Sentence sentence) throws Exception {
        this.pIdx = pIdx;
        this.aIdx = aIdx;
        this.sentence = sentence;
        extract();
    }

    public int pIdx;
    public int aIdx;
    public Sentence sentence;
    public Features<String> wordFeatures;
    public Features<String> posFeatures;
    public Features<String> dependencyFeatures;
    public  Features<String> senseFeatures;
    public Features<String> subcatFeatures;
    public Features<String> depRelPathFeatures;
    public Features<String> posPathFeatures;
    public Features<Integer> positionFeatures;

    // Lemma features.
    public String plem;

    // Depset features.
    public String pchilddepset;

    // Pos set features
    public String pchildposset;

    // Word set features.
    public String pchildwset;

    //word cluster features
    public String pw_cluster;
    public String plem_cluster;
    public String pprw_cluster;
    public String aw_cluster;
    public String leftw_cluster;
    public String rightw_cluster;
    public String rightsiblingw_cluster;
    public String leftsiblingw_cluster;

    private void extract() throws Exception {
        String[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        String[] sentenceWords = sentence.getWords();
        String[] sentencePOSTags = sentence.getPosTags();
        String[] sentenceLemmas = sentence.getLemmas();
        String[] sentenceWordsClusterIds = sentence.getWordClusterIds();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();
        HashMap<Integer, String> sentencePredicatesInfo = sentence.getPredicatesInfo();
        int leftMostDependentIndex = FeatureExtractor.getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int rightMostDependentIndex = FeatureExtractor.getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int parIndex = sentenceDepHeads[aIdx];
        int lefSiblingIndex = FeatureExtractor.getLeftSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);
        int rightSiblingIndex = FeatureExtractor.getRightSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);

        String pw = sentenceWords[pIdx];
        String pprw = sentenceWords[sentenceDepHeads[pIdx]];
        String aw = sentenceWords[aIdx];
        // todo then connect this to the null index in the new nn map.
        String leftw = leftMostDependentIndex == -1 ? "_NULL_" : sentenceWords[leftMostDependentIndex];
        String rightw = rightMostDependentIndex == -1 ? "_NULL_"  : sentenceWords[rightMostDependentIndex];
        String rightsiblingw = rightSiblingIndex == -1 ? "_NULL_"  : sentenceWords[rightSiblingIndex];
        String leftsiblingw = lefSiblingIndex == -1 ? "_NULL_"  : sentenceWords[lefSiblingIndex];

        wordFeatures = new Features<>(pw, pprw, aw, leftw, rightw, rightsiblingw, leftsiblingw);

        String ppos = sentencePOSTags[pIdx];
        String pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String apos = sentencePOSTags[aIdx];
        String leftpos = leftMostDependentIndex == -1 ? "_NULL_"  : sentencePOSTags[leftMostDependentIndex];
        String rightpos = rightMostDependentIndex == -1 ?"_NULL_" : sentencePOSTags[rightMostDependentIndex];
        String rightsiblingpos = rightSiblingIndex == -1 ?"_NULL_"  : sentencePOSTags[rightSiblingIndex];
        String leftsiblingpos = lefSiblingIndex == -1 ? "_NULL_"  : sentencePOSTags[lefSiblingIndex];

        posFeatures = new Features<>(ppos, pprpos, apos, leftpos, rightpos, leftsiblingpos, rightsiblingpos);

        String pdeprel = sentenceDepLabels[pIdx];
        String adeprel = sentenceDepLabels[aIdx];
        dependencyFeatures = new Features<>(adeprel, pdeprel);

        String pSense = sentencePredicatesInfo.get(pIdx);
        senseFeatures = new Features<>(pSense);

        String pdepsubcat = FeatureExtractor.getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags);
        subcatFeatures = new Features<>(pdepsubcat);

        String deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
        depRelPathFeatures = new Features<>(deprelpath);

        String pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));
        posPathFeatures = new Features<>(pospath);

        int position = 0;
        if (pIdx < aIdx)
            position = 2; //after
        else if (pIdx > aIdx)
            position = 1; //before
        positionFeatures = new Features<Integer>(position);

        // other features.
        pw_cluster = sentenceWordsClusterIds[pIdx];
        plem = sentenceLemmas[pIdx];
        pprw_cluster = sentenceWordsClusterIds[sentenceDepHeads[pIdx]];
        pchilddepset = FeatureExtractor.getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags);
        pchildposset = FeatureExtractor.getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags);
        pchildwset = FeatureExtractor.getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags);
        aw_cluster = sentenceWordsClusterIds[aIdx];
        leftw_cluster = leftMostDependentIndex == -1 ? "_NULL_"  : sentenceWordsClusterIds[leftMostDependentIndex];
        rightw_cluster = rightMostDependentIndex == -1? "_NULL_"  : sentenceWordsClusterIds[rightMostDependentIndex];
        rightsiblingw_cluster = rightSiblingIndex == -1 ? "_NULL_"  : sentenceWordsClusterIds[rightSiblingIndex];
        leftsiblingw_cluster = lefSiblingIndex == -1 ? "_NULL_"  : sentenceWordsClusterIds[lefSiblingIndex];
    }
}
