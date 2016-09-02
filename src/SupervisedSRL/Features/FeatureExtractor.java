package SupervisedSRL.Features;
/**
 * Created by Maryam Aminian on 5/17/16.
 */

import SentenceStructures.Sentence;
import util.StringUtils;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;

public class FeatureExtractor {
    static HashSet<String> punctuations = new HashSet<String>();

    static {
        punctuations.add("P");
        punctuations.add("PUNC");
        punctuations.add("PUNCT");
        punctuations.add("p");
        punctuations.add("punc");
        punctuations.add("punct");
        punctuations.add(",");
        punctuations.add(";");
        punctuations.add(".");
        punctuations.add("#");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add("[");
        punctuations.add("]");
        punctuations.add("{");
        punctuations.add("}");
        punctuations.add("\"");
        punctuations.add(",");
        punctuations.add(".");
        punctuations.add(":");
        punctuations.add("``");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add("-LSB-");
        punctuations.add("-RSB-");
        punctuations.add("-LCB-");
        punctuations.add("-RCB-");
        punctuations.add("!");
        punctuations.add(".");
        punctuations.add("#");
        punctuations.add("$");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add(",");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add(":");
        punctuations.add("?");
    }

    public static Object[] extractPDFeatures(int pIdx, Sentence sentence, int length)
            throws Exception {
        Object[] features = new Object[length];
        String[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        String[] sentenceWords = sentence.getWords();
        String[] sentencePOSTags = sentence.getPosTags();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        String pw = sentenceWords[pIdx];
        String ppos = sentencePOSTags[pIdx];
        String pdeprel = sentenceDepLabels[pIdx];
        String pprw = sentenceWords[sentenceDepHeads[pIdx]];
        String pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags);

        int index = 0;
        features[index++] = pw;
        features[index++] = ppos;
        features[index++] = pdeprel;
        features[index++] = pprw;
        features[index++] = pprpos;
        features[index++] = pdepsubcat;
        features[index++] = pchilddepset;
        features[index++] = pchildposset;
        features[index++] = pchildwset;

        return features;
    }

    public static String getDepSubCat(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                      String[] sentenceDepLabels, String[] posTags) throws Exception {
        StringBuilder subCat = new StringBuilder();
        if (sentenceReverseDepHeads[pIdx] != null) {
            //predicate has >1 children
            for (int child : sentenceReverseDepHeads[pIdx]) {
                String pos = posTags[child];
                if (!punctuations.contains(pos)) {
                    subCat.append(sentenceDepLabels[child]);
                    subCat.append("\t");
                }
            }
        }
        return subCat.toString().trim();
    }

    public static String getChildSet(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                     String[] collection, String[] posTags) throws Exception {
        StringBuilder childSet = new StringBuilder();
        TreeSet<String> children = new TreeSet<>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            for (int child : sentenceReverseDepHeads[pIdx]) {
                String pos = posTags[child];
                if (!punctuations.contains(pos)) {
                    children.add(collection[child]);
                }
            }
        }
        for (String child : children) {
            childSet.append(child);
            childSet.append("\t");
        }
        return childSet.toString().trim();
    }

    public static int getLeftMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null) {
            //this argument has at least one child
            int firstChild = sentenceReverseDepHeads[aIdx].first();
            // this should be on the left side.
            if (firstChild < aIdx) {
                return firstChild;
            }
        }
        return -1;
    }

    public static int getRightMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null) {
            int last = sentenceReverseDepHeads[aIdx].last();
            if (last > aIdx) {
                return sentenceReverseDepHeads[aIdx].last();
            }
        }
        return -1;
    }

    public static int getLeftSiblingIndex(int aIdx, int parIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[parIdx] != null) {
            argSiblings = sentenceReverseDepHeads[parIdx];
        }

        if (argSiblings.lower(aIdx) != null)
            return argSiblings.lower(aIdx);
        return -1;
    }

    public static int getRightSiblingIndex(int aIdx, int parIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[parIdx] != null)
            argSiblings = sentenceReverseDepHeads[parIdx];

        if (argSiblings.higher(aIdx) != null)
            return argSiblings.higher(aIdx);
        return -1;
    }

    public static class BaseFeatureFields {
        private int pIdx;
        private int aIdx;
        private Sentence sentence;
        private String pw;
        private String ppos;
        private String plem;
        private String pSense;
        private String pdeprel;
        private String pprw;
        private String pprpos;
        private String pdepsubcat;
        private String pchilddepset;
        private String pchildposset;
        private String pchildwset;
        private String aw;
        private String apos;
        private String adeprel;
        private String deprelpath;
        private String pospath;
        private int position;
        private String leftw;
        private String leftpos;
        private String rightw;
        private String rightpos;
        private String rightsiblingw;
        private String rightsiblingpos;
        private String leftsiblingw;
        private String leftsiblingpos;

        //word cluster features
        private String pw_cluster;
        private String plem_cluster;
        private String pprw_cluster;
        private String aw_cluster;
        private String leftw_cluster;
        private String rightw_cluster;
        private String rightsiblingw_cluster;
        private String leftsiblingw_cluster;

        public BaseFeatureFields(int pIdx, int aIdx, Sentence sentence) {
            this.pIdx = pIdx;
            this.aIdx = aIdx;
            this.sentence = sentence;
        }

        public String getPw() {
            return pw;
        }

        public String getPpos() {
            return ppos;
        }

        public String getPlem() {
            return plem;
        }

        public String getpSense() {
            return pSense;
        }

        public String getPdeprel() {
            return pdeprel;
        }

        public String getPprw() {
            return pprw;
        }

        public String getPprpos() {
            return pprpos;
        }

        public String getPdepsubcat() {
            return pdepsubcat;
        }

        public String getPchilddepset() {
            return pchilddepset;
        }

        public String getPchildposset() {
            return pchildposset;
        }

        public String getPchildwset() {
            return pchildwset;
        }

        public String getAw() {
            return aw;
        }

        public String getApos() {
            return apos;
        }

        public String getAdeprel() {
            return adeprel;
        }

        public String getDeprelpath() {
            return deprelpath;
        }

        public String getPospath() {
            return pospath;
        }

        public int getPosition() {
            return position;
        }

        public String getLeftw() {
            return leftw;
        }

        public String getLeftpos() {
            return leftpos;
        }

        public String getRightw() {
            return rightw;
        }

        public String getRightpos() {
            return rightpos;
        }

        public String getRightsiblingw() {
            return rightsiblingw;
        }

        public String getRightsiblingpos() {
            return rightsiblingpos;
        }

        public String getLeftsiblingw() {
            return leftsiblingw;
        }

        public String getLeftsiblingpos() {
            return leftsiblingpos;
        }

        public String getPw_cluster() {return pw_cluster;}

        public String getPlem_cluster() {return plem_cluster;}

        public String getAw_cluster() {return aw_cluster;}

        public String getPprw_cluster() {return pprw_cluster;}

        public String getLeftw_cluster() {return leftw_cluster;}

        public String getRightw_cluster() {return rightw_cluster;}

        public String getRightsiblingw_cluster() {return rightsiblingw_cluster;}

        public String getLeftsiblingw_cluster() {return leftsiblingw_cluster;}

        public BaseFeatureFields invoke() throws Exception {
            String[] sentenceDepLabels = sentence.getDepLabels();
            int[] sentenceDepHeads = sentence.getDepHeads();
            String[] sentenceWords = sentence.getWords();
            String[] sentencePOSTags = sentence.getPosTags();
            String[] sentenceLemmas = sentence.getLemmas();
            String[] sentenceWordsClusterIds = sentence.getWordClusterIds();
            TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();
            HashMap<Integer, String> sentencePredicatesInfo = sentence.getPredicatesInfo();

            //predicate features
            pw = sentenceWords[pIdx];
            pw_cluster= sentenceWordsClusterIds[pIdx];
            ppos = sentencePOSTags[pIdx];
            plem = sentenceLemmas[pIdx];
            pSense = sentencePredicatesInfo.get(pIdx);
            pdeprel = sentenceDepLabels[pIdx];
            pprw = sentenceWords[sentenceDepHeads[pIdx]];
            pprw_cluster= sentenceWordsClusterIds[sentenceDepHeads[pIdx]];
            pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
            pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags);
            pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags);
            pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags);
            pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags);

            int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int parIndex = sentenceDepHeads[aIdx];
            int lefSiblingIndex = getLeftSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);
            int rightSiblingIndex = getRightSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);

            //argument features
            aw = sentenceWords[aIdx];
            aw_cluster= sentenceWordsClusterIds[aIdx];
            apos = sentencePOSTags[aIdx];
            adeprel = sentenceDepLabels[aIdx];

            //predicate-argument features
            deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
            pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));

            position = 0;
            if (pIdx < aIdx)
                position = 2; //after
            else if (pIdx > aIdx)
                position = 1; //before

            leftw = leftMostDependentIndex == -1 ? "_NULL_" : sentenceWords[leftMostDependentIndex];
            leftw_cluster= leftMostDependentIndex == -1 ?"_NULL_": sentenceWordsClusterIds[leftMostDependentIndex];
            leftpos = leftMostDependentIndex == -1 ? "_NULL_": sentencePOSTags[leftMostDependentIndex];
            rightw = rightMostDependentIndex == -1 ? "_NULL_": sentenceWords[rightMostDependentIndex];
            rightw_cluster = rightMostDependentIndex == -1 ? "_NULL_" : sentenceWordsClusterIds[rightMostDependentIndex];
            rightpos = rightMostDependentIndex == -1 ? "_NULL_" : sentencePOSTags[rightMostDependentIndex];
            rightsiblingw = rightSiblingIndex == -1 ?"_NULL_" : sentenceWords[rightSiblingIndex];
            rightsiblingw_cluster = rightSiblingIndex == -1 ?"_NULL_" : sentenceWordsClusterIds[rightSiblingIndex];
            rightsiblingpos = rightSiblingIndex == -1 ? "_NULL_" : sentencePOSTags[rightSiblingIndex];
            leftsiblingw = lefSiblingIndex == -1 ? "_NULL_" : sentenceWords[lefSiblingIndex];
            leftsiblingw_cluster = lefSiblingIndex == -1 ? "_NULL_" : sentenceWordsClusterIds[lefSiblingIndex];
            leftsiblingpos = lefSiblingIndex == -1 ?"_NULL_" : sentencePOSTags[lefSiblingIndex];
            return this;
        }
    }
}
