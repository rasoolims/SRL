package SupervisedSRL.Features;
/**
 * Created by Maryam Aminian on 5/17/16.
 */

import SentenceStruct.Sentence;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Pair;
import util.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;

public class FeatureExtractor {
    static HashSet<String> punctuations = new HashSet<>();
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

    public static Object[] extractPIFeatures(int wordIdx, Sentence sentence, int length, IndexMap indexMap) throws Exception {
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentenceLemmas = sentence.getLemmas();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceCPOSTags = sentence.getcPosTags();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //word features
        int w = sentenceWords[wordIdx];
        int lem = sentenceLemmas[wordIdx];
        int pos = sentencePOSTags[wordIdx];
        int cPos = sentenceCPOSTags[wordIdx];
        int deprel = sentenceDepLabels[wordIdx];
        int prw = sentenceWords[sentenceDepHeads[wordIdx]];
        int prlem = sentenceLemmas[sentenceDepHeads[wordIdx]];
        int prpos = sentencePOSTags[sentenceDepHeads[wordIdx]];
        int prcpos = sentenceCPOSTags[sentenceDepHeads[wordIdx]];
        String depsubcat = getDepSubCat(wordIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String childdepset = getChildSet(wordIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String childposset = getChildSet(wordIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String childwset = getChildSet(wordIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

        ArrayList<Object> feats= new ArrayList<>();
        feats.add(w);
        feats.add(lem);
        feats.add(pos);
        feats.add(cPos);
        feats.add(deprel);
        feats.add(prw);
        feats.add(prlem);
        feats.add(prpos);
        feats.add(prcpos);
        feats.add(depsubcat);
        feats.add(childdepset);
        feats.add(childposset);
        feats.add(childwset);

        int index = 0;
        features[index++] = w;
        features[index++] = lem;
        features[index++] = pos;
        features[index++] = cPos;
        features[index++] = deprel;
        features[index++] = prw;
        features[index++] = prlem;
        features[index++] = prpos;
        features[index++] = prcpos;
        features[index++] = depsubcat;
        features[index++] = childdepset;
        features[index++] = childposset;
        features[index++] = childwset;

        for (int i=0;i<feats.size();i++){
            for (int j=0; j< feats.size(); j++){
                if (i!= j)
                    features[index++] =  feats.get(i) +" "+ feats.get(j);}
        }
        return features;
    }


    public static Object[] extractPDFeatures(int pIdx, Sentence sentence, int length, IndexMap indexMap)
            throws Exception {
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentenceLemmas = sentence.getLemmas();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceCPOSTags = sentence.getcPosTags();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pw = sentenceWords[pIdx];
        int plem = sentenceLemmas[pIdx];
        int ppos = sentencePOSTags[pIdx];
        int pcPos = sentenceCPOSTags[pIdx];
        int pdeprel = sentenceDepLabels[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int pprlem = sentenceLemmas[sentenceDepHeads[pIdx]];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        int pprcpos = sentenceCPOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

        ArrayList<Object> pFeats= new ArrayList<>();
        pFeats.add(pw);
        pFeats.add(plem);
        pFeats.add(ppos);
        pFeats.add(pcPos);
        pFeats.add(pdeprel);
        pFeats.add(pprw);
        pFeats.add(pprlem);
        pFeats.add(pprpos);
        pFeats.add(pprcpos);
        pFeats.add(pdepsubcat);
        pFeats.add(pchilddepset);
        pFeats.add(pchildposset);
        pFeats.add(pchildwset);

        int index = 0;
        features[index++] = pw;
        features[index++] = plem;
        features[index++] = ppos;
        features[index++] = pcPos;
        features[index++] = pdeprel;
        features[index++] = pprw;
        features[index++] = pprlem;
        features[index++] = pprpos;
        features[index++] = pprcpos;
        features[index++] = pdepsubcat;
        features[index++] = pchilddepset;
        features[index++] = pchildposset;
        features[index++] = pchildwset;

        for (int i=0;i<pFeats.size();i++){
            for (int j=0; j< pFeats.size(); j++){
                if (i!= j)
                    features[index++] =  pFeats.get(i) +" "+ pFeats.get(j);
            }
        }

        return features;
    }

    public static Object[] extractAIFeatures(int pIdx, int aIdx, Sentence sentence, int length,
                                             IndexMap indexMap, boolean extractGlobalFeatures, int label) throws Exception {
        Object[] features = new Object[length];
        BaseFeatureFields baseFeatureFields = new BaseFeatureFields(pIdx, aIdx, sentence, indexMap).invoke();
        Object[] predFeats = addAllPredicateFeatures(baseFeatureFields, features, 0, extractGlobalFeatures, label);
        Object[] argFeats = addAllArgumentFeatures(baseFeatureFields, (Object[]) predFeats[0], (Integer) predFeats[1], extractGlobalFeatures, label);
        Object[] bigramFeatures = addPredicateArgumentBigramFeatures(baseFeatureFields, (Object[]) argFeats[0], (Integer) argFeats[1], extractGlobalFeatures, label);

        Object[] AIFeatures = bigramFeatures;
        return (Object[]) AIFeatures[0];
    }

    public static Object[] extractACFeatures(int pIdx, int aIdx, Sentence sentence, int length,
                                             IndexMap indexMap, boolean extractGlobalFeatures, int label) throws Exception {

        Object[] features = new Object[length];
        BaseFeatureFields baseFeatureFields = new BaseFeatureFields(pIdx, aIdx, sentence, indexMap).invoke();
        Object[] predFeats = addAllPredicateFeatures(baseFeatureFields, features, 0, extractGlobalFeatures, label);
        Object[] argFeats = addAllArgumentFeatures(baseFeatureFields, (Object[]) predFeats[0], (Integer) predFeats[1], extractGlobalFeatures, label);
        Object[] bigramFeatures = addPredicateArgumentBigramFeatures(baseFeatureFields, (Object[]) argFeats[0], (Integer) argFeats[1], extractGlobalFeatures, label);

        Object[] ACFeatures = bigramFeatures;
        return (Object[]) ACFeatures[0];
    }

    public static Object[] extractGlobalFeatures(int pIdx, String pLabel, Pair<Double, ArrayList<Integer>> aiCandid,
                                                 Pair<Double, ArrayList<Integer>> acCandid, String[] labelMap) {

        String seqOfCoreArgumentLabels = "";
        boolean predicateSeen = false;
        for (int i = 0; i < aiCandid.second.size(); i++) {
            int wordIdx = aiCandid.second.get(i);
            String label = labelMap[acCandid.second.get(i)];
            if (!(label.equalsIgnoreCase("A0")
                    || label.equalsIgnoreCase("A1")
                    || label.equalsIgnoreCase("A2")
                    || label.equalsIgnoreCase("A3")
                    || label.equalsIgnoreCase("A4")))
                continue;

            if (wordIdx < pIdx || predicateSeen)
                seqOfCoreArgumentLabels += label + " ";
            else if (wordIdx > pIdx) {
                seqOfCoreArgumentLabels += pLabel + " " + label + " ";
                predicateSeen = true;
            }
        }
        if (predicateSeen == false)
            seqOfCoreArgumentLabels += pLabel;
        return new Object[]{seqOfCoreArgumentLabels.trim()};
    }

    private static String getDepSubCat(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                       int[] sentenceDepLabels, int[] posTags, IndexMap indexMap) throws Exception {
        StringBuilder subCat = new StringBuilder();
        if (sentenceReverseDepHeads[pIdx] != null) {
            //predicate has >1 children
            for (int child : sentenceReverseDepHeads[pIdx]) {
                String pos = indexMap.int2str(posTags[child]);
                if (!punctuations.contains(pos)) {
                    subCat.append(sentenceDepLabels[child]);
                    subCat.append("\t");
                }
            }
        }
        return subCat.toString().trim();
    }

    private static String getChildSet(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                      int[] collection, int[] posTags, IndexMap map) throws Exception {
        StringBuilder childSet = new StringBuilder();
        TreeSet<Integer> children = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            for (int child : sentenceReverseDepHeads[pIdx]) {
                String pos = map.int2str(posTags[child]);
                if (!punctuations.contains(pos)) {
                    children.add(collection[child]);
                }
            }
        }
        for (int child : children) {
            childSet.append(child);
            childSet.append("\t");
        }
        return childSet.toString().trim();
    }

    private static int getLeftMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null) {
            //this argument has at least one child
            int firstChild = sentenceReverseDepHeads[aIdx].first();
            // this should be on the left side.
            if (firstChild < aIdx) {
                return firstChild;
            }
        }
        return IndexMap.nullIdx;
    }

    private static int getRightMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null) {
            int last = sentenceReverseDepHeads[aIdx].last();
            if (last > aIdx) {
                return sentenceReverseDepHeads[aIdx].last();
            }
        }
        return IndexMap.nullIdx;
    }

    private static int getLeftSiblingIndex(int aIdx, int parIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[parIdx] != null) {
            argSiblings = sentenceReverseDepHeads[parIdx];
        }

        if (argSiblings.lower(aIdx) != null)
            return argSiblings.lower(aIdx);
        return IndexMap.nullIdx;
    }

    private static int getRightSiblingIndex(int aIdx, int parIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[parIdx] != null)
            argSiblings = sentenceReverseDepHeads[parIdx];

        if (argSiblings.higher(aIdx) != null)
            return argSiblings.higher(aIdx);
        return IndexMap.nullIdx;
    }

    private static Object[] addAllPredicateFeatures(BaseFeatureFields baseFeatureFields, Object[] currentFeatures,
                                                    int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pw) << 6 | label : baseFeatureFields.pw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.ppos) << 6 | label : baseFeatureFields.ppos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.plem) << 6 | label : baseFeatureFields.plem;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pdeprel) << 6 | label : baseFeatureFields.pdeprel;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pSense : baseFeatureFields.pSense;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pprw) << 6 | label : baseFeatureFields.pprw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pprpos) << 6 | label : baseFeatureFields.pprpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pdepsubcat : baseFeatureFields.pdepsubcat;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pchilddepset : baseFeatureFields.pchilddepset;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pchildposset : baseFeatureFields.pchildposset;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pchildwset : baseFeatureFields.pchildwset;
        return new Object[]{currentFeatures, index};
    }

    private static Object[] addAllArgumentFeatures(BaseFeatureFields baseFeatureFields, Object[] currentFeatures,
                                                   int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.aw << 6) | label : baseFeatureFields.aw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.apos << 6) | label : baseFeatureFields.apos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.adeprel << 6) | label : baseFeatureFields.adeprel;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.deprelpath : baseFeatureFields.deprelpath;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pospath : baseFeatureFields.pospath;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.position << 6) | label : baseFeatureFields.position;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftw << 6) | label : baseFeatureFields.leftw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftpos << 6) | label : baseFeatureFields.leftpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightw << 6) | label : baseFeatureFields.rightw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightpos << 6) | label : baseFeatureFields.rightpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftsiblingw << 6) | label : baseFeatureFields.leftsiblingw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftsiblingpos << 6) | label : baseFeatureFields.leftsiblingpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightsiblingw << 6) | label : baseFeatureFields.rightsiblingw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightsiblingpos << 6) | label : baseFeatureFields.rightsiblingpos;
        return new Object[]{currentFeatures, index};
    }

    private static Object[] addPredicateArgumentBigramFeatures(BaseFeatureFields baseFeatureFields, Object[] features,
                                                               int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        int pw = (extractGlobalFeatures) ? (baseFeatureFields.getPw() << 6) | label : baseFeatureFields.getPw();
        int ppos = (extractGlobalFeatures) ? (baseFeatureFields.getPpos() << 6) | label : baseFeatureFields.getPpos();
        int plem = (extractGlobalFeatures) ? (baseFeatureFields.getPlem() << 6) | label : baseFeatureFields.getPlem();
        String pSense = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getpSense() : baseFeatureFields.getpSense();
        int pdeprel = (extractGlobalFeatures) ? (baseFeatureFields.getPdeprel() << 6) | label : baseFeatureFields.getPdeprel();
        int pprw = (extractGlobalFeatures) ? (baseFeatureFields.getPprw() << 6) | label : baseFeatureFields.getPprw();
        int pprpos = (extractGlobalFeatures) ? (baseFeatureFields.getPprpos() << 6) | label : baseFeatureFields.getPprpos();
        String pdepsubcat = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPdepsubcat() : baseFeatureFields.getPdepsubcat();
        String pchilddepset = (extractGlobalFeatures) ? label + " " + label + " " + baseFeatureFields.getPchilddepset() : baseFeatureFields.getPchilddepset();
        String pchildposset = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPchildposset() : baseFeatureFields.getPchildposset();
        String pchildwset = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPchildwset() : baseFeatureFields.getPchildwset();
        int aw = baseFeatureFields.getAw();
        int apos = baseFeatureFields.getApos();
        int adeprel = baseFeatureFields.getAdeprel();
        String deprelpath = baseFeatureFields.getDeprelpath();
        String pospath = baseFeatureFields.getPospath();
        int position = baseFeatureFields.getPosition();
        int leftw = baseFeatureFields.getLeftw();
        int leftpos = baseFeatureFields.getLeftpos();
        int rightw = baseFeatureFields.getRightw();
        int rightpos = baseFeatureFields.getRightpos();
        int leftsiblingw = baseFeatureFields.getLeftsiblingw();
        int leftsiblingpos = baseFeatureFields.getLeftsiblingpos();
        int rightsiblingw = baseFeatureFields.getRightsiblingw();
        int rightsiblingpos = baseFeatureFields.getRightsiblingpos();

        // pw + argument features
        long pw_aw = (pw << 20) | aw;
        features[index++] = pw_aw;
        long pw_apos = (pw << 10) | apos;
        features[index++] = pw_apos;
        long pw_adeprel = (pw << 10) | adeprel;
        features[index++] = pw_adeprel;
        String pw_deprelpath = pw + " " + deprelpath;
        features[index++] = pw_deprelpath;
        String pw_pospath = pw + " " + pospath;
        features[index++] = pw_pospath;
        long pw_position = (pw << 2) | position;
        features[index++] = pw_position;
        long pw_leftw = (pw << 20) | leftw;
        features[index++] = pw_leftw;
        long pw_leftpos = (pw << 10) | leftpos;
        features[index++] = pw_leftpos;
        long pw_rightw = (pw << 20) | rightw;
        features[index++] = pw_rightw;
        long pw_rightpos = (pw << 10) | rightpos;
        features[index++] = pw_rightpos;
        long pw_leftsiblingw = (pw << 20) | leftsiblingw;
        features[index++] = pw_leftsiblingw;
        long pw_leftsiblingpos = (pw << 10) | leftsiblingpos;
        features[index++] = pw_leftsiblingpos;
        long pw_rightsiblingw = (pw << 20) | rightsiblingw;
        features[index++] = pw_rightsiblingw;
        long pw_rightsiblingpos = (pw << 10) | rightsiblingpos;
        features[index++] = pw_rightsiblingpos;

        //ppos + argument features
        long aw_ppos = (aw << 10) | ppos;
        features[index++] = aw_ppos;
        long ppos_apos = (ppos << 10) | apos;
        features[index++] = ppos_apos;
        long ppos_adeprel = (ppos << 10) | adeprel;
        features[index++] = ppos_adeprel;
        String ppos_deprelpath = ppos + " " + deprelpath;
        features[index++] = ppos_deprelpath;
        String ppos_pospath = ppos + " " + pospath;
        features[index++] = ppos_pospath;
        long ppos_position = (ppos << 2) | position;
        features[index++] = ppos_position;
        long leftw_ppos = (leftw << 10) | ppos;
        features[index++] = leftw_ppos;
        long ppos_leftpos = (ppos << 10) | leftpos;
        features[index++] = ppos_leftpos;
        long rightw_ppos = (rightw << 10) | ppos;
        features[index++] = rightw_ppos;
        long ppos_rightpos = (ppos << 10) | rightpos;
        features[index++] = ppos_rightpos;
        long leftsiblingw_ppos = (leftsiblingw << 10) | ppos;
        features[index++] = leftsiblingw_ppos;
        long ppos_leftsiblingpos = (ppos << 10) | leftsiblingpos;
        features[index++] = ppos_leftsiblingpos;
        long rightsiblingw_ppos = (rightsiblingw << 10) | ppos;
        features[index++] = rightsiblingw_ppos;
        long ppos_rightsiblingpos = (ppos << 10) | rightsiblingpos;
        features[index++] = ppos_rightsiblingpos;

        //pdeprel + argument features
        long aw_pdeprel = (aw << 10) | pdeprel;
        features[index++] = aw_pdeprel;
        long pdeprel_apos = (pdeprel << 10) | apos;
        features[index++] = pdeprel_apos;
        long pdeprel_adeprel = (pdeprel << 10) | adeprel;
        features[index++] = pdeprel_adeprel;
        String pdeprel_deprelpath = pdeprel + " " + deprelpath;
        features[index++] = pdeprel_deprelpath;
        String pdeprel_pospath = pdeprel + " " + pospath;
        features[index++] = pdeprel_pospath;
        long pdeprel_position = (pdeprel << 2) | position;
        features[index++] = pdeprel_position;
        long leftw_pdeprel = (leftw << 10) | pdeprel;
        features[index++] = leftw_pdeprel;
        long pdeprel_leftpos = (pdeprel << 10) | leftpos;
        features[index++] = pdeprel_leftpos;
        long rightw_pdeprel = (rightw << 10) | pdeprel;
        features[index++] = rightw_pdeprel;
        long pdeprel_rightpos = (pdeprel << 10) | rightpos;
        features[index++] = pdeprel_rightpos;
        long leftsiblingw_pdeprel = (leftsiblingw << 10) | pdeprel;
        features[index++] = leftsiblingw_pdeprel;
        long pdeprel_leftsiblingpos = (pdeprel << 10) | leftsiblingpos;
        features[index++] = pdeprel_leftsiblingpos;
        long rightsiblingw_pdeprel = (rightsiblingw << 10) | pdeprel;
        features[index++] = rightsiblingw_pdeprel;
        long pdeprel_rightsiblingpos = (pdeprel << 10) | rightsiblingpos;
        features[index++] = pdeprel_rightsiblingpos;


        //plem + argument features
        long aw_plem = (aw << 20) | plem;
        features[index++] = aw_plem;
        long plem_apos = (plem << 10) | apos;
        features[index++] = plem_apos;
        long plem_adeprel = (plem << 10) | adeprel;
        features[index++] = plem_adeprel;
        String plem_deprelpath = plem + " " + deprelpath;
        features[index++] = plem_deprelpath;
        String plem_pospath = plem + " " + pospath;
        features[index++] = plem_pospath;
        long plem_position = (plem << 2) | position;
        features[index++] = plem_position;
        long leftw_plem = (leftw << 20) | plem;
        features[index++] = leftw_plem;
        long plem_leftpos = (plem << 10) | leftpos;
        features[index++] = plem_leftpos;
        long rightw_plem = (rightw << 20) | plem;
        features[index++] = rightw_plem;
        long plem_rightpos = (plem << 10) | rightpos;
        features[index++] = plem_rightpos;
        long leftsiblingw_plem = (leftsiblingw << 20) | plem;
        features[index++] = leftsiblingw_plem;
        long plem_leftsiblingpos = (plem << 10) | leftsiblingpos;
        features[index++] = plem_leftsiblingpos;
        long rightsiblingw_plem = (rightsiblingw << 20) | plem;
        features[index++] = rightsiblingw_plem;
        long plem_rightsiblingpos = (plem << 10) | rightsiblingpos;
        features[index++] = plem_rightsiblingpos;

        //psense + argument features
        String psense_aw = pSense + " " + aw;
        features[index++] = psense_aw;
        String psense_apos = pSense + " " + apos;
        features[index++] = psense_apos;
        String psense_adeprel = pSense + " " + adeprel;
        features[index++] = psense_adeprel;
        String psense_deprelpath = pSense + " " + deprelpath;
        features[index++] = psense_deprelpath;
        String psense_pospath = pSense + " " + pospath;
        features[index++] = psense_pospath;
        String psense_position = pSense + " " + position;
        features[index++] = psense_position;
        String psense_leftw = pSense + " " + leftw;
        features[index++] = psense_leftw;
        String psense_leftpos = pSense + " " + leftpos;
        features[index++] = psense_leftpos;
        String psense_rightw = pSense + " " + rightw;
        features[index++] = psense_rightw;
        String psense_rightpos = pSense + " " + rightpos;
        features[index++] = psense_rightpos;
        String psense_leftsiblingw = pSense + " " + leftsiblingw;
        features[index++] = psense_leftsiblingw;
        String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
        features[index++] = psense_leftsiblingpos;
        String psense_rightsiblingw = pSense + " " + rightsiblingw;
        features[index++] = psense_rightsiblingw;
        String psense_rightsiblingpos = pSense + " " + rightsiblingpos;
        features[index++] = psense_rightsiblingpos;

        //pprw  + argument features
        long aw_pprw = (aw << 20) | pprw;
        features[index++] = aw_pprw;
        long pprw_apos = (pprw << 10) | apos;
        features[index++] = pprw_apos;
        long pprw_adeprel = (pprw << 10) | adeprel;
        features[index++] = pprw_adeprel;
        String pprw_deprelpath = pprw + " " + deprelpath;
        features[index++] = pprw_deprelpath;
        String pprw_pospath = pprw + " " + pospath;
        features[index++] = pprw_pospath;
        long pprw_position = (pprw << 2) | position;
        features[index++] = pprw_position;
        long leftw_pprw = (leftw << 20) | pprw;
        features[index++] = leftw_pprw;
        long pprw_leftpos = (pprw << 10) | leftpos;
        features[index++] = pprw_leftpos;
        long rightw_pprw = (rightw << 20) | pprw;
        features[index++] = rightw_pprw;
        long pprw_rightpos = (pprw << 10) | rightpos;
        features[index++] = pprw_rightpos;
        long leftsiblingw_pprw = (leftsiblingw << 20) | pprw;
        features[index++] = leftsiblingw_pprw;
        long pprw_leftsiblingpos = (pprw << 10) | leftsiblingpos;
        features[index++] = pprw_leftsiblingpos;
        long rightsiblingw_pprw = (rightsiblingw << 20) | pprw;
        features[index++] = rightsiblingw_pprw;
        long pprw_rightsiblingpos = (pprw << 10) | rightsiblingpos;
        features[index++] = pprw_rightsiblingpos;

        //pprpos + argument features
        long aw_pprpos = (aw << 10) | pprpos;
        features[index++] = aw_pprpos;
        long pprpos_apos = (pprpos << 10) | apos;
        features[index++] = pprpos_apos;
        long pprpos_adeprel = (pprpos << 10) | adeprel;
        features[index++] = pprpos_adeprel;
        String pprpos_deprelpath = pprpos + " " + deprelpath;
        features[index++] = pprpos_deprelpath;
        String pprpos_pospath = pprpos + " " + pospath;
        features[index++] = pprpos_pospath;
        long pprpos_position = (pprpos << 2) | position;
        features[index++] = pprpos_position;
        long leftw_pprpos = (leftw << 10) | pprpos;
        features[index++] = leftw_pprpos;
        long pprpos_leftpos = (pprpos << 10) | leftpos;
        features[index++] = pprpos_leftpos;
        long rightw_pprpos = (rightw << 10) | pprpos;
        features[index++] = rightw_pprpos;
        long pprpos_rightpos = (pprpos << 10) | rightpos;
        features[index++] = pprpos_rightpos;
        long leftsiblingw_pprpos = (leftsiblingw << 10) | pprpos;
        features[index++] = leftsiblingw_pprpos;
        long pprpos_leftsiblingpos = (pprpos << 10) | leftsiblingpos;
        features[index++] = pprpos_leftsiblingpos;
        long rightsiblingw_pprpos = (rightsiblingw << 10) | pprpos;
        features[index++] = rightsiblingw_pprpos;
        long pprpos_rightsiblingpos = (pprpos << 10) | rightsiblingpos;
        features[index++] = pprpos_rightsiblingpos;

        //pchilddepset + argument features
        String pchilddepset_aw = pchilddepset + " " + aw;
        features[index++] = pchilddepset_aw;
        String pchilddepset_apos = pchilddepset + " " + apos;
        features[index++] = pchilddepset_apos;
        String pchilddepset_adeprel = pchilddepset + " " + adeprel;
        features[index++] = pchilddepset_adeprel;
        String pchilddepset_deprelpath = pchilddepset + " " + deprelpath;
        features[index++] = pchilddepset_deprelpath;
        String pchilddepset_pospath = pchilddepset + " " + pospath;
        features[index++] = pchilddepset_pospath;
        String pchilddepset_position = pchilddepset + " " + position;
        features[index++] = pchilddepset_position;
        String pchilddepset_leftw = pchilddepset + " " + leftw;
        features[index++] = pchilddepset_leftw;
        String pchilddepset_leftpos = pchilddepset + " " + leftpos;
        features[index++] = pchilddepset_leftpos;
        String pchilddepset_rightw = pchilddepset + " " + rightw;
        features[index++] = pchilddepset_rightw;
        String pchilddepset_rightpos = pchilddepset + " " + rightpos;
        features[index++] = pchilddepset_rightpos;
        String pchilddepset_leftsiblingw = pchilddepset + " " + leftsiblingw;
        features[index++] = pchilddepset_leftsiblingw;
        String pchilddepset_leftsiblingpos = pchilddepset + " " + leftsiblingpos;
        features[index++] = pchilddepset_leftsiblingpos;
        String pchilddepset_rightsiblingw = pchilddepset + " " + rightsiblingw;
        features[index++] = pchilddepset_rightsiblingw;
        String pchilddepset_rightsiblingpos = pchilddepset + " " + rightsiblingpos;
        features[index++] = pchilddepset_rightsiblingpos;

        //pdepsubcat + argument features
        String pdepsubcat_aw = pdepsubcat + " " + aw;
        features[index++] = pdepsubcat_aw;
        String pdepsubcat_apos = pdepsubcat + " " + apos;
        features[index++] = pdepsubcat_apos;
        String pdepsubcat_adeprel = pdepsubcat + " " + adeprel;
        features[index++] = pdepsubcat_adeprel;
        String pdepsubcat_deprelpath = pdepsubcat + " " + deprelpath;
        features[index++] = pdepsubcat_deprelpath;
        String pdepsubcat_pospath = pdepsubcat + " " + pospath;
        features[index++] = pdepsubcat_pospath;
        String pdepsubcat_position = pdepsubcat + " " + position;
        features[index++] = pdepsubcat_position;
        String pdepsubcat_leftw = pdepsubcat + " " + leftw;
        features[index++] = pdepsubcat_leftw;
        String pdepsubcat_leftpos = pdepsubcat + " " + leftpos;
        features[index++] = pdepsubcat_leftpos;
        String pdepsubcat_rightw = pdepsubcat + " " + rightw;
        features[index++] = pdepsubcat_rightw;
        String pdepsubcat_rightpos = pdepsubcat + " " + rightpos;
        features[index++] = pdepsubcat_rightpos;
        String pdepsubcat_leftsiblingw = pdepsubcat + " " + leftsiblingw;
        features[index++] = pdepsubcat_leftsiblingw;
        String pdepsubcat_leftsiblingpos = pdepsubcat + " " + leftsiblingpos;
        features[index++] = pdepsubcat_leftsiblingpos;
        String pdepsubcat_rightsiblingw = pdepsubcat + " " + rightsiblingw;
        features[index++] = pdepsubcat_rightsiblingw;
        String pdepsubcat_rightsiblingpos = pdepsubcat + " " + rightsiblingpos;
        features[index++] = pdepsubcat_rightsiblingpos;

        //pchildposset + argument features
        String pchildposset_aw = pchildposset + " " + aw;
        features[index++] = pchildposset_aw;
        String pchildposset_apos = pchildposset + " " + apos;
        features[index++] = pchildposset_apos;
        String pchildposset_adeprel = pchildposset + " " + adeprel;
        features[index++] = pchildposset_adeprel;
        String pchildposset_deprelpath = pchildposset + " " + deprelpath;
        features[index++] = pchildposset_deprelpath;
        String pchildposset_pospath = pchildposset + " " + pospath;
        features[index++] = pchildposset_pospath;
        String pchildposset_position = pchildposset + " " + position;
        features[index++] = pchildposset_position;
        String pchildposset_leftw = pchildposset + " " + leftw;
        features[index++] = pchildposset_leftw;
        String pchildposset_leftpos = pchildposset + " " + leftpos;
        features[index++] = pchildposset_leftpos;
        String pchildposset_rightw = pchildposset + " " + rightw;
        features[index++] = pchildposset_rightw;
        String pchildposset_rightpos = pchildposset + " " + rightpos;
        features[index++] = pchildposset_rightpos;
        String pchildposset_leftsiblingw = pchildposset + " " + leftsiblingw;
        features[index++] = pchildposset_leftsiblingw;
        String pchildposset_leftsiblingpos = pchildposset + " " + leftsiblingpos;
        features[index++] = pchildposset_leftsiblingpos;
        String pchildposset_rightsiblingw = pchildposset + " " + rightsiblingw;
        features[index++] = pchildposset_rightsiblingw;
        String pchildposset_rightsiblingpos = pchildposset + " " + rightsiblingpos;
        features[index++] = pchildposset_rightsiblingpos;

        //pchildwset + argument features
        String pchildwset_aw = pchildwset + " " + aw;
        features[index++] = pchildwset_aw;
        String pchildwset_apos = pchildwset + " " + apos;
        features[index++] = pchildwset_apos;
        String pchildwset_adeprel = pchildwset + " " + adeprel;
        features[index++] = pchildwset_adeprel;
        String pchildwset_deprelpath = pchildwset + " " + deprelpath;
        features[index++] = pchildwset_deprelpath;
        String pchildwset_pospath = pchildwset + " " + pospath;
        features[index++] = pchildwset_pospath;
        String pchildwset_position = pchildwset + " " + position;
        features[index++] = pchildwset_position;
        String pchildwset_leftw = pchildwset + " " + leftw;
        features[index++] = pchildwset_leftw;
        String pchildwset_leftpos = pchildwset + " " + leftpos;
        features[index++] = pchildwset_leftpos;
        String pchildwset_rightw = pchildwset + " " + rightw;
        features[index++] = pchildwset_rightw;
        String pchildwset_rightpos = pchildwset + " " + rightpos;
        features[index++] = pchildwset_rightpos;
        String pchildwset_leftsiblingw = pchildwset + " " + leftsiblingw;
        features[index++] = pchildwset_leftsiblingw;
        String pchildwset_leftsiblingpos = pchildwset + " " + leftsiblingpos;
        features[index++] = pchildwset_leftsiblingpos;
        String pchildwset_rightsiblingw = pchildwset + " " + rightsiblingw;
        features[index++] = pchildwset_rightsiblingw;
        String pchildwset_rightsiblingpos = pchildwset + " " + rightsiblingpos;
        features[index++] = pchildwset_rightsiblingpos;
        return new Object[]{features, index};
    }

    private static class BaseFeatureFields {
        private int pIdx;
        private int aIdx;
        private Sentence sentence;
        private IndexMap indexMap;
        private int pw;
        private int ppos;
        private int plem;
        private String pSense;
        private int pdeprel;
        private int pprw;
        private int pprpos;
        private String pdepsubcat;
        private String pchilddepset;
        private String pchildposset;
        private String pchildwset;
        private int aw;
        private int apos;
        private int adeprel;
        private String deprelpath;
        private String pospath;
        private int position;
        private int leftw;
        private int leftpos;
        private int rightw;
        private int rightpos;
        private int rightsiblingw;
        private int rightsiblingpos;
        private int leftsiblingw;
        private int leftsiblingpos;


        public BaseFeatureFields(int pIdx, int aIdx, Sentence sentence, IndexMap indexMap) {
            this.pIdx = pIdx;
            this.aIdx = aIdx;
            this.sentence = sentence;
            this.indexMap = indexMap;
        }

        public int getPw() {
            return pw;
        }

        public int getPpos() {
            return ppos;
        }

        public int getPlem() {
            return plem;
        }

        public String getpSense() {
            return pSense;
        }

        public int getPdeprel() {
            return pdeprel;
        }

        public int getPprw() {
            return pprw;
        }

        public int getPprpos() {
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

        public int getAw() {
            return aw;
        }

        public int getApos() {
            return apos;
        }

        public int getAdeprel() {
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

        public int getLeftw() {
            return leftw;
        }

        public int getLeftpos() {
            return leftpos;
        }

        public int getRightw() {
            return rightw;
        }

        public int getRightpos() {
            return rightpos;
        }

        public int getRightsiblingw() {
            return rightsiblingw;
        }

        public int getRightsiblingpos() {
            return rightsiblingpos;
        }

        public int getLeftsiblingw() {
            return leftsiblingw;
        }

        public int getLeftsiblingpos() {
            return leftsiblingpos;
        }

        public BaseFeatureFields invoke() throws Exception {
            int[] sentenceDepLabels = sentence.getDepLabels();
            int[] sentenceDepHeads = sentence.getDepHeads();
            int[] sentenceWords = sentence.getWords();
            int[] sentencePOSTags = sentence.getPosTags();
            int[] sentenceLemmas = sentence.getLemmas();
            TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();
            HashMap<Integer, String> sentencePredicatesInfo = sentence.getPredicatesAutoLabelMap();

            //predicate features
            pw = sentenceWords[pIdx];
            ppos = sentencePOSTags[pIdx];
            plem = sentenceLemmas[pIdx];
            pSense = sentencePredicatesInfo.get(pIdx);
            pdeprel = sentenceDepLabels[pIdx];
            pprw = sentenceWords[sentenceDepHeads[pIdx]];
            pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
            pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
            pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
            pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
            pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

            int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int parIndex = sentenceDepHeads[aIdx];
            int lefSiblingIndex = getLeftSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);
            int rightSiblingIndex = getRightSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);

            //argument features
            aw = sentenceWords[aIdx];
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

            leftw = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[leftMostDependentIndex];
            leftpos = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[leftMostDependentIndex];
            rightw = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[rightMostDependentIndex];
            rightpos = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[rightMostDependentIndex];
            rightsiblingw = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[rightSiblingIndex];
            rightsiblingpos = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[rightSiblingIndex];
            leftsiblingw = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[lefSiblingIndex];
            leftsiblingpos = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[lefSiblingIndex];
            return this;
        }
    }
}