package SentenceStructures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Created by Maryam Aminian on 12/9/15.
 */
public class Sentence {
    int[] depHeads;
    String[] depLabels;
    String[] words;
    String[] wordClusterIds;
    String[] posTags;
    String[] cPosTags;
    String[] lemmas;
    String[] lemmas_str;
    TreeSet<Integer>[] reverseDepHeads;
    PAs predicateArguments;


    public Sentence(String sentence) {
        String[] tokens = sentence.trim().split("\n");

        int numTokens = tokens.length + 1; //add one more token for ROOT
        int predicatesSeq = -1;

        depHeads = new int[numTokens];
        depHeads[0] = -1;
        depLabels = new String[numTokens];
        depLabels[0] = "";
        words = new String[numTokens];
        words[0] = "ROOT";
        posTags = new String[numTokens];
        posTags[0] = words[0];
        cPosTags = new String[numTokens];
        cPosTags[0] = words[0];
        lemmas = new String[numTokens];
        lemmas[0] = words[0];
        lemmas_str = new String[numTokens];
        lemmas_str[0] = "ROOT";
        wordClusterIds = new String[numTokens];
        wordClusterIds[0] = "_";

        reverseDepHeads = new TreeSet[numTokens];
        predicateArguments = new PAs();

        for (int tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {
            String token = tokens[tokenIdx];
            String[] fields = token.split("\t");

            int index = Integer.parseInt(fields[0]);
            int depHead = Integer.parseInt(fields[9]);
            depHeads[index] = depHead;

            words[index] = fields[1];
            wordClusterIds[index] = fields[1];
            depLabels[index] = fields[11];
            posTags[index] = fields[5];
            cPosTags[index] = util.StringUtils.getCoarsePOS(fields[5]);
            lemmas[index] = fields[3];

            if (reverseDepHeads[depHead] == null) {
                TreeSet<Integer> children = new TreeSet<Integer>();
                children.add(index);
                reverseDepHeads[depHead] = children;
            } else
                reverseDepHeads[depHead].add(index);

            //setPredicate predicate information
            String predicate = "_";
            if (!fields[13].equals("_")) {
                predicatesSeq++;
                predicate = fields[13];
                predicateArguments.setPredicate(predicatesSeq, index, predicate);
            }

            if (fields.length > 14) //we have at least one argument
            {
                for (int i = 14; i < fields.length; i++) {
                    if (!fields[i].equals("_")) //found an argument
                    {
                        String argumentType = fields[i];
                        int associatedPredicateSeq = i - 14;
                        predicateArguments.setArgument(associatedPredicateSeq, index, argumentType);
                    }
                }
            }
        }
    }

    public ArrayList<String> getDepPath(int source, int target) {
        String right = ">>";
        String left = "<<";
        ArrayList<String> visited = new ArrayList<>();

        if (source != target) {
            if (reverseDepHeads[source] != null) {
                //source has some children
                for (int child : reverseDepHeads[source]) {
                    if (child == target) {
                        if (child > source) {
                            visited.add(depLabels[child]+"|"+ right);
                        } else {
                            visited.add(depLabels[child] +"|"+ left);
                        }
                        break;
                    } else {
                        if (child > source) {
                            visited.add(depLabels[child] +"|"+  right);
                        } else {
                            visited.add(depLabels[child] +"|"+  left);
                        }
                        ArrayList<String> visitedFromThisChild = getDepPath(child, target);
                        if (visitedFromThisChild.size() != 0) {
                            visited.addAll(visitedFromThisChild);
                            break;
                        } else
                            visited.clear();
                    }
                }
            } else {
                //source does not have any children + we have not still met the target --> there is no path between source and target
                visited.clear();
            }
        }
        return visited;
    }


    public ArrayList<String> getPOSPath(int source, int target) {
        String right = ">>";
        String left = "<<";
        ArrayList<String> visited = new ArrayList<>();

        if (source != target) {
            if (reverseDepHeads[source] != null) {
                //source has some children
                for (int child : reverseDepHeads[source]) {
                    if (child == target) {
                        if (child > source) {
                            visited.add(right);
                        } else {
                            visited.add(left);
                        }
                        break;
                    } else {
                        if (child > source) {
                            visited.add(posTags[child] +"|"+ right);
                        } else {
                            visited.add(posTags[child] +"|"+ left);
                        }
                        ArrayList<String> visitedFromThisChild = getPOSPath(child, target);
                        if (visitedFromThisChild.size() != 0) {
                            visited.addAll(visitedFromThisChild);
                            break;
                        } else
                            visited.clear();
                    }
                }
            } else {
                //source does not have any children + we have not still met the target --> there is no path between source and target
                visited.clear();
            }
        }
        return visited;
    }


    public PAs getPredicateArguments() {
        return predicateArguments;
    }


    public String[] getPosTags() {
        return posTags;
    }


    public String[] getCPosTags() {
        return cPosTags;
    }

    public int[] getDepHeads() {
        return depHeads;
    }

    public String[] getLemmas_str() {
        return lemmas_str;
    }

    public String[] getDepHeads_as_str() {
        String[] depHeads_str = new String[depHeads.length];
        for (int i = 0; i < depHeads.length; i++)
            depHeads_str[i] = Integer.toString(depHeads[i]);
        return depHeads_str;
    }

    public String[] getWords() {
        return words;
    }

    public String[] getDepLabels() {
        return depLabels;
    }

    public String[] getLemmas() {
        return lemmas;
    }

    public String[] getWordClusterIds() {return wordClusterIds;}

    public TreeSet<Integer>[] getReverseDepHeads() {
        return reverseDepHeads;
    }

    public HashMap<Integer, String> getPredicatesInfo() {
        HashMap<Integer, String> predicatesInfo = new HashMap<Integer, String>();
        for (PA pa : predicateArguments.getPredicateArgumentsAsArray())
            predicatesInfo.put(pa.getPredicateIndex(), pa.getPredicateLabel());

        return predicatesInfo;
    }
}
