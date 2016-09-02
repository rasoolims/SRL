package util;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by monadiab on 5/17/16.
 */
public class StringUtils {

    // todo
    public static String convertPathArrayIntoString(ArrayList<String> depPathArray) {
        // todo StringBuilder
        //StringBuilder depPath= new StringBuilder();
        String depPath = "";
        for (String dep : depPathArray) {
            //depPath.append(dep);
            //depPath.append("\t");
            depPath += dep + "\t";
        }
        //todo find .replaceAll("\t","_") in all occurrences and remove them!
        return depPath.trim();
    }

    public static String join(Collection<String> collection, String del) {
        //StringBuilder output= new StringBuilder();
        String output = "";
        for (String element : collection) {
            //output.append(element);
            //output.append("\t");
            output += element + "\t";
        }
        //todo find .replaceAll("\t",del) in all occurrences and remove them!
        return output.trim();
    }

    public static String getCoarsePOS(String originalPOS) {
        String coarsePOS = originalPOS;
        if (originalPOS.startsWith("JJ"))
            coarsePOS = "JJ"; //covers JJ, JJR, JJS
        else if (originalPOS.startsWith("NN"))
            coarsePOS = "NN"; //covers NN, NNS, NNP, NNPS
        else if (originalPOS.startsWith("PR"))
            coarsePOS = "PR"; //covers PRP, PRP$
        else if (originalPOS.startsWith("RB"))
            coarsePOS = "RB"; //covers RB, RBR, RBS
        else if (originalPOS.startsWith("VB"))
            coarsePOS = "VB"; //covers VB, VBD, VBG, VBN, VBP, VBZ
        return coarsePOS;
    }
}
