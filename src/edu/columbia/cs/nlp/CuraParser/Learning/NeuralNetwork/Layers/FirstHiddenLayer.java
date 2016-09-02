package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 7:50 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import SupervisedSRL.Strcutures.NNIndexMaps;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Activation;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.FixInit;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.Initializer;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * This class shows the first hidden layer layer which its input is a concatenation of different embedding layer.
 */
public class FirstHiddenLayer extends Layer {
    WordEmbeddingLayer wordEmbeddings;
    EmbeddingLayer posEmbeddings;
    EmbeddingLayer depEmbeddings;
    EmbeddingLayer subcatEmbeddings;
    EmbeddingLayer depPathEmbeddings;
    EmbeddingLayer posPathEmbeddings;
    EmbeddingLayer positionEmbeddings;

    int numWordLayers;
    int numPosLayers;
    int numDepLayers;
    int numSubcatLayers;
    int numDepPathLayers;
    int numPosPathLayers;
    int numPositionLayers;

    // pre-computed items.
    private double[][][] saved;

    public FirstHiddenLayer(Activation activation, int nIn, int nOut, Initializer initializer, Initializer biasInit,
                            NNIndexMaps nnIndexMaps,
                            Random random, HashMap<Integer, Integer>[] precomputationMap,
                            int wDim, int posDim, int depDim, int subcatDim, int depPathDim, int posPathDim, int positionDim,
                            HashMap<Integer, double[]> embeddingsDictionary) {
        super(activation, nIn, nOut, initializer, biasInit);
        this.wordEmbeddings = new WordEmbeddingLayer(wDim, nnIndexMaps.wordFeatMap.size()+2, random, precomputationMap);
        this.posEmbeddings = new EmbeddingLayer(posDim, nnIndexMaps.posFeatMap.size()+2, random);
        this.depEmbeddings = new EmbeddingLayer(depDim, nnIndexMaps.depFeatMap.size()+2, random);
        this.subcatEmbeddings = new EmbeddingLayer(subcatDim, nnIndexMaps.subcatFeatMap.size()+2, random);
        this.depPathEmbeddings = new EmbeddingLayer(depPathDim, nnIndexMaps.depPathFeatMap.size()+2, random);
        this.posPathEmbeddings = new EmbeddingLayer(posPathDim, nnIndexMaps.posPathFeatMap.size()+2, random);
        this.positionEmbeddings = new EmbeddingLayer(positionDim, 3, random);

        this.numWordLayers = nnIndexMaps.numWordFeatures;
        this.numPosLayers = nnIndexMaps.numPosFeatures;
        this.numDepLayers = nnIndexMaps.numDepFeatures;
        this.numSubcatLayers=nnIndexMaps.numSubcatFeatures;
        this.numDepPathLayers=nnIndexMaps.numDepPathFeatures;
        this.numPosPathLayers=nnIndexMaps.numPosPathFeatures;
        this.numPositionLayers=1;

        this.wordEmbeddings.addPretrainedVectors(embeddingsDictionary);
        if (getPrecomputationMap() != null) {
            assert wordEmbeddings.numOfEmbeddingSlot() == numWordLayers;
            preCompute();
        }
    }


    public FirstHiddenLayer(Activation activation, int nIn, int nOut, Initializer initializer, Initializer biasInit,
                            int numWordLayers, int numPosLayers, int numDepLayers, int numSubcatLayers, int numDepPathLayers,
                            int numPosPathLayers,  int numPositionLayers,
                            Random random, HashMap<Integer, Integer>[] precomputationMap,
                            int numW, int numPos, int numDep, int numSubcat, int numDepPath, int numPosPath,
                            int wDim, int posDim, int depDim, int subcatDim, int depPathDim, int posPathDim, int positionDim,
                            HashMap<Integer, double[]> embeddingsDictionary) {
        super(activation, nIn, nOut, initializer, biasInit);

        this.wordEmbeddings = new WordEmbeddingLayer(wDim, numW, random, precomputationMap);
        this.posEmbeddings = new EmbeddingLayer(posDim, numPos, random);
        this.depEmbeddings = new EmbeddingLayer(depDim, numDep, random);
        this.subcatEmbeddings = new EmbeddingLayer(subcatDim, numSubcat, random);
        this.depPathEmbeddings = new EmbeddingLayer(depPathDim, numDepPath, random);
        this.posPathEmbeddings = new EmbeddingLayer(posPathDim,numPosPath, random);
        this.positionEmbeddings = new EmbeddingLayer(positionDim, 3, random);

        this.numWordLayers = numWordLayers;
        this.numPosLayers = numPosLayers;
        this.numDepLayers =numDepLayers;
        this.numSubcatLayers=numSubcatLayers;
        this.numDepPathLayers=numDepPathLayers;
        this.numPosPathLayers=numPosPathLayers;
        this.numPositionLayers=numPositionLayers;

        this.wordEmbeddings.addPretrainedVectors(embeddingsDictionary);
        if (getPrecomputationMap() != null) {
            preCompute();
        }
    }

    public void preCompute() {
        saved = new double[numWordLayers + numPosLayers + numDepLayers][][];
        for (int i = 0; i < numWordLayers; i++)
            saved[i] = new double[wordEmbeddings.numOfPrecomputedItems(i)][nOut()];
        for (int i = numWordLayers; i < numWordLayers + numPosLayers; i++)
            saved[i] = new double[posEmbeddings.vocabSize()][nOut()];
        for (int i = numWordLayers + numPosLayers; i < numWordLayers + numPosLayers + numDepLayers; i++)
            saved[i] = new double[depEmbeddings.vocabSize()][nOut()];

        int offset = 0;
        for (int pos = 0; pos < numWordLayers; pos++) {
            for (int tok : wordEmbeddings.preComputedIds(pos)) {
                int id = wordEmbeddings.preComputeId(pos, tok);
                for (int h = 0; h < nOut(); h++) {
                    for (int k = 0; k < wordEmbeddings.dim(); k++) {
                        saved[pos][id][h] += w[h][offset + k] * wordEmbeddings.w(tok, k);
                    }
                }
            }
            offset += wordEmbeddings.dim();
        }

        for (int pos = 0; pos < numPosLayers; pos++) {
            for (int tok = 0; tok < posEmbeddings.vocabSize(); tok++) {
                int indOffset = numWordLayers;
                for (int h = 0; h < nOut(); h++) {
                    for (int k = 0; k < posEmbeddings.dim(); k++) {
                        saved[pos + indOffset][tok][h] += w[h][offset + k] * posEmbeddings.w(tok, k);
                    }
                }
            }
            offset += posEmbeddings.dim();
        }

        for (int pos = 0; pos < numDepLayers; pos++) {
            for (int tok = 0; tok < depEmbeddings.vocabSize(); tok++) {
                int indOffset = numWordLayers + numPosLayers;
                for (int h = 0; h < nOut(); h++) {
                    for (int k = 0; k < depEmbeddings.dim(); k++) {
                        saved[pos + indOffset][tok][h] += w[h][offset + k] * depEmbeddings.w(tok, k);
                    }
                }
            }
            offset += depEmbeddings.dim();
        }
    }

    /**
     * Uses pre-computed maps in order to speed things up.
     *
     * @param input
     * @return
     */
    @Override
    public double[] forward(double[] input) {
        int offset = 0;
        double[] hidden = new double[nOut()];
        for (int j = 0; j < input.length; j++) {
            int tok = (int) input[j];
            EmbeddingLayer embedding;
            if (j < numWordLayers)
                embedding = wordEmbeddings;
            else if (j < numWordLayers + numPosLayers)
                embedding = posEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers)
                embedding = depEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers + numSubcatLayers)
                embedding = subcatEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers + numSubcatLayers + numDepPathLayers)
                embedding = depPathEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers + numSubcatLayers + numDepPathLayers + numPosPathLayers)
                embedding = posPathEmbeddings;
            else embedding = positionEmbeddings;

            if (saved != null && ((j >= numWordLayers && j<numWordLayers+numDepLayers+numPosLayers) || wordEmbeddings.isFrequent(j, tok))) {
                int id = tok;
                if (j < numWordLayers)
                    id = wordEmbeddings.preComputeId(j, tok);
                double[] s = saved[j][id];
                for (int i = 0; i < hidden.length; i++) {
                    hidden[i] += s[i];
                }
            } else {
                for (int i = 0; i < hidden.length; i++) {
                    for (int k = 0; k < embedding.dim(); k++) {
                        hidden[i] += w[i][offset + k] * embedding.w(tok, k);
                    }
                }
            }
            offset += embedding.dim();
        }

        Utils.sumi(hidden, b);

        return hidden;
    }

    @Override
    public double[] forward(double[] input, HashSet<Integer> hiddenToUSe) {
        int offset = 0;
        double[] hidden = new double[nOut()];
        for (int j = 0; j < input.length; j++) {
            int tok = (int) input[j];
            EmbeddingLayer embedding;
            if (j < numWordLayers)
                embedding = wordEmbeddings;
            else if (j < numWordLayers + numPosLayers)
                embedding = posEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers)
                embedding = depEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers + numSubcatLayers)
                embedding = subcatEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers + numSubcatLayers + numDepPathLayers)
                embedding = depPathEmbeddings;
            else if (j < numWordLayers + numPosLayers+ numDepLayers + numSubcatLayers + numDepPathLayers + numPosPathLayers)
                embedding = posPathEmbeddings;
            else embedding = positionEmbeddings;

            if (saved != null && ((j >= numWordLayers && j < numWordLayers + numDepLayers + numPosLayers)
                    || (j < numWordLayers && wordEmbeddings.isFrequent(j, tok)))) {
                int id = tok;
                if (j < numWordLayers)
                    id = wordEmbeddings.preComputeId(j, tok);
                Utils.sumi(hidden, saved[j][id], hiddenToUSe);
            } else {
                for (int i : hiddenToUSe) {
                    for (int k = 0; k < embedding.dim(); k++) {
                            hidden[i] += w[i][offset + k] * embedding.w(tok, k);
                    }
                }
            }
            offset += embedding.dim();
        }
        Utils.sumi(hidden, b);

        return hidden;
    }

    @Override
    public double[] backward(final double[] delta, int layerIndex, double[] hInput, double[] prevH, double[] activations, HashSet<Integer>[]
            featuresSeen, double[][][] savedGradients, MLPNetwork network) {
        assert layerIndex == 0;
        final double[][] nextW = network.layer(layerIndex + 1).getW();
        final double[][] curW = network.layer(layerIndex).getW();
        WordEmbeddingLayer netWordEmbeddings = ((FirstHiddenLayer) network.layer(layerIndex)).wordEmbeddings;
        EmbeddingLayer netSubcatEmbeddings = ((FirstHiddenLayer) network.layer(layerIndex)).subcatEmbeddings;
        EmbeddingLayer netDepPathEmbeddings = ((FirstHiddenLayer) network.layer(layerIndex)).depPathEmbeddings;
        EmbeddingLayer netPosPathEmbeddings = ((FirstHiddenLayer) network.layer(layerIndex)).posPathEmbeddings;
        EmbeddingLayer netPositionEmbeddings = ((FirstHiddenLayer) network.layer(layerIndex)).positionEmbeddings;

        int offset = 0;
        double[][] wordEmbeddings = this.wordEmbeddings.getW();
        double[][] subcatEmbeddings = this.subcatEmbeddings.getW();
        double[][] depPathEmbeddings = this.depPathEmbeddings.getW();
        double[][] posPathEmbeddings = this.posPathEmbeddings.getW();
        double[][] positionEmbeddings = this.positionEmbeddings.getW();

        double[] newDelta = activation.gradient(hInput, Utils.dotTranspose(nextW, delta), activations, false);
        assert newDelta.length == w.length;
        Utils.sumi(b, newDelta);

        for (int index = 0; index < numWordLayers; index++) {
            int tok = (int) prevH[index];
            if (netWordEmbeddings.isFrequent(index, tok)) {
                featuresSeen[index].add(tok);
                int id = netWordEmbeddings.preComputeId(index, tok);
                Utils.sumi(savedGradients[index][id], newDelta);
            } else {
                double[] embeddings = netWordEmbeddings.w(tok);
                for (int h = 0; h < w.length; h++) {
                    for (int k = 0; k < embeddings.length; k++) {
                        w[h][offset + k] += newDelta[h] * embeddings[k];
                        wordEmbeddings[tok][k] += newDelta[h] * curW[h][offset + k];
                    }
                }
            }
            offset += netWordEmbeddings.dim();
        }

        for (int index = numWordLayers; index < numWordLayers + numPosLayers; index++) {
            int tok = (int) prevH[index];
            Utils.sumi(savedGradients[index][tok], newDelta);
            offset += posEmbeddings.dim();
        }

        for (int index = numWordLayers + numPosLayers; index < numWordLayers + numPosLayers + numDepLayers; index++) {
            int tok = (int) prevH[index];
            Utils.sumi(savedGradients[index][tok], newDelta);
            offset += depEmbeddings.dim();
        }

        for (int index = numWordLayers + numPosLayers + numDepLayers; index <  numWordLayers + numPosLayers + numDepLayers+ numSubcatLayers; index++) {
            int tok = (int) prevH[index];
            double[] embeddings = netSubcatEmbeddings.w(tok);
            for (int h = 0; h < w.length; h++) {
                for (int k = 0; k < embeddings.length; k++) {
                    w[h][offset + k] += newDelta[h] * embeddings[k];
                    subcatEmbeddings[tok][k] += newDelta[h] * curW[h][offset + k];
                }
            }
            offset += netSubcatEmbeddings.dim();
        }

        for (int index = numWordLayers + numPosLayers + numDepLayers + numSubcatLayers;
             index <  numWordLayers + numPosLayers + numDepLayers+ numSubcatLayers + numDepPathLayers; index++) {
            int tok = (int) prevH[index];
            double[] embeddings = netDepPathEmbeddings.w(tok);
            for (int h = 0; h < w.length; h++) {
                for (int k = 0; k < embeddings.length; k++) {
                    w[h][offset + k] += newDelta[h] * embeddings[k];
                    depPathEmbeddings[tok][k] += newDelta[h] * curW[h][offset + k];
                }
            }
            offset += netDepPathEmbeddings.dim();
        }

        for (int index = numWordLayers + numPosLayers + numDepLayers + numSubcatLayers + numDepPathLayers;
             index <  numWordLayers + numPosLayers + numDepLayers+ numSubcatLayers + numDepPathLayers + numPosPathLayers; index++) {
            int tok = (int) prevH[index];
            double[] embeddings = netPosPathEmbeddings.w(tok);
            for (int h = 0; h < w.length; h++) {
                for (int k = 0; k < embeddings.length; k++) {
                    w[h][offset + k] += newDelta[h] * embeddings[k];
                    posPathEmbeddings[tok][k] += newDelta[h] * curW[h][offset + k];
                }
            }
            offset += netPosPathEmbeddings.dim();
        }

        for (int index = numWordLayers + numPosLayers + numDepLayers + numSubcatLayers + numDepPathLayers+ numPosPathLayers;
             index <  numWordLayers + numPosLayers + numDepLayers+ numSubcatLayers + numDepPathLayers + numPosPathLayers+ numPositionLayers; index++) {
            int tok = (int) prevH[index];
            double[] embeddings = netPositionEmbeddings.w(tok);
            for (int h = 0; h < w.length; h++) {
                for (int k = 0; k < embeddings.length; k++) {
                    w[h][offset + k] += newDelta[h] * embeddings[k];
                    positionEmbeddings[tok][k] += newDelta[h] * curW[h][offset + k];
                }
            }
            offset += netPositionEmbeddings.dim();
        }

        // does not need to back-propagate anymore.
        return newDelta;
    }

    @Override
    public Layer clone() {
        return copy(false, true);
    }

    @Override
    public Layer copy(boolean zeroOut, boolean deepCopy) {
        FirstHiddenLayer layer = new FirstHiddenLayer(activation, w[0].length, w.length, new FixInit(0), new FixInit(0),
                numWordLayers, numPosLayers, numDepLayers, numSubcatLayers, numDepPathLayers, numPosPathLayers, numPositionLayers,
                new Random(), null,
                wordEmbeddings.vocabSize(), posEmbeddings.vocabSize(), depEmbeddings.vocabSize(),
                subcatEmbeddings.vocabSize(), depPathEmbeddings.vocabSize(), posPathEmbeddings.vocabSize(),
                wordEmbeddings.dim(), posEmbeddings.dim(), depEmbeddings.dim(),
                subcatEmbeddings.dim(), depPathEmbeddings.dim(), posPathEmbeddings.dim(),  posEmbeddings.dim(),
                null);
        if (!zeroOut) {
            layer.setW(deepCopy ? Utils.clone(w) : w);
            layer.setB(deepCopy ? Utils.clone(b) : b);

            layer.getWordEmbeddings().setW(deepCopy ? Utils.clone(wordEmbeddings.getW()) : wordEmbeddings.getW());
            layer.getPosEmbeddings().setW(deepCopy ? Utils.clone(posEmbeddings.getW()) : posEmbeddings.getW());
            layer.getDepEmbeddings().setW(deepCopy ? Utils.clone(depEmbeddings.getW()) : depEmbeddings.getW());
            layer.getSubcatEmbeddings().setW(deepCopy ? Utils.clone(subcatEmbeddings.getW()) : subcatEmbeddings.getW());
            layer.getDepPathEmbeddings().setW(deepCopy ? Utils.clone(depPathEmbeddings.getW()) : depPathEmbeddings.getW());
            layer.getPosPathEmbeddings().setW(deepCopy ? Utils.clone(posPathEmbeddings.getW()) : posPathEmbeddings.getW());
            layer.getPositionEmbeddings().setW(deepCopy ? Utils.clone(positionEmbeddings.getW()) : positionEmbeddings.getW());
        } else {
            layer.getWordEmbeddings().setW(new double[wordEmbeddings.nOut()][wordEmbeddings.nIn()]);
            layer.getPosEmbeddings().setW(new double[posEmbeddings.nOut()][posEmbeddings.nIn()]);
            layer.getDepEmbeddings().setW(new double[depEmbeddings.nOut()][depEmbeddings.nIn()]);
            layer.getSubcatEmbeddings().setW(new double[subcatEmbeddings.nOut()][subcatEmbeddings.nIn()]);
            layer.getDepPathEmbeddings().setW(new double[depPathEmbeddings.nOut()][depPathEmbeddings.nIn()]);
            layer.getPosPathEmbeddings().setW(new double[posPathEmbeddings.nOut()][posPathEmbeddings.nIn()]);
            layer.getPositionEmbeddings().setW(new double[positionEmbeddings.nOut()][positionEmbeddings.nIn()]);
        }
        return layer;
    }

    public void modify(EmbeddingTypes type, int i, int j, double change) {
        if (type == EmbeddingTypes.WORD)
            wordEmbeddings.modifyW(i, j, change);
        else if (type == EmbeddingTypes.POS)
            posEmbeddings.modifyW(i, j, change);
        else if (type == EmbeddingTypes.DEPENDENCY)
            depEmbeddings.modifyW(i, j, change);
        else if (type == EmbeddingTypes.SUBCAT)
            subcatEmbeddings.modifyW(i, j, change);
        else if (type == EmbeddingTypes.DEPPATH)
            depPathEmbeddings.modifyW(i, j, change);
        else if (type == EmbeddingTypes.POSPATH)
            posPathEmbeddings.modifyW(i, j, change);
        else if (type == EmbeddingTypes.POSITION)
            positionEmbeddings.modifyW(i, j, change);
        else if (type == EmbeddingTypes.HIDDENLAYER)
            w[i][j] += change;
        else if (type == EmbeddingTypes.HIDDENLAYERBIAS) {
            assert j == -1;
            b[i] += change;
        } else
            throw new NotImplementedException();
    }

    public WordEmbeddingLayer getWordEmbeddings() {
        return wordEmbeddings;
    }

    public EmbeddingLayer getPosEmbeddings() {
        return posEmbeddings;
    }

    public EmbeddingLayer getDepEmbeddings() {
        return depEmbeddings;
    }

    public ArrayList<EmbeddingLayer> embeddingLayers(){
        ArrayList<EmbeddingLayer> layers = new ArrayList<>();
        layers.add(wordEmbeddings);
        layers.add(posEmbeddings);
        layers.add(depEmbeddings);
        layers.add(subcatEmbeddings);
        layers.add(depPathEmbeddings);
        layers.add(posPathEmbeddings);
        layers.add(positionEmbeddings);
        return layers;
    }

    public EmbeddingLayer getSubcatEmbeddings() {
        return subcatEmbeddings;
    }

    public EmbeddingLayer getDepPathEmbeddings() {
        return depPathEmbeddings;
    }

    public EmbeddingLayer getPosPathEmbeddings() {
        return posPathEmbeddings;
    }

    public EmbeddingLayer getPositionEmbeddings() {
        return positionEmbeddings;
    }

    @Override
    public void mergeInPlace(Layer anotherLayer) {
        super.mergeInPlace(anotherLayer);
        wordEmbeddings.mergeInPlace(((FirstHiddenLayer) anotherLayer).getWordEmbeddings());
        posEmbeddings.mergeInPlace(((FirstHiddenLayer) anotherLayer).getPosEmbeddings());
        depEmbeddings.mergeInPlace(((FirstHiddenLayer) anotherLayer).getDepEmbeddings());
        subcatEmbeddings.mergeInPlace(((FirstHiddenLayer) anotherLayer).subcatEmbeddings);
        depPathEmbeddings.mergeInPlace(((FirstHiddenLayer) anotherLayer).depPathEmbeddings);
        posPathEmbeddings.mergeInPlace(((FirstHiddenLayer) anotherLayer).posPathEmbeddings);
        positionEmbeddings.mergeInPlace(((FirstHiddenLayer) anotherLayer).positionEmbeddings);
    }

    public void emptyPrecomputedMap() {
        saved = null;
        wordEmbeddings.emptyPrecomputedMap();
    }

    public final HashMap<Integer, Integer>[] getPrecomputationMap() {
        return wordEmbeddings.getPrecomputationMap();
    }

    public void setPrecomputationMap(HashMap<Integer, Integer>[] precomputationMap) {
        wordEmbeddings.setPrecomputationMap(precomputationMap);
    }

    @Override
    public void setLayer(Layer layer) {
        super.setLayer(layer);
        saved = ((FirstHiddenLayer) layer).saved;
        wordEmbeddings.setLayer(((FirstHiddenLayer) layer).wordEmbeddings);
        posEmbeddings.setLayer(((FirstHiddenLayer) layer).posEmbeddings);
        depEmbeddings.setLayer(((FirstHiddenLayer) layer).depEmbeddings);
        subcatEmbeddings.setLayer(((FirstHiddenLayer) layer).subcatEmbeddings);
        depPathEmbeddings.setLayer(((FirstHiddenLayer) layer).depPathEmbeddings);
        posPathEmbeddings.setLayer(((FirstHiddenLayer) layer).posPathEmbeddings);
        positionEmbeddings.setLayer(((FirstHiddenLayer) layer).positionEmbeddings);
    }
}
