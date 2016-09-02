package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork;

import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.EmbeddingLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.FirstHiddenLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.WordEmbeddingLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.*;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/27/16
 * Time: 10:40 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MLPTrainer {
    MLPNetwork net;

    /**
     * for multi-threading
     */
    ExecutorService executor;
    CompletionService<Pair<Pair<Double, Double>, MLPNetwork>> pool;
    int numThreads;

    /**
     * Keep track of loss function
     */
    double cost = 0.0;
    double correct = 0.0;
    int samples = 0;
    Updater updater;
    Random random;
    /**
     * Gradients
     */
    private ArrayList<Layer> gradients;
    private double regCoef;
    private double dropoutProb;
    private boolean regularizeAllLayers;

    public MLPTrainer(MLPNetwork net, Options options) throws Exception {
        this.net = net;
        random = new Random();
        this.dropoutProb = options.networkProperties.dropoutProbForHiddenLayer;
        if (options.updaterProperties.updaterType == UpdaterType.SGD)
            updater = new SGD(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm,
                    options.updaterProperties.momentum, options.updaterProperties.sgdType);
        else if (options.updaterProperties.updaterType == UpdaterType.ADAGRAD)
            updater = new Adagrad(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm, 1e-6);
        else if (options.updaterProperties.updaterType == UpdaterType.ADAM)
            updater = new Adam(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm, 0.9, 0.9999, 1e-8);
        else if (options.updaterProperties.updaterType == UpdaterType.ADAMAX)
            updater = new AdaMax(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm, 0.9, 0.9999, 1e-8);
        else
            throw new Exception("Updater not implemented");
        this.regCoef = options.networkProperties.regularization;
        this.numThreads = options.generalProperties.numOfThreads;
        this.regularizeAllLayers = options.networkProperties.regualarizeAllLayers;
        executor = Executors.newFixedThreadPool(numThreads);
        pool = new ExecutorCompletionService<>(executor);
    }

    private void regularizeWithL2() {
        double regCost = 0.0;

        ArrayList<Layer> layers = net.getLayers();
        for (int i = 0; i < layers.size() - 1; i++) {
            Layer layer = layers.get(i);
            Layer gradient = gradients.get(i);
            for (int d1 = 0; d1 < layer.nOut(); d1++) {
                if (regularizeAllLayers) {
                    regCost += Math.pow(layer.b(d1), 2);
                    gradient.modifyB(d1, regCoef * 2 * layer.b(d1));
                }
                for (int d2 = 0; d2 < layer.nIn(); d2++) {
                    regCost += Math.pow(layer.w(d1, d2), 2);
                    gradient.modifyW(d1, d2, regCoef * 2 * layer.w(d1, d2));
                }
            }
        }

        if (regularizeAllLayers) {
            // regularizing wrt embedding layers.
            FirstHiddenLayer fLayer = (FirstHiddenLayer) layers.get(0);
            FirstHiddenLayer fGradient = (FirstHiddenLayer) gradients.get(0);

            ArrayList<EmbeddingLayer> flayers = fLayer.embeddingLayers();
            ArrayList<EmbeddingLayer> glayers = fGradient.embeddingLayers();
            for(int i=0;i<flayers.size();i++){
                EmbeddingLayer l = flayers.get(i);
                EmbeddingLayer gl = glayers.get(i);

                for (int d1 = 0; d1 < l.nOut(); d1++) {
                    for (int d2 = 0; d2 < l.nIn(); d2++) {
                        regCost += Math.pow(l.w(d1, d2), 2);
                        gl.modifyW(d1, d2, regCoef * 2 * l.w(d1, d2));
                    }
                }
            }

            // regularizing wrt last layer.
            Layer layer = layers.get(layers.size() - 1);
            Layer gradient = gradients.get(layers.size() - 1);
            for (int d1 = 0; d1 < layer.nOut(); d1++) {
                regCost += Math.pow(layer.b(d1), 2);
                gradient.modifyB(d1, regCoef * 2 * layer.b(d1));
                for (int d2 = 0; d2 < layer.nIn(); d2++) {
                    regCost += Math.pow(layer.w(d1, d2), 2);
                    gradient.modifyW(d1, d2, regCoef * 2 * layer.w(d1, d2));
                }
            }
        }

        cost += regCoef * regCost;
    }

    public double fit(List instances, int iteration, boolean print) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");
        DecimalFormat format4 = new DecimalFormat("##.0000");

        cost(instances);
        regularizeWithL2();
        updater.update(gradients);
        ((FirstHiddenLayer) net.getLayers().get(0)).preCompute();

        double acc = correct / samples;
        if (print) {
            System.out.println(Utils.timeStamp() + " ---  iteration " + iteration + " --- size " +
                    samples + " --- Correct " + format.format(100. * acc) + " --- cost: " + format4.format(cost / samples));
            cost = 0;
            samples = 0;
            correct = 0;
        }
        return acc;
    }

    public void cost(List instances) throws Exception {
        submitThreads(instances);
        mergeCosts(instances);
        samples += instances.size();
    }

    private void submitThreads(List instances) {
        int chunkSize = Math.max(1, instances.size() / numThreads);
        int s = 0;
        int e = Math.min(instances.size(), chunkSize);
        for (int i = 0; i < Math.min(instances.size(), numThreads); i++) {
            pool.submit(new CostThread(instances.subList(s, e), instances.size()));
            s = e;
            e = Math.min(instances.size(), e + chunkSize);
        }
    }

    private void mergeCosts(List<NeuralTrainingInstance> instances) throws Exception {
        Pair<Pair<Double, Double>, MLPNetwork> firstResult = pool.take().get();
        gradients = firstResult.second.getLayers();

        cost += firstResult.first.first;
        correct += firstResult.first.second;

        for (int i = 1; i < Math.min(instances.size(), numThreads); i++) {
            Pair<Pair<Double, Double>, MLPNetwork> result = pool.take().get();

            for (int l = 0; l < gradients.size(); l++)
                gradients.get(l).mergeInPlace(result.second.getLayers().get(l));
            cost += result.first.first;
            correct += result.first.second;
        }
        if (Double.isNaN(cost))
            throw new Exception("cost is not a number");
    }

    public double getLearningRate() {
        return updater.getLearningRate();
    }

    public void setLearningRate(double learningRate) {
        updater.setLearningRate(learningRate);
    }

    public void shutDownLiveThreads() {
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }

    private void backPropSavedGradients(MLPNetwork g, double[][][] savedGradients, HashSet<Integer>[] wordsSeen) {
        int offset = 0;
        final FirstHiddenLayer firstHiddenLayer = (FirstHiddenLayer) net.layer(0);
        final double[][] hiddenLayer = firstHiddenLayer.getW();
        WordEmbeddingLayer wordEmbeddingLayer = firstHiddenLayer.getWordEmbeddings();
        final double[][] wE = wordEmbeddingLayer.getW();
        final double[][] pE = firstHiddenLayer.getPosEmbeddings().getW();
        final double[][] lE = firstHiddenLayer.getDepEmbeddings().getW();

        for (int index = 0; index < g.getNumWordLayers(); index++) {
            for (int tok : wordsSeen[index]) {
                int id = wordEmbeddingLayer.preComputeId(index, tok);
                double[] embedding = wE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][id][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                            g.modify(EmbeddingTypes.WORD, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += wordEmbeddingLayer.dim();
        }

        for (int index = g.getNumWordLayers(); index < g.getNumWordLayers() + g.getNumPosLayers(); index++) {
            for (int tok = 0; tok < firstHiddenLayer.getPosEmbeddings().vocabSize(); tok++) {
                double[] embedding = pE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.POS, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += firstHiddenLayer.getPosEmbeddings().dim();
        }

        for (int index = g.getNumWordLayers() + g.getNumPosLayers();
             index < g.getNumWordLayers() + g.getNumPosLayers() + g.getNumDepLayers(); index++) {
            for (int tok = 0; tok < firstHiddenLayer.getDepEmbeddings().vocabSize(); tok++) {
                double[] embedding = lE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.DEPENDENCY, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += firstHiddenLayer.getDepEmbeddings().dim();
        }
    }

    /**
     * @param instances      this is actually a List of NeuralTrainingInstance(s).
     * @param batchSize
     * @param g
     * @param savedGradients
     * @return
     * @throws Exception
     */
    public Pair<Double, Double> cost(List instances, int batchSize, MLPNetwork g, double[][][] savedGradients)
            throws Exception {
        double cost = 0;
        double correct = 0;
        HashSet<Integer>[] featuresSeen = Utils.createHashSetArray(g.getNumWordLayers());

        double[][] features = new double[instances.size()][];
        double[][] labels = new double[instances.size()][];
        ArrayList<double[][]> activations = new ArrayList<>();
        ArrayList<double[][]> zs = new ArrayList<>();
        HashSet<Integer>[] hiddenNodesToUse = dropout(instances.size(), net.layer(0).nOut());
        HashSet<Integer>[] finalHiddenNodesToUse = hiddenNodesToUse;

        for (int i = 0; i < instances.size(); i++) {
            features[i] = ((NeuralTrainingInstance) instances.get(i)).getFeatures();
            labels[i] = ((NeuralTrainingInstance) instances.get(i)).getLabel();
        }

        zs.add(features);
        activations.add(features);
        int lIndex = 0;
        zs.add(net.layer(lIndex).forward(activations.get(activations.size() - 1), hiddenNodesToUse));
        activations.add(net.layer(lIndex++).activate(zs.get(zs.size() - 1), false));
        if (g.getLayers().size() >= 3) {
            HashSet<Integer>[] secondHiddenNodesToUse = dropout(instances.size(), net.layer(lIndex).nOut());
            zs.add(net.layer(lIndex).forward(activations.get(activations.size() - 1), secondHiddenNodesToUse, hiddenNodesToUse));
            activations.add(net.layer(lIndex++).activate(zs.get(zs.size() - 1), false));
            finalHiddenNodesToUse = secondHiddenNodesToUse;
        }
        zs.add(net.layer(lIndex).forward(activations.get(activations.size() - 1), labels, false, false));
        activations.add(net.layer(lIndex++).activate(zs.get(zs.size() - 1), false));
        double[][] probs = activations.get(activations.size() - 1);
        for (int i = 0; i < probs.length; i++) {
            int argmax = Utils.argmax(probs[i]);
            int gold = ((NeuralTrainingInstance) instances.get(i)).gold();
            double goldProb = probs[i][gold] == 0 ? 1e-120 : probs[i][gold];
            cost += -Math.log(goldProb);
            if (argmax == gold) correct++;
        }

        double[][] delta = new double[probs.length][probs[0].length];
        for (int i = 0; i < probs.length; i++) {
            for (int j = 0; j < probs[0].length; j++) {
                if (labels[i][j] >= 0)
                    delta[i][j] = (-labels[i][j] + probs[i][j]) / batchSize;
            }
        }

        // Back-propagating for the last layer.
        double[][] lastHiddenActivation = activations.get(activations.size() - 2);
        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j < delta[i].length; j++) {
                if (labels[i][j] >= 0 && delta[i][j] != 0) {
                    g.modify(net.numLayers() - 1, j, -1, delta[i][j]);
                    for (int h : finalHiddenNodesToUse[i]) {
                        g.modify(net.numLayers() - 1, j, h, delta[i][j] * lastHiddenActivation[i][h]);
                    }
                }
            }
        }

        // backwarding to all other layers.
        for (int i = net.numLayers() - 2; i >= 0; i--) {
            delta = g.layer(i).backward(delta, i, zs.get(i + 1), activations.get(i), activations.get(i + 1), featuresSeen, savedGradients, net);
        }

        backPropSavedGradients(g, savedGradients, featuresSeen);
        return new Pair<>(cost, correct);
    }

    private HashSet<Integer>[] dropout(int numOfInstances, int size) {
        HashSet<Integer>[] hiddenNodesToUse = new HashSet[numOfInstances];
        for (int i = 0; i < hiddenNodesToUse.length; i++)
            hiddenNodesToUse[i] = dropout(size);
        return hiddenNodesToUse;
    }

    private HashSet<Integer> dropout(int size) {
        HashSet<Integer> hiddenNodesToUse = new HashSet<>();
        for (int h = 0; h < size; h++) {
            if (dropoutProb <= 0 || random.nextDouble() >= dropoutProb)
                hiddenNodesToUse.add(h);
        }
        return hiddenNodesToUse;
    }

    public ArrayList<Layer> getGradients() {
        return gradients;
    }

    public class CostThread implements Callable<Pair<Pair<Double, Double>, MLPNetwork>> {
        List<Object> instances;
        int batchSize;
        MLPNetwork g;
        double[][][] savedGradients;

        public CostThread(List<Object> instances, int batchSize) {
            this.instances = instances;
            this.batchSize = batchSize;
            g = net.clone(true, false);
            savedGradients = net.instantiateSavedGradients();
        }

        @Override
        public Pair<Pair<Double, Double>, MLPNetwork> call() throws Exception {
            Pair<Double, Double> costValue= cost(instances, batchSize, g, savedGradients);
            return new Pair<>(costValue, g);
        }
    }
}
