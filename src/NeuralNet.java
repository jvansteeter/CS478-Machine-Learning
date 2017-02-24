import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class NeuralNet extends SupervisedLearner
{
    private Random rand;
    private int hiddenLayerSize;
    private double learningRate = 0.3;
    private boolean momentum;
    private double momentumCoeff = 1.3;
    private int layers = 1;
    private TargetNode[] targetNodes;
//    private HiddenNode[] hiddenNodes;
    private HiddenNode[][] hiddenLayers;
    private Writer fileWriter;

    public NeuralNet(Random rand)
    {
        this.rand = rand;
        this.momentum = false;
    }

    public NeuralNet(Random rand, boolean momentum)
    {
        this.rand = rand;
        this.momentum = momentum;
    }

    @Override
    public void train(Matrix inputs, Matrix targets) throws Exception
    {
//        hiddenLayerSize = inputs.cols() * 2;
        hiddenLayers = new HiddenNode[layers][hiddenLayerSize];
        for (int i = 0; i < layers; i++)
        {
            hiddenLayers[i] = new HiddenNode[hiddenLayerSize];
        }
//        hiddenNodes = new HiddenNode[hiddenLayerSize];
        TargetNode[] bestTargetNodes = targetNodes;
//        HiddenNode[] bestHiddenNodes = hiddenNodes;
        HiddenNode[][] bestHiddenLayers = hiddenLayers;
        double bestVSAccuracy = 0.0;
        double bestTrainMSE = 0.0;
        double bestVSMSE = 0.0;

        // if there are multiple output classes
        if (targets.valueCount(0) > 2)
        {
            targetNodes = new TargetNode[targets.valueCount(0)];
            for (int i = 0; i < targets.valueCount(0); i++)
            {
                targetNodes[i] = new TargetNode(hiddenLayerSize, i);
            }
        }
        // if there is only one output class
        else
        {
            targetNodes = new TargetNode[1];
            targetNodes[0] = new TargetNode(hiddenLayerSize, 1);
        }
//        for (int i = 0; i < hiddenLayerSize; i++)
//        {
//            hiddenNodes[i] = new HiddenNode(inputs.cols());
//        }
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            hiddenLayers[0][i] = new HiddenNode(inputs.cols());
        }
        for (int i = 1; i < layers; i++)
        {
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                hiddenLayers[i][j] = new HiddenNode(hiddenLayerSize);
            }
        }

        // separate input into training and validation sets
        int trainingSetSize = (int) (inputs.rows() * .8);
        Matrix trainingFeatures = new Matrix(inputs, 0, 0, trainingSetSize, inputs.cols());
        Matrix trainingTargets = new Matrix(targets, 0, 0, trainingSetSize, 1);
        Matrix validationFeatures = new Matrix(inputs, trainingSetSize, 0, inputs.rows() - trainingSetSize, inputs.cols());
        Matrix validationTargets = new Matrix(targets, trainingSetSize, 0, inputs.rows() - trainingSetSize, 1);

        // save results of training to file
//        File outputFile = new File("results.csv");
//        if (outputFile.exists())
//        {
//            outputFile.delete();
//        }
//        outputFile.createNewFile();
//        Writer fileWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "utf-8"));
//        fileWriter.write("Learning Rate,VS MSE,Training MSE,Test MSE,VS Accuracy,Test Accuracy,Epoch\n");
//        fileWriter.write("Epoch,Training MSE,VS MSE,VS Accuracy\n");

        // start learning
        boolean learning = true;
        int epochsWithoutImprovement = 0;
        double bestValidationAccuracy = 0.0;
        double lastValidationAccuracy = 0.0;
        int epochCount = 0;
        int bestFoundAtEpoch = 0;
        while (learning)
        {
            double mseTrainSum = 0.0;
            int mseTrainCount = 0;
            double mseVS = 0.0;
            trainingFeatures.shuffle(rand, trainingTargets);
            epochCount++;
            // one epoch
            for (int i = 0; i < trainingFeatures.rows(); i++)
            {
                double[] row = trainingFeatures.row(i);
                double target = trainingTargets.row(i)[0];
                // get the outputs from the hidden nodes by feeding them the input nodes
//                double[] hiddenOutputs = new double[hiddenNodes.length];
//                for (int j = 0; j < hiddenNodes.length; j++)
//                {
//                    hiddenOutputs[j] = hiddenNodes[j].output(row);
//                }
                double[][] hiddenOutputs = new double[layers][hiddenLayerSize];
                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    hiddenOutputs[0][j] = hiddenLayers[0][j].output(row);
                }
                for (int j = 1; j < layers; j++)
                {
                    for (int k = 0; k < hiddenLayerSize; k++)
                    {
                        hiddenOutputs[j][k] = hiddenLayers[j][k].output(hiddenOutputs[j - 1]);
                    }
                }
                // get the outputs from the target nodes by feeding them the output form the hidden nodes
                double[] targetOutputs = new double[targetNodes.length];
                double[] targetErrors = new double[targetNodes.length];
                for (int j = 0; j < targetNodes.length; j++)
                {
                    // outputs
//                    targetOutputs[j] = targetNodes[j].output(hiddenOutputs);
                    targetOutputs[j] = targetNodes[j].output(hiddenOutputs[hiddenOutputs.length - 1]);
                    // calculate the errors
                    targetErrors[j] = targetNodes[j].error(targetOutputs[j], target);
                    // calc mean squared error for training set
                    mseTrainSum += Math.pow(targetErrors[j], 2);
                    mseTrainCount++;
                    // update the weights for the target nodes
//                    targetNodes[j].updateWeight(targetErrors[j], hiddenOutputs);
                    targetNodes[j].updateWeight(targetErrors[j], hiddenOutputs[hiddenOutputs.length - 1]);
                }
                // back propogate the weights through the hidden layer
//                for (int j = 0; j < hiddenNodes.length; j++)
//                {
//                    double[] targetWeights = new double[targetNodes.length];
//                    for (int k = 0; k < targetNodes.length; k++)
//                    {
//                        targetWeights[k] = targetNodes[k].getWeight(j);
//                    }
//                    hiddenNodes[j].updateWeight(targetErrors, targetWeights, hiddenOutputs[j], row);
//                }
                double[] layerErrors = new double[hiddenLayers[0].length];
                for (int j = 0; j < hiddenLayers[hiddenLayers.length - 1].length; j++)
                {
                    double[] lastInputs;
                    if (hiddenOutputs.length == 1)
                    {
                        lastInputs = row;
                    }
                    else
                    {
                        lastInputs = hiddenOutputs[hiddenOutputs.length - 2];
                    }
                    double[] targetWeights = new double[targetNodes.length];
                    for (int k = 0; k < targetNodes.length; k++)
                    {
                        targetWeights[k] = targetNodes[k].getWeight(j);
                    }
                    layerErrors[j] = hiddenLayers[hiddenLayers.length - 1][j].updateWeight(targetErrors, targetWeights, hiddenOutputs[hiddenOutputs.length - 1][j], lastInputs);
                }
                for (int j = hiddenLayers.length - 2; j >= 0; j--)
                {
                    double[] lastInputs;
                    if (j == 0)
                    {
                        lastInputs = row;
                    }
                    else
                    {
                        lastInputs = hiddenOutputs[j - 1];
                    }
                    double[] nextLayerErrors = new double[hiddenLayers[j + 1].length];
                    for (int k = 0; k < hiddenLayers[j].length; k++)
                    {
                        double[] targetWeights = new double[hiddenLayers[j + 1].length];
                        for (int l = 0; l < hiddenLayers[j + 1].length; l++)
                        {
                            targetWeights[l] = hiddenLayers[j + 1][l].getWeight(k);
                        }
                        nextLayerErrors[k] = layerErrors[k] = hiddenLayers[j][k].updateWeight(layerErrors, targetWeights, hiddenOutputs[j][k], lastInputs);
                    }
                }
            }

            // epoch complete, check validation accuracy
            double correct = 0.0;
            double total = 0.0;
            for (int i = 0; i < validationFeatures.rows(); i++)
            {
                total++;
                double[] row = validationFeatures.row(i);
                double target = validationTargets.row(i)[0];
                double[] prediction = {0};
                predict(row, prediction);
                if (target == prediction[0])
                {
                    correct++;
                }
                // calc mean squared error for training set
                mseVS = calcMeanSquaredError(row, target);
            }
            double validationAccuracy = correct / total;
            if (validationAccuracy > lastValidationAccuracy)
            {
                lastValidationAccuracy = validationAccuracy;
                if (validationAccuracy >= bestValidationAccuracy)
                {
                    bestFoundAtEpoch = epochCount;
                    epochsWithoutImprovement = 0;
                    bestValidationAccuracy = validationAccuracy;
//                    bestHiddenNodes = hiddenNodes.clone();
                    bestHiddenLayers = new HiddenNode[layers][hiddenLayerSize];
                    for (int i = 0; i < layers; i++)
                    {
                        bestHiddenLayers[i] = hiddenLayers[i].clone();
                    }
                    bestTargetNodes = targetNodes.clone();
                    bestVSAccuracy = validationAccuracy;
                    bestTrainMSE = mseTrainSum / mseTrainCount;
                    bestVSMSE = mseVS;
                }
            }
            else if (validationAccuracy <= lastValidationAccuracy)
            {
                epochsWithoutImprovement++;
                lastValidationAccuracy = validationAccuracy;
            }
            if (epochsWithoutImprovement > 10 * Math.pow(10, layers))
            {
                learning = false;
            }

            // save to file the result of the epoch
//            fileWriter.write(epochCount + "," + mseTrainSum / mseTrainCount + "," + mseVS + "," + validationAccuracy + "\n");
        }
//        fileWriter.close();
        System.out.println("Total epochs: " + bestFoundAtEpoch);
        System.out.println("Training MSE: " + bestTrainMSE);
        System.out.println("Validation MSE: " + bestVSMSE);
        System.out.println("Validation Accuracy: " + bestValidationAccuracy);
//        fileWriter.write(learningRate + "," + bestFoundAtEpoch + "," + bestTrainMSE + "," + bestVSAccuracy + "," + bestVSMSE + ",");
//        fileWriter.write(hiddenLayerSize + "," + bestFoundAtEpoch + "," + bestTrainMSE + "," + bestVSAccuracy + "," + bestVSMSE + ",");
//        fileWriter.write(momentumCoeff + "," + bestFoundAtEpoch + "," + bestTrainMSE + "," + bestVSAccuracy + "," + bestVSMSE + ",");
        fileWriter.write(layers + "," + bestFoundAtEpoch + "," + bestTrainMSE + "," + bestVSAccuracy + "," + bestVSMSE + ",");
//        hiddenNodes = bestHiddenNodes;
        hiddenLayers = bestHiddenLayers;
        targetNodes = bestTargetNodes;
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
//        double[] hiddenOutputs = new double[hiddenNodes.length];
//        for (int i = 0; i < hiddenNodes.length; i++)
//        {
//            hiddenOutputs[i] = hiddenNodes[i].output(features);
//        }
        double[] lastLayerOutputs = features;
        double[][] hiddenOutputs = new double[hiddenLayers.length][hiddenLayers[0].length];
        for (int i = 0; i < hiddenLayers.length; i++)
        {
            for (int j = 0; j <hiddenLayers[i].length; j++)
            {
                hiddenOutputs[i][j] = hiddenLayers[i][j].output(lastLayerOutputs);
            }
            lastLayerOutputs = hiddenOutputs[i];
        }

        double[] targetOutputs = new double[targetNodes.length];
        double prediction = -1.0;
        double highest = 0.0;
        if (targetNodes.length > 1)
        {
            for (int i = 0; i < targetNodes.length; i++)
            {
//                targetOutputs[i] = targetNodes[i].output(hiddenOutputs);
                targetOutputs[i] = targetNodes[i].output(hiddenOutputs[hiddenOutputs.length - 1]);
                if (targetOutputs[i] > highest)
                {
                    highest = targetOutputs[i];
                    prediction = i;
                }
            }
        }
        else
        {
//            double output = targetNodes[0].output(hiddenOutputs);
            double output = targetNodes[0].output(hiddenOutputs[hiddenOutputs.length - 1]);
            if (output >= 0.5)
            {
                prediction = 1.0;
            }
            else
            {
                prediction = 0.0;
            }
        }

        labels[0] = prediction;
    }

    public double calcMeanSquaredError(double[] features, double target)
    {
        double meanSquaredSum = 0.0;
        int meanSquaredCount = 0;
//        double[] hiddenOutputs = new double[hiddenNodes.length];
//        for (int j = 0; j < hiddenNodes.length; j++)
//        {
//            hiddenOutputs[j] = hiddenNodes[j].output(features);
//        }
        double[] lastLayerOutputs = features;
        double[][] hiddenOutputs = new double[hiddenLayers.length][hiddenLayers[0].length];
        for (int i = 0; i < hiddenLayers.length; i++)
        {
            for (int j = 0; j <hiddenLayers[i].length; j++)
            {
                hiddenOutputs[i][j] = hiddenLayers[i][j].output(lastLayerOutputs);
            }
            lastLayerOutputs = hiddenOutputs[i];
        }
        // get the outputs from the target nodes by feeding them the output form the hidden nodes
        double[] targetOutputs = new double[targetNodes.length];
        double[] targetErrors = new double[targetNodes.length];
        for (int j = 0; j < targetNodes.length; j++)
        {
            // calc mean squared error for training set
            // outputs
            targetOutputs[j] = targetNodes[j].output(hiddenOutputs[hiddenOutputs.length - 1]);
            // calculate the errors
            targetErrors[j] = targetNodes[j].error(targetOutputs[j], target);
            meanSquaredSum += Math.pow(targetErrors[j], 2);
            meanSquaredCount++;
            // update the weights for the target nodes
        }

        return meanSquaredSum / meanSquaredCount;
    }

    public void incrementLearningRate()
    {
        learningRate -= 0.01;
    }

    public void incrementMomentum()
    {
        momentumCoeff += 0.1;
    }

    public void setFileWriter(Writer writer)
    {
        this.fileWriter = writer;
    }

    public void setHiddenLayerSize(int size)
    {
        hiddenLayerSize = size;
    }

    public void setLayers(int layers)
    {
        this.layers = layers;
    }

    private abstract class Node
    {
        protected double[] weights;
        protected double[] lastDeltas;
        protected Node(int numInputs)
        {
            weights = new double[numInputs + 1];
            lastDeltas = new double[numInputs + 1];
            for (int i = 0; i < weights.length; i++)
            {
                weights[i] = (rand.nextDouble() - .5) * (rand.nextBoolean() ? 1 : -1);
                lastDeltas[i] = 0;
            }
        }

        public double output(double[] inputs)
        {
            double net = 0;
            for (int i = 0; i < inputs.length + 1; i++)
            {
                // account for the bias
                if (i == inputs.length)
                {
                    net += weights[i] * 1;
                }
                else
                {
                    net += weights[i] * inputs[i];
                }
            }
            double output = 1 / (1 + Math.pow(Math.E, -net));

            return output;
        }

        public double getWeight(int input)
        {
            return weights[input];
        }


        public void print()
        {
            for (double weight : weights)
            {
                System.out.print(weight + " ");
            }
            System.out.println();
        }
    }

    private class TargetNode extends Node
    {
        private double target;

        public TargetNode(int numInputs, double target)
        {
            super(numInputs);
            this.target = target;
        }

        public double error(double output, double target)
        {
            if (target == this.target)
            {
                target = 1;
            }
            else
            {
                target = 0;
            }
            double error = (target - output) * output * (1 - output);

            return error;
        }

        public void updateWeight(double error, double[] inputs)
        {
            double[] deltas = new double[inputs.length + 1];
            for (int i = 0; i < inputs.length + 1; i++)
            {
                // account for bias
                if (i == inputs.length)
                {
                    deltas[i] = learningRate * error * 1;
                }
                else
                {
                    deltas[i] = learningRate * error * inputs[i];
                }
            }
            for (int i = 0; i < weights.length; i++)
            {
                weights[i] += deltas[i] + (momentum ? momentumCoeff * lastDeltas[i] : 0);
            }
            lastDeltas = deltas;
        }

        @Override
        public TargetNode clone()
        {
            TargetNode newNode = new TargetNode(weights.length - 1, this.target);
            newNode.weights = Arrays.copyOf(this.weights, this.weights.length);

            return newNode;
        }
    }

    private class HiddenNode extends Node
    {
        public HiddenNode(int numInputs)
        {
            super(numInputs);
        }

        public double updateWeight(double[] errors, double[] outputWeights, double output, double[] inputs)
        {
            double sum = 0;
            for (int i = 0; i < outputWeights.length; i++)
            {
                sum += errors[i] * outputWeights[i];
            }
            double error = output * (1 - output) * sum;

            double[] deltas = new double[inputs.length + 1];
            for (int i = 0; i < inputs.length + 1; i++)
            {
                // account for bias
                if (i == inputs.length)
                {
                    deltas[i] = learningRate * error * 1;
                }
                else
                {
                    deltas[i] = learningRate * error * inputs[i];
                }
            }
            for (int i = 0; i < weights.length; i++)
            {
                weights[i] += deltas[i] + (momentum ? momentumCoeff * lastDeltas[i] : 0);
            }
            lastDeltas = deltas;

            return error;
        }

        @Override
        public HiddenNode clone()
        {
            HiddenNode newNode = new HiddenNode(weights.length - 1);
            newNode.weights = Arrays.copyOf(this.weights, this.weights.length);

            return newNode;
        }
    }
}
