import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class NeuralNet extends SupervisedLearner
{
    private Random rand;
    private int hiddenLayerSize;
    private double learningRate = 0.3;
    private boolean momentum;
    private double momentumCoeff = 0.09;
    private TargetNode[] targetNodes;
    private HiddenNode[] hiddenNodes;
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
        hiddenNodes = new HiddenNode[hiddenLayerSize];
        TargetNode[] bestTargetNodes = targetNodes;
        HiddenNode[] bestHiddenNodes = hiddenNodes;
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
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            hiddenNodes[i] = new HiddenNode(inputs.cols());
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
                double[] hiddenOutputs = new double[hiddenNodes.length];
                for (int j = 0; j < hiddenNodes.length; j++)
                {
                    hiddenOutputs[j] = hiddenNodes[j].output(row);
                }
                // get the outputs from the target nodes by feeding them the output form the hidden nodes
                double[] targetOutputs = new double[targetNodes.length];
                double[] targetErrors = new double[targetNodes.length];
                for (int j = 0; j < targetNodes.length; j++)
                {
                    // outputs
                    targetOutputs[j] = targetNodes[j].output(hiddenOutputs);
                    // calculate the errors
                    targetErrors[j] = targetNodes[j].error(targetOutputs[j], target);
                    // calc mean squared error for training set
                    mseTrainSum += Math.pow(targetErrors[j], 2);
                    mseTrainCount++;
                    // update the weights for the target nodes
                    targetNodes[j].updateWeight(targetErrors[j], hiddenOutputs);
                }
                // back propogate the weights through the hidden layer
                for (int j = 0; j < hiddenNodes.length; j++)
                {
                    double[] targetWeights = new double[targetNodes.length];
                    for (int k = 0; k < targetNodes.length; k++)
                    {
                        targetWeights[k] = targetNodes[k].getWeight(j);
                    }
                    hiddenNodes[j].updateWeight(targetErrors, targetWeights, hiddenOutputs[j], row);
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
                    bestHiddenNodes = hiddenNodes.clone();
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
            if (epochsWithoutImprovement > 50)
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
        fileWriter.write(hiddenLayerSize + "," + bestFoundAtEpoch + "," + bestTrainMSE + "," + bestVSAccuracy + "," + bestVSMSE + ",");
        hiddenNodes = bestHiddenNodes;
        targetNodes = bestTargetNodes;
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        double[] hiddenOutputs = new double[hiddenNodes.length];
        for (int i = 0; i < hiddenNodes.length; i++)
        {
            hiddenOutputs[i] = hiddenNodes[i].output(features);
        }
        double[] targetOutputs = new double[targetNodes.length];
        double prediction = -1.0;
        double highest = 0.0;
        if (targetNodes.length > 1)
        {
            for (int i = 0; i < targetNodes.length; i++)
            {
                targetOutputs[i] = targetNodes[i].output(hiddenOutputs);
                if (targetOutputs[i] > highest)
                {
                    highest = targetOutputs[i];
                    prediction = i;
                }
            }
        }
        else
        {
            double output = targetNodes[0].output(hiddenOutputs);
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
        double[] hiddenOutputs = new double[hiddenNodes.length];
        for (int j = 0; j < hiddenNodes.length; j++)
        {
            hiddenOutputs[j] = hiddenNodes[j].output(features);
        }
        // get the outputs from the target nodes by feeding them the output form the hidden nodes
        double[] targetOutputs = new double[targetNodes.length];
        double[] targetErrors = new double[targetNodes.length];
        for (int j = 0; j < targetNodes.length; j++)
        {
            // calc mean squared error for training set
            // outputs
            targetOutputs[j] = targetNodes[j].output(hiddenOutputs);
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

    public void setFileWriter(Writer writer)
    {
        this.fileWriter = writer;
    }

    public void setHiddenLayerSize(int size)
    {
        hiddenLayerSize = size;
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

        public double getWeight(int input)
        {
            return weights[input];
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

        public void updateWeight(double[] errors, double[] outputWeights, double output, double[] inputs)
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
