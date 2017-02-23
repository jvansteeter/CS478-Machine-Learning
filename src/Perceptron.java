import java.util.Random;

public class Perceptron extends SupervisedLearner
{
    private double learningRate = .1;
    private Random rand;
    private double[] weights;

    public Perceptron(Random rand)
    {
        this.rand = rand;
    }

    @Override
    public void train(Matrix inputs, Matrix targets) throws Exception
    {
        // initialize weights to 0
        weights = new double[inputs.row(0).length + 1];
        double weightSum = 0;
        for (int i = 0; i < weights.length; i++)
        {
            if (i == weights.length + 1)
            {
                weights[i] = 0 - weightSum;
            }
            else
            {
                double weight = rand.nextFloat() * 0.1;
                weights[i] = weight;
                weightSum += weight;
            }
//            weights[i] = 0.0;
        }

        // while the model is still learning iterate through another epoch
        int epochsWithoutImprovement = 0;
        double lastAccuracy = 0.0;
        int epoch = 0;
        inputs.shuffle(rand, targets);
        System.out.println("\nEpoch Accuracy:");
        while (epochsWithoutImprovement < 5)
        {
            epoch++;
            int correct = 0;
            for (int i = 0; i < inputs.rows(); i++)
            {
                double target = targets.row(i)[0];
                double net = 0;
                double output;
                for (int j = 0; j < inputs.row(i).length + 1; j++)
                {
                    // add the bias
                    if (j == inputs.row(i).length)
                    {
                        net += 1 * weights[j];
                    }
                    // add the features
                    else
                    {
                        net += inputs.row(i)[j] * weights[j];
                    }
                }
                if (net > 0)
                {
                    output = 1;
                }
                else
                {
                    output = 0;
                }

                if (output == target)
                {
                    correct++;
                }
                else
                {
                    updateWeights(inputs.row(i), weights, target, output);
                }
            }
            double accuracy = (double)correct / inputs.rows();
            if (accuracy - lastAccuracy > .01)
            {
                epochsWithoutImprovement = 0;
            }
            else
            {
                epochsWithoutImprovement++;
            }
            lastAccuracy = accuracy;
            System.out.println(epoch + ", " + accuracy);
        }
        System.out.println("Number of epochs before haulting: " + epoch);
        System.out.println("\nWeight Values");
        for (int i = 0; i < inputs.cols(); i++)
        {
            System.out.println(inputs.attrName(i) + ", " + weights[i]);
        }
        System.out.println();
    }

    @Override
    public void predict(double[] features, double[] prediction) throws Exception
    {
        double net = 0;
        for (int i = 0; i < features.length + 1; i++)
        {
            if (i == features.length)
            {
                net += 1 * weights[i];
            }
            else
            {
                net += features[i] * weights[i];
            }
        }
        if (net > 0)
        {
            prediction[0] = 1;
        }
        else
        {
            prediction[0] = 0;
        }
    }

    private void updateWeights(double[] features, double[] weights, double target, double output)
    {
        double[] delta = new double[weights.length];
        for (int i = 0; i < features.length; i++)
        {
            delta[i] = learningRate*(target - output)*features[i];
        }
        for (int i = 0; i < weights.length; i++)
        {
            weights[i] += delta[i];
        }
    }
}
