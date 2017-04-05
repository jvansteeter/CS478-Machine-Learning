// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------


import java.io.*;
import java.util.ArrayList;
import java.util.Random;


public class ClusteringManager
{

    public void run(String[] args) throws Exception
    {

        //args = new String[]{"-L", "baseline", "-A", "data/iris.arff", "-E", "cross", "10", "-N"};

        Random rand = new Random(1234); // Use a seed for deterministic results (makes debugging easier)
//		Random rand = new Random(); // No seed for non-deterministic results

        //Parse the command line arguments
        ArgParser parser = new ArgParser(args);
        String fileName = parser.getARFF(); //File specified by the user
        String learnerName = parser.getLearner(); //Learning algorithm specified by the user
        String evalMethod = parser.getEvaluation(); //Evaluation method specified by the user
        String evalParameter = parser.getEvalParameter(); //Evaluation parameters specified by the user
        boolean printConfusionMatrix = parser.getVerbose();
        boolean normalize = parser.getNormalize();

        // Load the model
        KMeans learner = new KMeans(rand);

        // Load the ARFF file
        Matrix data = new Matrix();
        data.loadArff(fileName);

        if (normalize)
        {
            System.out.println("Using normalized data\n");
            data.normalize();
        }

        // Print some stats
        System.out.println();
        System.out.println("Dataset name: " + fileName);
        System.out.println("Number of instances: " + data.rows());
        System.out.println("Number of attributes: " + data.cols());
        System.out.println("Learning algorithm: " + learnerName);
        System.out.println("Evaluation method: " + evalMethod);
        System.out.println();


        System.out.println("Calculating accuracy on training set...");
        learner.run(data);
//        Matrix features = new Matrix(data, 0, 0, data.rows(), data.cols() - 1);
//        Matrix labels = new Matrix(data, 0, data.cols() - 1, data.rows(), 1);
//        Matrix confusion = new Matrix();
//        double startTime = System.currentTimeMillis();
//        learner.train(features, labels);
//        double elapsedTime = System.currentTimeMillis() - startTime;
//        System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
//        double accuracy = learner.measureAccuracy(features, labels, confusion);
//        System.out.println("Training set accuracy: " + accuracy);
//        if (printConfusionMatrix)
//        {
//            System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
//            confusion.print();
//            System.out.println("\n");
//        }
    }

    /**
     * Class for parsing out the command line arguments
     */
    private class ArgParser
    {

        String arff;
        String learner;
        String evaluation;
        String evalExtra;
        boolean verbose;
        boolean normalize;

        //You can add more options for specific learning models if you wish
        public ArgParser(String[] argv)
        {
            try
            {

                for (int i = 0; i < argv.length; i++)
                {

                    if (argv[i].equals("-V"))
                    {
                        verbose = true;
                    }
                    else if (argv[i].equals("-N"))
                    {
                        normalize = true;
                    }
                    else if (argv[i].equals("-A"))
                    {
                        arff = argv[++i];
                    }
                    else
                    {
                        System.out.println("Invalid parameter: " + argv[i]);
                        System.exit(0);
                    }
                }

            }
            catch (Exception e)
            {
                System.out.println("Usage:");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
                System.out.println("OPTIONS:");
                System.out.println("-V Print the confusion matrix and learner accuracy on individual class values\n");

                System.out.println("Possible evaluation methods are:");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
                System.exit(0);
            }

            if (arff == null)
            {
                System.out.println("Usage:");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
                System.out.println("OPTIONS:");
                System.out.println("-V Print the confusion matrix and learner accuracy on individual class values");
                System.out.println("-N Use normalized data");
                System.out.println();
                System.out.println("Possible evaluation methods are:");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
                System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
                System.exit(0);
            }
        }

        //The getter methods
        public String getARFF()
        {
            return arff;
        }

        public String getLearner()
        {
            return learner;
        }

        public String getEvaluation()
        {
            return evaluation;
        }

        public String getEvalParameter()
        {
            return evalExtra;
        }

        public boolean getVerbose()
        {
            return verbose;
        }

        public boolean getNormalize()
        {
            return normalize;
        }
    }

    public static void main(String[] args) throws Exception
    {
        ClusteringManager ml = new ClusteringManager();
        ml.run(args);
    }
}
