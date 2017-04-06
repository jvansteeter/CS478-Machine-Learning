import java.util.Random;
import java.util.TreeMap;

import static java.lang.Double.NaN;


public class ClusteringManager
{

    public void run(String[] args) throws Exception
    {

        //args = new String[]{"-L", "baseline", "-A", "data/iris.arff", "-E", "cross", "10", "-N"};

        Random rand = new Random(); // Use a seed for deterministic results (makes debugging easier)
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
        TreeMap<Double, Integer> silhouetteScores = new TreeMap<>();
        for (int i = 2; i < 20; i++)
        {
            learner.setK(i);
            learner.run(data);
            double averageSilhouetteScore = learner.averageSilhoutteScore();
            System.out.println("Averages- K=" + i + " silhouette=" + averageSilhouetteScore);
            silhouetteScores.put(averageSilhouetteScore, i);
        }
        silhouetteScores.remove(NaN);
        System.out.println("Best score is " + silhouetteScores.lastKey() + " at K=" + silhouetteScores.lastEntry().getValue());
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
