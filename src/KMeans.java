import java.io.*;
import java.util.*;

public class KMeans
{
    private Random rand;
    private int k = 4;
    private boolean[] nominals;
    private Writer fileWriter;

    public KMeans(Random rand)
    {
        this.rand = rand;
    }

    public void run(Matrix features) throws Exception
    {
        // create file for result storing
        File outputFile = new File("results.csv");
        if (outputFile.exists())
        {
            outputFile.delete();
        }
        outputFile.createNewFile();
        fileWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "utf-8"));

        nominals = new boolean[features.cols()];
        for (int i = 0; i < features.cols(); i++)
        {
            nominals[i] = features.valueCount(i) != 0;
        }

        CentroidMap map = new CentroidMap(k, features);
        double[][] centroids = map.getCentroids();
        while (true)
        {

        }
    }


    private double distance(double[] one, double[] two)
    {
        double sum = 0;
        for (int i = 0; i < one.length; i++)
        {
            if (one[i] == Double.MAX_VALUE || two[i] == Double.MAX_VALUE)
            {
                sum += 1;
                continue;
            }
            if (nominals[i])
            {
                if (one[i] != two[i])
                {
                    sum += 1;
                }
                continue;
            }
            sum += Math.pow(one[i] - two[i], 2);
        }

        return Math.sqrt(sum);
    }

    private class CentroidMap
    {
        private int k;
        private double[][] centroids;
        private Matrix data;
        private HashMap<Integer, Integer> assignments;
        private ArrayList<TreeSet<Integer>> centroidMembers;
        private ArrayList<ArrayList<Double>> distances;

        private CentroidMap(int k, Matrix data)
        {
            this.k = k;
            this.data = data;
            assignments = new HashMap<>();
            centroidMembers = new ArrayList<>();
            distances = new ArrayList<>();

            centroids = new double[this.k][];
            for (int i = 0; i < k; i++)
            {
                centroids[i] = data.row(i).clone();
            }
            for (int i = 0; i < centroids.length; i++)
            {
                centroidMembers.add(new TreeSet<>());
            }
        }

        private void assignToCentroid()
        {
            for (int i = 0; i < data.rows(); i++)
            {
                double[] node = data.row(i);
                int closestCentroid = -1;
                double closestDistance = Double.MAX_VALUE;
                for (int j = 0; j < centroids.length; j++)
                {
                    double centroidDistance = distance(node, centroids[j]);
                    distances.get(i).add(centroidDistance);
                    if (centroidDistance < closestDistance)
                    {
                        closestCentroid = j;
                        closestDistance = centroidDistance;
                    }
                }
                assign(i,closestCentroid);
            }
        }

        private void outputIteration() throws IOException
        {
            fileWriter.write("***************\n");
            fileWriter.write("Iteration 1\n");
            fileWriter.write("***************\n");
            fileWriter.write("Computing Centroids:\n");
            StringBuilder line = new StringBuilder();

            for (int i = 0; i < centroids.length; i++)
            {
                line.append("Centroid " + i + " =");
                for (int j = 0; j < data.cols(); j++)
                {
                    if (centroids[i][j] == Double.MAX_VALUE)
                    {
                        line.append(" ?,");
                    }
                    else
                    {
                        if (nominals[j])
                        {
                            line.append(" " + data.m_enum_to_str.get(j).get(centroids[i][j]) + ",");
                        }
                        else
                        {
                            line.append(" " + centroids[i][j] + ",");
                        }
                    }
                }
                fileWriter.write(line.toString());
            }
            line.setLength(0);
            fileWriter.write("Making Assignments");
            for (int i = 0; i < assignments.size(); i++)
            {
                line.append(i + "=" + assignments.get(i) + "\t");
                if ((i + 1) % 10 == 0)
                {
                    line.append("\n");
                }
            }
            fileWriter.write(line.toString());
            fileWriter.write("SSE: ****");
        }

        private void assign(int node, int centroid)
        {
            assignments.put(node,centroid);
            centroidMembers.get(centroid).add(node);
        }

        private double[][] getCentroids()
        {
            return centroids;
        }
    }
}
