import java.io.*;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;

public class KMeans
{
    private int k = 4;
    private Random rand;
    private boolean[] nominals;
    private DecimalFormat rounder;
    private Writer fileWriter;

    public KMeans(Random rand)
    {
        this.rand = rand;
        rounder = new DecimalFormat("0.000");
        rounder.setRoundingMode(RoundingMode.HALF_EVEN);
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
        double lastIterationSSE = 0.0;
        boolean changing = true;
        while (changing)
        {
            map.makeAssignments();
            map.outputIteration();
            map.calcNewCentroids();

            changing = false;
            double nextIterationSSE = map.sumSquaredError();
            if (lastIterationSSE != nextIterationSSE)
            {
                lastIterationSSE = nextIterationSSE;
                changing = true;
            }
        }

        System.out.println("Number of clusters: " + k);
        for (int i = 0; i < k; i++)
        {
            System.out.println(map.centroidInfo(i));
        }
        for (int i = 0; i < k; i++)
        {
            System.out.println("Cluster " + i + ": Instances-" + map.centroidMembers.get(i).size() + "\tSSE-" + rounder.format(map.sumSquaredError(i)) + "\tMSE-" + rounder.format(map.meanSquaredError(i)));
        }
        System.out.println("Total SSE: " + rounder.format(map.sumSquaredError()));

        fileWriter.close();
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
        private int iteration;
        private double[][] centroids;
        private Matrix data;
        private HashMap<Integer, Integer> assignments;
        private ArrayList<TreeSet<Integer>> centroidMembers;
        private ArrayList<ArrayList<Double>> distances;

        private CentroidMap(int k, Matrix data)
        {
            this.k = k;
            this.iteration = 1;
            this.data = data;
            assignments = new HashMap<>();
            centroidMembers = new ArrayList<>();
            distances = new ArrayList<>();

//            centroids = new double[this.k][];
//            for (int i = 0; i < k; i++)
//            {
//                centroids[i] = data.row(i).clone();
//            }
            centroids = new double[this.k][];
            for (int i = 0; i < k; i++)
            {
                int randomIndex = rand.nextInt() % data.rows();
                centroids[i] = data.row(randomIndex).clone();
            }

            for (int i = 0; i < centroids.length; i++)
            {
                centroidMembers.add(new TreeSet<>());
            }
            for (int i = 0; i < data.rows(); i++)
            {
                distances.add(new ArrayList<>());
            }
        }

        private boolean makeAssignments()
        {
            if (iteration > 1)
            {
                clear();
            }
            boolean change = false;
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
                boolean result = assign(i, closestCentroid);
                change = (change || result);
            }

            return change;
        }

        private void calcNewCentroids()
        {
            iteration++;
            for (int centroid = 0; centroid < k; centroid++)
            {
                for (int feature = 0; feature < data.cols(); feature++)
                {
                    if (nominals[feature])
                    {
                        HashMap<Double, Integer> counts = new HashMap<>();
                        for (int member : centroidMembers.get(centroid))
                        {
                            double nominalValue = data.row(member)[feature];
                            if (nominalValue != Double.MAX_VALUE)
                            {
                                if (counts.containsKey(nominalValue))
                                {
                                    counts.put(nominalValue, counts.get(nominalValue) + 1);
                                }
                                else
                                {
                                    counts.put(nominalValue, 1);
                                }
                            }
                        }
                        int greatestCount = -1;
                        double mostCommonNominalValue = -1.0;
                        for (Map.Entry<Double, Integer> entry : counts.entrySet())
                        {
                            if (entry.getValue() > greatestCount || (entry.getValue() == greatestCount && entry.getKey() < mostCommonNominalValue))
                            {
                                greatestCount = entry.getValue();
                                mostCommonNominalValue = entry.getKey();
                            }
                        }
                        centroids[centroid][feature] = mostCommonNominalValue;
                    }
                    else
                    {
                        double sum = 0.0;
                        int count = 0;
                        for (int member : centroidMembers.get(centroid))
                        {
                            double realValue = data.row(member)[feature];
                            if (realValue == Double.MAX_VALUE)
                            {
                                sum += 0;
                            }
                            else
                            {
                                sum += realValue;
                                count++;
                            }
                        }
                        double result;
                        if (count > 0)
                        {
                            result = sum / count;
                        }
                        else
                        {
                            result = Double.MAX_VALUE;
                        }
                        centroids[centroid][feature] = result;
                    }
                }
            }
        }

        private void outputIteration() throws IOException
        {
            fileWriter.write("***************\n");
            fileWriter.write("Iteration " + iteration + "\n");
            fileWriter.write("***************\n");
            fileWriter.write("Computing Centroids:\n");
            StringBuilder line = new StringBuilder();

            for (int i = 0; i < centroids.length; i++)
            {
//                line.append("Centroid " + i + " =");
//                for (int j = 0; j < data.cols(); j++)
//                {
//                    if (nominals[j])
//                    {
//                        if (centroids[i][j] == Double.MAX_VALUE)
//                        {
//                            line.append(" ?,");
//                        }
//                        else
//                        {
//                            line.append(" " + data.m_enum_to_str.get(j).get(((int) centroids[i][j])) + ",");
//                        }
//                    }
//                    else
//                    {
//                        if (centroids[i][j] == Double.MAX_VALUE)
//                        {
//                            line.append(" ?,");
//                        }
//                        else
//                        {
//                            line.append(" " + rounder.format(centroids[i][j]) + ",");
//                        }
//                    }
//                }
                line.append(centroidInfo(i));
                line.append("\n");
                fileWriter.write(line.toString());
                line.setLength(0);
            }
            fileWriter.write("Making Assignments\n");
            for (int i = 0; i < assignments.size(); i++)
            {
                line.append("\t" + i + "=" + assignments.get(i) + " ");
                if ((i + 1) % 10 == 0)
                {
                    line.append("\n");
                }
            }
            line.append("\n");
            fileWriter.write(line.toString());
            fileWriter.write("SSE: " + rounder.format(sumSquaredError()) + "\n\n");
        }

        private String centroidInfo(int centoid)
        {
            StringBuilder line = new StringBuilder();
            line.append("Centroid " + centoid + " =");
            for (int j = 0; j < data.cols(); j++)
            {
                if (nominals[j])
                {
                    if (centroids[centoid][j] == Double.MAX_VALUE)
                    {
                        line.append(" ?");
                    }
                    else
                    {
                        line.append(" " + data.m_enum_to_str.get(j).get(((int) centroids[centoid][j])));
                    }
                }
                else
                {
                    if (centroids[centoid][j] == Double.MAX_VALUE)
                    {
                        line.append(" ?");
                    }
                    else
                    {
                        line.append(" " + rounder.format(centroids[centoid][j]));
                    }
                }
                if (j + 1 < data.cols())
                {
                    line.append(",");
                }
            }

            return line.toString();
        }

        private boolean assign(int node, int centroid)
        {
            boolean change = false;
            if (!assignments.containsKey(node) || assignments.get(node) != centroid)
            {
                assignments.put(node, centroid);
                change = true;
            }
            else
            {
//                assignments.put(node, centroid);
            }
            centroidMembers.get(centroid).add(node);

//            System.out.println("assign: " + change);
            return change;
        }

        private double sumSquaredError()
        {
            double sum = 0.0;
            for (int i = 0; i < centroids.length; i++)
            {
                sum += sumSquaredError(i);
            }

            return sum;
        }

        private double sumSquaredError(int centroid)
        {
            double sum = 0.0;
            TreeSet<Integer> members = centroidMembers.get(centroid);
            for (int member : members)
            {
                int assignedTo = assignments.get(member);
                sum += Math.pow(distances.get(member).get(assignedTo), 2);
            }

            return sum;
        }

        private double meanSquaredError(int centroid)
        {
            return sumSquaredError(centroid) / centroidMembers.get(centroid).size();
        }

        private double[][] getCentroids()
        {
            return centroids;
        }

        private void clear()
        {
            centroidMembers.forEach((TreeSet<Integer> element) -> element.clear());
            distances.forEach((ArrayList<Double> element) -> element.clear());
        }
    }
}
