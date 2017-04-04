import java.util.*;

public class KMeans
{
    private Random rand;
    private int k = 4;
    private boolean[] nominals;

    public KMeans(Random rand)
    {
        this.rand = rand;
    }

    public void run(Matrix features) throws Exception
    {
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

            centroids = new double[k][];
            for (int i = 0; i < k; i++)
            {
                centroids[i] = data.row(i).clone();
            }
            for (int i = 0; i < centroids.length; i++)
            {
                centroidMembers.add(new TreeSet<>());
            }
        }

        private void iterate()
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
                assign(i,-=);
            }
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
