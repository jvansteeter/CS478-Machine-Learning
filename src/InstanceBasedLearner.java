import java.io.Writer;
import java.util.*;

public class InstanceBasedLearner extends SupervisedLearner
{
    private Random rand;
    private int kNeighbors;
    private NeighborSet neighbors;
    private boolean regression;
    private boolean[] nominals;
    private Set<Integer> ignore;

    private Writer fileWriter;

    public InstanceBasedLearner(Random rand)
    {
        kNeighbors = 1;
        regression = false;
        this.rand = rand;
        ignore = new HashSet<>();
    }

    @Override
    public void train(Matrix features, Matrix labels) throws Exception
    {
        neighbors = new NeighborSet(features, labels);
        nominals = new boolean[neighbors.cols()];
        for (int i = 0; i < neighbors.cols(); i++)
        {
            nominals[i] = neighbors.valueCount(i) != 0;
        }
        neighbors.buildDistanceTable();

//        double threshold = neighbors.averageDistance();
//        System.out.println("Threshold: " + threshold);

        for (int i = 0; i < neighbors.rows(); i++)
        {
            double score = neighbors.getDistanceScore(i);
            if (score > 1.0)
            {
                ignore.add(i);
            }
        }
        System.out.println("ignored: " + ignore.size());
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        Directory directory = new Directory(kNeighbors);
        for (int i = 0; i < neighbors.rows(); i++)
        {
            if (ignore.contains(i))
            {
                continue;
            }
            double distance = distance(features, neighbors.row(i));
            directory.add(i, distance);
        }
        double vote = neighbors.vote(directory.nn(), features);

        labels[0] = vote;
    }

    public int getkNeighbors()
    {
        return kNeighbors;
    }

    public void setkNeighbors(int kNeighbors)
    {
        this.kNeighbors = kNeighbors;
    }

    public void setFileWriter(Writer fileWriter)
    {
        this.fileWriter = fileWriter;
    }

    public void setRegression(boolean regression)
    {
        this.regression = regression;
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

    private class NeighborSet extends Matrix
    {
        private Matrix targets;
        private double[][] distanceTable;

        private NeighborSet(Matrix features, Matrix targets)
        {
            super(features, 0, 0, features.rows(), features.cols());
            this.targets = targets;
        }

        private double vote(List<Integer> neighbors, double[] subject)
        {
            HashMap<Double, Double> votes = new HashMap<>();
            double regressionVote = 0;
            double weightsSum = 0;
            for (Integer neighbor : neighbors)
            {
                double vote = targets.row(neighbor)[0];
                regressionVote += vote * 1.0 / (Math.pow(distance(subject, this.row(neighbor)), 2));
                weightsSum += 1.0 / (Math.pow(distance(subject, this.row(neighbor)), 2));
                if (votes.containsKey(vote))
                {
//                    votes.replace(vote, (votes.get(vote) + 1.0));
                    votes.replace(vote, (votes.get(vote) + (1.0 / (Math.pow(distance(subject, this.row(neighbor)), 2)))));
                }
                else
                {
//                    votes.put(vote, 1.0);
                    votes.put(vote, (1.0 / (Math.pow(distance(subject, this.row(neighbor)), 2))));
                }
            }
//            regressionVote = regressionVote / neighbors.size();
            regressionVote = regressionVote / weightsSum;
            if (regression)
            {
                return regressionVote;
            }
            double vote = -1.0;
            for (Map.Entry<Double, Double> entry : votes.entrySet())
            {
                if (vote == -1.0)
                {
                    vote = entry.getKey();
                }
                else
                {
                    if (votes.get(vote) < entry.getValue())
                    {
                        vote = entry.getKey();
                    }
                }
            }

            return vote;
        }

        private void buildDistanceTable()
        {
            distanceTable = new double[this.rows()][this.rows()];
            for (int i = 0; i < this.rows(); i++)
            {
                for (int j = i; j < this.rows(); j++)
                {
                    double distance;
                    if (i == j)
                    {
                        distance = 0.0;
                    }
                    else
                    {
                        distance = distance(this.row(i), this.row(j));
                    }
                    distanceTable[i][j] = distance;
                    distanceTable[j][i] = distance;
                }
            }
        }

        private double averageDistance()
        {
            double sum = 0.0;
            for (int i = 0; i < rows(); i++)
            {
                sum += distanceTable[0][i];
            }

            return sum / rows();
        }

        private double getDistanceScore(int row)
        {
            double result = 0.0;
            TreeSet<Double> que = new TreeSet<>();
            for (int i = 0; i < this.rows(); i++)
            {
                que.add(distanceTable[row][i]);
            }
            for (int i = 0; i < kNeighbors; i++)
            {
                result += que.first();
                que.remove(que.first());
            }

            return result;
        }
    }

    private class Directory extends HashMap<Integer, Double>
    {
        private int size;

        private Directory(int size)
        {
            super();
            this.size = size;
        }

        private void add(int key, double value)
        {
            super.put(key, value);
            if (this.size() > this.size)
            {
                int farthest = -1;
                for (Entry<Integer, Double> entry : this.entrySet())
                {
                    if (farthest == -1)
                    {
                        farthest = entry.getKey();
                    }
                    else
                    {
                        if (this.get(farthest) < entry.getValue())
                        {
                            farthest = entry.getKey();
                        }
                    }
                }
                this.remove(farthest);
            }
        }

        private List<Integer> nn()
        {
            List<Integer> nearest = new ArrayList<>();
            for (Entry<Integer, Double> entry : this.entrySet())
            {
                nearest.add(entry.getKey());
            }

            return nearest;
        }
    }
}
