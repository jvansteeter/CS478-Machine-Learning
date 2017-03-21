import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class InstanceBasedLearner extends SupervisedLearner
{
    private int kNeighbors;
    private NeighborSet neighbors;
    private boolean regression;

    private Writer fileWriter;

    public InstanceBasedLearner()
    {
        kNeighbors = 3;
        regression = false;
    }

    @Override
    public void train(Matrix features, Matrix labels) throws Exception
    {
        neighbors = new NeighborSet(features, labels);
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        Directory directory = new Directory(kNeighbors);
        for (int i = 0; i < neighbors.rows(); i++)
        {
            double distance = distance(features, neighbors.row(i));
            directory.add(i, distance);
        }
        double vote = neighbors.vote(directory.nn());

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
            sum += Math.pow(one[i] - two[i], 2);
        }

        return Math.sqrt(sum);
    }

    private class NeighborSet extends Matrix
    {
        private Matrix targets;

        private NeighborSet(Matrix features, Matrix targets)
        {
            super(features, 0, 0, features.rows(), features.cols());
            this.targets = targets;
        }

        private double vote(List<Integer> neighbors)
        {
            HashMap<Double, Integer> votes = new HashMap<>();
            double regressionVote = 0;
            for (Integer neighbor : neighbors)
            {
                double vote = targets.row(neighbor)[0];
                regressionVote += vote;
                if (votes.containsKey(vote))
                {
                    votes.replace(vote, votes.get(vote) + 1);
                }
                else
                {
                    votes.put(vote, 1);
                }
            }
            regressionVote = regressionVote / neighbors.size();
            double vote = -1.0;
            for (Map.Entry<Double, Integer> entry : votes.entrySet())
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

            if (regression)
            {
                return regressionVote;
            }
            return vote;
        }
    }

    private class Directory extends HashMap<Integer,Double>
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
                for(Entry<Integer,Double> entry : this.entrySet())
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
