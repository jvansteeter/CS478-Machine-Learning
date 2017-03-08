import java.util.ArrayList;

public class DecisionTree extends SupervisedLearner
{
    public DecisionTree()
    {

    }

    @Override
    public void train(Matrix features, Matrix targets) throws Exception
    {
        EntrySet entrySet = new EntrySet(features, targets);
        // calculate the total info for the tree
        double totalInfo = 0.0;
        int numberOfClasses = targets.valueCount(0);
        int entries = features.rows();

        for (int i = 0; i < numberOfClasses; i++)
        {
            double count = 0;
            for (int j = 0; j < entries; j++)
            {
                if (targets.row(j)[0] == i)
                {
                    count += 1;
                }
            }
            totalInfo += -(count/entries) * (Math.log(count/entries)/Math.log(2));
        }

        double[] infoGains = new double[features.cols()];
        for (double info : infoGains)
        {
            info = 0.0;
        }
        for (int feature = 0; feature < features.cols(); feature++)
        {
            int[] featureCount = entrySet.featureCounts(feature);
            double featureInfo = 0.0;
            for (int i = 0; i < featureCount.length; i++)
            {
                double featureClassInfo = 0.0;
                int featureTotal = featureCount[i];
                int[] featureClassCount = entrySet.featureClassCounts(feature, i);
                for (int featureClassTotal : featureClassCount)
                {
                    if (featureClassTotal == 0)
                    {
                        continue;
                    }
                    featureClassInfo += -((double)featureClassTotal/(double)featureTotal) * (Math.log((double)featureClassTotal/(double)featureTotal)/Math.log(2));
                }
                featureInfo += ((double)featureCount[i]/(double)entries) * featureClassInfo;
            }
            infoGains[feature] = totalInfo - featureInfo;
        }

        for (double info : infoGains)
        {
            System.out.println(info);
        }
    }

    @Override
    public void predict(double[] features, double[] prediction) throws Exception
    {

    }

    private class EntrySet extends Matrix
    {
        private Matrix targets;

        public EntrySet(Matrix matrix, Matrix targets)
        {
            super(matrix, 0, 0, matrix.rows(), matrix.cols());
            this.targets = targets;
        }

        public int[] featureCounts(int col)
        {
            int[] counts = new int[this.valueCount(col)];
            for (int item : counts)
            {
                item = 0;
            }
            for (int i = 0; i < this.rows(); i++)
            {
                counts[(int)this.row(i)[col]]++;
            }

            return counts;
        }

        public int[] featureClassCounts(int col, double nominalValue)
        {
            int[] counts = new int[this.targets.valueCount(0)];
            for (int item : counts)
            {
                item = 0;
            }
            for (int i = 0; i < this.rows(); i++)
            {
                if (this.row(i)[col] == nominalValue)
                {
                    counts[(int) targets.row(i)[0]]++;
                }
            }

            return counts;
        }
    }
}
