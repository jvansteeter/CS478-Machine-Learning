import java.util.ArrayList;

public class DecisionTree extends SupervisedLearner
{
    private Node head;

    public DecisionTree()
    {
        head = new Node();
    }

    @Override
    public void train(Matrix features, Matrix targets) throws Exception
    {
        EntrySet entrySet = new EntrySet(features, targets);
        double[] infoGains = entrySet.getFeatureInfoGains();

        for (double info : infoGains)
        {
            System.out.println(info);
        }

//        features.print();
//        targets.print();
        for (int i = 0; i < features.rows(); i++)
        {
            for (double item : features.row(i))
            {
                System.out.print(item + " ");
            }
            System.out.println();
        }
        EntrySet[] test = entrySet.splitOnFeature(0);
        test[0].print();
        test[0].targets.print();
        test[1].print();
        test[1].targets.print();
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

        public double getInfo()
        {
            double totalInfo = 0.0;
            int numberOfClasses = targets.valueCount(0);
            int entries = this.rows();

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
                totalInfo += -(count / entries) * (Math.log(count / entries) / Math.log(2));
            }

            return totalInfo;
        }

        public double[] getFeatureInfoGains()
        {
            double[] infoGains = new double[this.cols()];
            int entries = this.rows();
            double totalInfo = this.getInfo();
            for (int feature = 0; feature < this.cols(); feature++)
            {
                int[] featureCount = this.featureCounts(feature);
                double featureInfo = 0.0;
                for (int i = 0; i < featureCount.length; i++)
                {
                    double featureClassInfo = 0.0;
                    int featureTotal = featureCount[i];
                    int[] featureClassCount = this.featureClassCounts(feature, i);
                    for (int featureClassTotal : featureClassCount)
                    {
                        if (featureClassTotal == 0)
                        {
                            continue;
                        }
                        featureClassInfo += -((double) featureClassTotal / (double) featureTotal) * (Math.log((double) featureClassTotal / (double) featureTotal) / Math.log(2));
                    }
                    featureInfo += ((double) featureCount[i] / (double) entries) * featureClassInfo;
                }
                infoGains[feature] = totalInfo - featureInfo;
            }

            return infoGains;
        }

        public int[] featureCounts(int col)
        {
            int[] counts = new int[this.valueCount(col)];
            for (int i = 0; i < this.rows(); i++)
            {
                counts[(int) this.row(i)[col]]++;
            }

            return counts;
        }

        public int[] featureClassCounts(int col, double nominalValue)
        {
            int[] counts = new int[this.targets.valueCount(0)];
            for (int i = 0; i < this.rows(); i++)
            {
                if (this.row(i)[col] == nominalValue)
                {
                    counts[(int) targets.row(i)[0]]++;
                }
            }

            return counts;
        }

        public EntrySet[] splitOnFeature(int col)
        {
            EntrySet[] results = new EntrySet[this.valueCount(col)];
            int[] featureCounts = this.featureCounts(col);
            for (int featureIndex = 0; featureIndex < this.valueCount(col); featureIndex++)
            {
                Matrix newMatrix = new Matrix();
                Matrix newTargets = new Matrix();
                newMatrix.setSize(featureCounts[featureIndex], this.cols() - 1);
                newTargets.setSize(featureCounts[featureIndex], 1);
                int newRow = 0;
                for (int row = 0; row < this.rows(); row++)
                {
                    if (this.row(row)[col] == featureIndex)
                    {
                        newTargets.set(newRow, 0, targets.row(row)[0]);
                        for (int column = 0; column < this.cols() - 1; column++)
                        {
                            if (column >= col)
                            {
                                newMatrix.set(newRow, column, this.get(row, column + 1));
                            }
                            else
                            {
                                newMatrix.set(newRow, column, this.get(row, column));
                            }
                        }
                        newRow++;
                    }
                }
                results[featureIndex] = new EntrySet(newMatrix, newTargets);
            }

            return results;
        }
    }

    private class Node
    {
        private boolean endNode;
        private Node[] children;
        private int splitOnFeature;
        private double featureValue;

        public Node()
        {

        }
    }
}
