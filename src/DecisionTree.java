import java.util.ArrayList;
import java.util.Random;

public class DecisionTree extends SupervisedLearner
{
    private Node head;
    private Random rand;

    public DecisionTree(Random rand)
    {
        this.rand = rand;
    }

    @Override
    public void train(Matrix features, Matrix targets) throws Exception
    {
        EntrySet entrySet = new EntrySet(features, targets);
        head = new Node(entrySet);
        head.address = "head";
//        head.train(-1, 0);
        head.train();
//        head.printTree();
    }

    @Override
    public void predict(double[] features, double[] prediction) throws Exception
    {
//        System.out.print("predict: ");
//        for (double thing : features)
//        {
//            System.out.print(thing + " ");
//        }
//        System.out.println();
        prediction[0] = head.predict(features);
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
            int numberOfClasses = (int) targets.columnMax(0) + 1;
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
                if (count == 0)
                {
                    continue;
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

        private int[] featureCounts(int col)
        {
            int[] counts = new int[(int)this.columnMax(col) + 1];
            for (int i = 0; i < this.rows(); i++)
            {
                counts[(int) this.row(i)[col]]++;
            }

            return counts;
        }

        private int[] featureClassCounts(int col, double nominalValue)
        {
            int[] counts = new int[(int)this.targets.columnMax(0) + 1];
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
            EntrySet[] results = new EntrySet[(int)this.columnMax(col) + 1];
            int[] featureCounts = this.featureCounts(col);
            for (int featureIndex = 0; featureIndex < this.columnMax(col) + 1; featureIndex++)
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
        private EntrySet entrySet;
        private String address = "";

        public Node(EntrySet entrySet)
        {
            this.entrySet = entrySet;
            endNode = false;
        }

//        public void train(int parent, int childNum)
        public void train()
        {
//            parent++;
//            System.out.println("Info Layer: " + parent + " Child#: " + childNum);
            if (entrySet.cols() > 1)
            {
                double[] infoGains = entrySet.getFeatureInfoGains();
//                for (double info : infoGains)
//                {
//                    System.out.println(info);
//                }

                double bestFeature = 0;
                int bestFeatureIndex = 0;
                for (int i = 0; i < infoGains.length; i++)
                {
                    if (infoGains[i] > bestFeature)
                    {
                        bestFeature = infoGains[i];
                        bestFeatureIndex = i;
                    }
                }
                splitOnFeature = bestFeatureIndex;
                EntrySet[] splits = entrySet.splitOnFeature(splitOnFeature);
                children = new Node[splits.length];
                for (int i = 0; i < splits.length; i++)
                {
                    children[i] = new Node(splits[i]);
                    children[i].address = this.address + "->" + i;
//                    children[i].train(parent, i);
                    children[i].train();
                }
            }
            else
            {
                endNode = true;
            }
        }

        public double predict(double[] features)
        {
//            System.out.println(address);
            if (!endNode)
            {
                double nominalValue = features[splitOnFeature];
                features = removeFeature(features, splitOnFeature);
//                System.out.println("Nom: " + nominalValue);
//                System.out.println(entrySet.featureCounts(0)[0]);
//                System.out.println(children.length);
                if (nominalValue >= children.length)
                {
                    nominalValue = Math.abs(rand.nextInt()) % children.length;
                }
                return children[(int) nominalValue].predict(features);
            }

            return entrySet.targets.mostCommonValue(0);
        }

        public void printTree()
        {
            System.out.println("Head");
            System.out.println("Split on: " + splitOnFeature);
            for (int i = 0; i < children.length; i++)
            {
                children[i].printTree(1, "" + i);
            }
        }

        public void printTree(int layer, String address)
        {
            System.out.println("Layer: " + layer + ": " + address);
            System.out.println(children == null ? "0" : children.length);
            layer++;
            if (!endNode)
            {
                System.out.println("Split on " + splitOnFeature);
                for (int i = 0; i < children.length; i++)
                {
                    children[i].printTree(layer, address + "->" + i);
                }
            }
            else
            {
                System.out.println("End Node of class: " + entrySet.targets.mostCommonValue(0));
            }
        }

        private double[] removeFeature(double[] input, int col)
        {
            double[] newFeatures = new double[input.length - 1];
            for(int i = 0; i < input.length - 1; i++)
            {
                if (i < col)
                {
                    newFeatures[i] = input[i];
                }
                else
                {
                    newFeatures[i] = input[i + 1];
                }
            }

            return newFeatures;
        }
    }
}
