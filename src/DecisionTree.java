import java.util.HashSet;
import java.util.Random;

public class DecisionTree extends SupervisedLearner
{
    private Node head;
    private Random rand;
    private HashSet<Node> prunableNodes;
    private boolean prune;

    public DecisionTree(Random rand)
    {
        this.rand = rand;
        prunableNodes = new HashSet<>();
        prune = false;
    }

    @Override
    public void train(Matrix features, Matrix targets) throws Exception
    {
        if (!prune)
        {
            EntrySet entrySet = new EntrySet(features, targets);
            head = new Node(entrySet);
            head.address = "head";
            head.train();
//            head.printTree();
        }
        else
        {
            // separate into test and training sets
            int trainingSetSize = (int) (features.rows() * .8);
            features.shuffle(rand, targets);
            Matrix trainingFeatures = new Matrix(features, 0, 0, trainingSetSize, features.cols());
            Matrix trainingTargets = new Matrix(targets, 0, 0, trainingSetSize, 1);
            Matrix validationFeatures = new Matrix(features, trainingSetSize, 0, features.rows() - trainingSetSize, features.cols());
            Matrix validationTargets = new Matrix(targets, trainingSetSize, 0, features.rows() - trainingSetSize, 1);

            EntrySet entrySet = new EntrySet(trainingFeatures, trainingTargets);
            head = new Node(entrySet);
            head.address = "head";
            head.train();
//            head.printTree();

            // begin the super obnoxious process of error pruning
            double originalAccuracy = this.measureAccuracy(validationFeatures, validationTargets, null);
            HashSet<Node> pruned = new HashSet<>();
            double bestValidationAccuracy = originalAccuracy;
            System.out.println("Original Validation Accuracy: " + originalAccuracy);
            while (prunableNodes.size() > 0)
            {
                HashSet<Node> nodesToRemove = new HashSet<>();
                HashSet<Node> nodesToAdd = new HashSet<>();
                for (Node node : prunableNodes)
                {
                    head.unPrune();
                    for (Node hasBeenPruned : pruned)
                    {
                        hasBeenPruned.isPruned = true;
                    }
                    node.isPruned = true;
                    double validationAccuracy = this.measureAccuracy(validationFeatures, validationTargets, null);
                    if (validationAccuracy < bestValidationAccuracy)
                    {
                        nodesToRemove.add(node);
                    }
                    else
                    {
                        bestValidationAccuracy = validationAccuracy;
                        pruned.add(node);
                        nodesToRemove.add(node);
                        if (node.parent != null)
                        {
                            nodesToAdd.add(node.parent);
                        }
                    }
                }
                prunableNodes.removeAll(nodesToRemove);
                prunableNodes.addAll(nodesToAdd);
            }

            head.unPrune();
            for (Node node : pruned)
            {
                node.isPruned = true;
            }
            System.out.println("Pruned Validation Accuracy: " + bestValidationAccuracy);
            System.out.println("Able to prune " + pruned.size() + " nodes");
        }
    }

    @Override
    public void predict(double[] features, double[] prediction) throws Exception
    {
        double[] columnMaxes = head.entrySet.getColumnMaxes();
        for (int i = 0; i < features.length; i++)
        {
            if (features[i] > columnMaxes[i])
            {
                features[i] = columnMaxes[i];
            }
        }
        prediction[0] = head.predict(features);
    }

    public void setPrune(boolean prune)
    {
        this.prune = prune;
    }

    public boolean isPrune()
    {
        return prune;
    }

    public int nodeCount()
    {
        return head.nodeCount();
    }

    public int depth()
    {
        return head.getDepth();
    }

    private class EntrySet extends Matrix
    {
        private Matrix targets;

        public EntrySet(Matrix matrix, Matrix targets)
        {
            this.targets = targets;
            this.setSize(matrix.rows(), matrix.cols());
            for (int row = 0; row < matrix.rows(); row++)
            {
                for (int col = 0; col < matrix.cols(); col++)
                {
                    double max = matrix.columnMax(col);
                    double value = matrix.get(row, col);
                    if (value > max)
                    {
                        value = max + 1;
                    }
                    set(row, col, value);
                }
            }
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
            if (totalInfo == 0)
            {
                return infoGains;
            }
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
            int[] counts = new int[(int) this.columnMax(col) + 1];
            for (int i = 0; i < this.rows(); i++)
            {
                counts[(int) this.row(i)[col]]++;
            }

            return counts;
        }

        private int[] featureClassCounts(int col, double nominalValue)
        {
            int[] counts = new int[(int) this.targets.columnMax(0) + 1];
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
            EntrySet[] results = new EntrySet[(int) this.columnMax(col) + 1];
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

        public double[] getColumnMaxes()
        {
            double[] maxes = new double[this.cols()];
            for (int i = 0; i < this.cols(); i++)
            {
                maxes[i] = this.columnMax(i);
            }

            return maxes;
        }
    }

    private class Node
    {
        private boolean endNode;
        private Node[] children;
        private int splitOnFeature;
        private EntrySet entrySet;
        private String address = "";
        private boolean isPruned = false;
        private Node parent = null;
        private int depth;

        public Node(EntrySet entrySet)
        {
            this.entrySet = entrySet;
            endNode = false;
            depth = 0;
        }

        public void train()
        {
            if (entrySet.cols() > 1)
            {
                double[] infoGains = entrySet.getFeatureInfoGains();
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
                if (bestFeature == 0)
                {
                    prunableNodes.add(this.parent);
                    endNode = true;
                    return;
                }
                splitOnFeature = bestFeatureIndex;
                EntrySet[] splits = entrySet.splitOnFeature(splitOnFeature);
                children = new Node[splits.length];
                for (int i = 0; i < splits.length; i++)
                {
                    children[i] = new Node(splits[i]);
                    children[i].parent = this;
                    children[i].address = this.address + "->" + i;
                    children[i].depth = this.depth + 1;
                    children[i].train();
                }
            }
            else
            {
                endNode = true;
                prunableNodes.add(this.parent);
            }
        }

        public double predict(double[] features)
        {
            if (!endNode || isPruned)
            {
                double nominalValue = features[splitOnFeature];
                features = removeFeature(features, splitOnFeature);
                if (nominalValue >= children.length)
                {
                    nominalValue = Math.abs(rand.nextInt()) % children.length;
                }
                return children[(int) nominalValue].predict(features);
            }

            return entrySet.targets.mostCommonValue(0);
        }

        public int getDepth()
        {
            if (isPruned)
            {
                return 0;
            }
            if (children != null)
            {
                int deepest = 0;
                for (Node child : children)
                {
                    if (child.getDepth() > deepest)
                    {
                        deepest = child.getDepth();
                    }
                }
                return deepest;
            }

            return depth;
        }

        public int nodeCount()
        {
            int[] count = {1};
            nodeCount(count);

            return count[0];
        }

        private void nodeCount(int[] count)
        {
            if (!isPruned)
            {
                count[0]++;
            }
            if (children != null)
            {
                for (Node child : children)
                {
                    child.nodeCount(count);
                }
            }
        }

        public void unPrune()
        {
            isPruned = false;
            if (children != null)
            {
                for (Node child : children)
                {
                    child.unPrune();
                }
            }
        }

        public void printTree()
        {
            System.out.println(address);
            if (!endNode)
            {
                System.out.println("Split on " + splitOnFeature);
                for (int i = 0; i < children.length; i++)
                {
                    children[i].printTree();
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
            for (int i = 0; i < input.length - 1; i++)
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
