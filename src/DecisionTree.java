import java.util.HashSet;
import java.util.Random;

public class DecisionTree extends SupervisedLearner
{
    private Node head;
    private Random rand;
    private HashSet<Node> prunableNodes;
    private boolean prune;
    private int deepest;

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

            EntrySet entrySet = new EntrySet(trainingFeatures, trainingTargets, true);
            head = new Node(entrySet);
            head.address = "head";
            head.train();

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
        return deepest;
    }

    private void deepest(int depth)
    {
        if (depth > deepest)
        {
            deepest = depth;
        }
    }

////////////////////////////////////////////////////////////////////////
//                ENTRY SET
////////////////////////////////////////////////////////////////////////

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

        public EntrySet(Matrix matrix, Matrix targets, boolean smart)
        {
            this.targets = targets;
            Matrix result = new Matrix();
            result.setSize(matrix.rows(), matrix.cols());
            for (int row = 0; row < matrix.rows(); row++)
            {
                for (int col = 0; col < matrix.cols(); col++)
                {
                    double value = matrix.get(row, col);
                    result.set(row, col, value);
                }
            }

            boolean unknownsRemain = true;
            while (unknownsRemain)
            {
                Matrix original = copy(result);
                unknownsRemain = false;
                for (int col = 0; col < result.cols(); col++)
                {
                    if (colHasUnknown(col, original))
                    {
                        unknownsRemain = true;
                        Matrix column = getCol(col, original);
                        Matrix dataTargets = replaceCol(col, original, targets);
                        Matrix[] pieces = divideByKnown(dataTargets, column);
                        Matrix trainingFeatures = pieces[0];
                        Matrix trainingTargets = pieces[1];
                        Matrix unknownFeatures = pieces[2];
                        Matrix unknownTargets = pieces[3];

                        EntrySet entrySet = new EntrySet(trainingFeatures, trainingTargets);
                        Node tempHead = new Node(entrySet);
                        tempHead.train();
                        // find the new values
                        Matrix newFeatures = new Matrix();
                        newFeatures.setSize(unknownTargets.rows(), 1);
                        for (int row = 0; row < unknownFeatures.rows(); row++)
                        {
                            double[] features = unknownFeatures.row(row);
                            double[] columnMaxes = tempHead.entrySet.getColumnMaxes();
                            for (int i = 0; i < features.length; i++)
                            {
                                if (features[i] > columnMaxes[i])
                                {
                                    features[i] = columnMaxes[i];
                                }
                            }
                            newFeatures.set(row, 0, tempHead.predict(features));
                        }
                        Matrix results = recombine(trainingTargets, newFeatures);
                        result = replaceCol(col, result, results);
                    }
                }
            }

            this.setSize(result.rows(), result.cols());
            for (int row = 0; row < result.rows(); row++)
            {
                for (int col = 0; col < result.cols(); col++)
                {
                    double value = result.get(row, col);
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

        public boolean colHasUnknown(int col, Matrix matrix)
        {
            for (int i = 0; i < matrix.rows(); i++)
            {
                if (matrix.get(i, col) == Double.MAX_VALUE)
                {
                    return true;
                }
            }

            return false;
        }

        public Matrix copy(Matrix original)
        {
            Matrix newMatrix = new Matrix();
            newMatrix.setSize(original.rows(), original.cols());
            for (int row = 0; row < original.rows(); row++)
            {
                for (int col = 0; col < original.cols(); col++)
                {
                    double value = original.get(row, col);
                    newMatrix.set(row, col, value);
                }
            }

            return newMatrix;
        }

        private Matrix getCol(int col, Matrix matrix)
        {
            Matrix column = new Matrix();
            column.setSize(matrix.rows(), 1);
            for (int i = 0; i < matrix.rows(); i++)
            {
                column.set(i, 0, matrix.get(i, col));
            }

            return column;
        }

        private Matrix[] divideByKnown(Matrix features, Matrix targets)
        {
            int unknownCount = 0;
            for (int i = 0; i < targets.rows(); i++)
            {
                if (targets.get(i, 0) == Double.MAX_VALUE)
                {
                    unknownCount++;
                }
            }

            Matrix trainingFeatures = new Matrix();
            Matrix trainingTargets = new Matrix();
            Matrix unknownFeatures = new Matrix();
            Matrix unknownTargets = new Matrix();
            trainingFeatures.setSize(targets.rows() - unknownCount, features.cols());
            trainingTargets.setSize(targets.rows() - unknownCount, 1);
            unknownFeatures.setSize(unknownCount, features.cols());
            unknownTargets.setSize(unknownCount, 1);
            int trainingIndex = 0;
            int unknownIndex = 0;
            for (int row = 0; row < targets.rows(); row++)
            {
                // if this target is unknown
                if (targets.get(row, 0) == Double.MAX_VALUE)
                {
                    unknownTargets.set(unknownIndex, 0, targets.get(row, 0));
                    for (int col = 0; col < features.cols(); col++)
                    {
                        unknownFeatures.set(unknownIndex, col, features.get(row, col));
                    }
                    unknownIndex++;
                }
                else
                {
                    trainingTargets.set(trainingIndex, 0, targets.get(row, 0));
                    for (int col = 0; col < features.cols(); col++)
                    {
                        trainingFeatures.set(trainingIndex, col, features.get(row, col));
                    }
                    trainingIndex++;
                }
            }
            Matrix[] result = new Matrix[4];
            result[0] = trainingFeatures;
            result[1] = trainingTargets;
            result[2] = unknownFeatures;
            result[3] = unknownTargets;

            return result;
        }

        private Matrix replaceCol(int col, Matrix original, Matrix replacement)
        {
            Matrix result = copy(original);
            for (int i = 0; i < result.rows(); i++)
            {
                result.set(i, col, replacement.get(i, 0));
            }

            return result;
        }

        private Matrix recombine(Matrix trainingTargets, Matrix unknownTargets)
        {
            Matrix combinedTargets = new Matrix();
            combinedTargets.setSize(trainingTargets.rows() + unknownTargets.rows(), 1);
            for (int row = 0; row < trainingTargets.rows(); row++)
            {
                combinedTargets.set(row, 0, trainingTargets.get(row, 0));
            }
            for (int row = 0; row < unknownTargets.rows(); row++)
            {
                combinedTargets.set(trainingTargets.rows() + row, 0, unknownTargets.get(row, 0));
            }

            return combinedTargets;
        }
    }

////////////////////////////////////////////////////////////////////////
//                NODE
////////////////////////////////////////////////////////////////////////

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
                    children[i].setDepth(this.depth + 1);
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

        public int nodeCount()
        {
            int[] count = {1};
            nodeCount(count);

            return count[0];
        }

        private void setDepth(int depth)
        {
            this.depth = depth;
            deepest(depth);
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
