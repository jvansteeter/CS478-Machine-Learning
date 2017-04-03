import java.util.Arrays;
import java.util.Random;

public class KMeans
{
    private Random rand;
    private int k = 4;

    public KMeans(Random rand)
    {
        this.rand = rand;
    }

    public void run(Matrix features) throws Exception
    {
        double[][] centroids = new double[k][];
        for (int i = 0; i < k; i++)
        {
            centroids[i] = features.row(i).clone();
        }
    }

}
