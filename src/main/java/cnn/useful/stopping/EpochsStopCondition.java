package cnn.useful.stopping;

import cnn.graphs.Graph;
import cnn.dataset.DataSet;

public class EpochsStopCondition implements StopCondition {

    private int n;
    private int epochs;

    public EpochsStopCondition(int epochs) {
        this.n = -1;
        this.epochs = epochs;
    }

    @Override
    public boolean shouldStop(int i, Graph nn, DataSet dataSet) {
        if (i == 0 || !dataSet.hasNextBatch(true)) {
            dataSet.reload();
            n++;
            if (n < epochs)
                System.out.println("Epochs:" + (n + 1) + "/" + epochs + ".");
        }
        return n >= epochs;
    }
}
