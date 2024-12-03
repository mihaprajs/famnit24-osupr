package ann;

import java.util.function.Function;

public class Layer {
    private Matrix weights;
    private Matrix bias;

    private Function<Matrix, Matrix> activationFunction;

    private int layerSize;

    public Layer(int layerSize, Function<Matrix,Matrix> activationFunction) {
        this.layerSize = layerSize;
        this.activationFunction = activationFunction;
    }

    public int initWeights(int layerSizeBefore) {
        this.weights = Matrix.randomMatrix(layerSizeBefore, layerSize);
        this.bias = Matrix.randomMatrix(1, layerSize);

        return layerSize;
    }

    public Matrix feedForoward(Matrix input) {
        Matrix multResult = Matrix.multiply(weights, input);
        Matrix weightedSums = Matrix.addMatrices(multResult, bias);
        Matrix out = activationFunction.apply(weightedSums);
        return out;
    }
}
