package ann;

public class ArtificialNeuralNetwork {
    private Layer[] layers;
    private float learningRate;

    public ArtificialNeuralNetwork(int inputSize, float learningRate, Layer[] layers) {
        this.layers = layers;
        this.learningRate = learningRate;

        int sizeOfLayerBefore = inputSize;
        for (Layer layer : layers) {
            sizeOfLayerBefore = layer.initWeights(sizeOfLayerBefore);
        }
    }

    public float[] predictOne(float[] input) {
        Matrix inputMatrix = new Matrix(input).transpose();

        Matrix inputToNextLayer = inputMatrix;
        for (Layer layer : layers) {
            inputToNextLayer = layer.feedForoward(inputToNextLayer);
        }

        return inputToNextLayer.transpose().getValues()[0];
    }
}
