import ann.ActivationFunction;
import ann.ArtificialNeuralNetwork;
import ann.Layer;

public class Main2 {
    static Point[] data = generateData(500);

    static Gui gui = new Gui(data);

    public static void main(String[] args) {
        ArtificialNeuralNetwork ann = new ArtificialNeuralNetwork(2, 0.1f, new Layer[] {
                new Layer(5, ActivationFunction.sigmoid),
                new Layer(5, ActivationFunction.sigmoid),
                new Layer(1, ActivationFunction.sigmoid)
        });

        test(ann);
    }

    public static void test(ArtificialNeuralNetwork model) {
        for (Point p : data) {
            float[] input = {p.x, p.y};
            float prediction = Math.round(model.predictOne(input)[0]);

            p.guessed = (prediction == p.label);
        }
        gui.repaint();
    }

    public static Point[] generateData(int numOfPoints) {
        Point[] points = new Point[numOfPoints];
        for (int i = 0; i < numOfPoints; i++) {
            points[i] = new Point();
        }
        return points;
    }
}
