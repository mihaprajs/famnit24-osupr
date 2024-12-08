package naive_bayes;

import java.util.ArrayList;
import java.util.Arrays;

class NaiveBayes {
    private ColumnTable[] columnTables;
    private ClassTable classTable;

    public void train(String[][] X, String[] Y) {
        int numOfCols = X[0].length;
        columnTables = new ColumnTable[numOfCols];

        for (int colIndex = 0; colIndex < numOfCols; colIndex++) {
            String[] column = DataHelper.extractColumn(X, colIndex);
            columnTables[colIndex] = new ColumnTable(column, Y);
        }

        classTable = new ClassTable(Y);
    }

    public String predict(String[] valuseToPredict) {
        ArrayList<String> classValues = classTable.getClassValues();
        double mostProbableValue = Double.MIN_VALUE;
        String mostProbableClass = "";

        for (String classValue : classValues) {
            double probability = 1.0;
            for (int columnIndex = 0; columnIndex < columnTables.length; columnIndex++) {
                probability *= columnTables[columnIndex].getProbability(valuseToPredict[columnIndex], classValue);
            }
            probability *= classTable.getProbability(classValue);

            System.out.println("Class value:" + classValue + ", " + probability);
            if (mostProbableValue < probability) {
                mostProbableValue = probability;
                mostProbableClass = classValue;
            }
        }

        return mostProbableClass;
    }

    public void display() {
        for (ColumnTable table : columnTables) {
            table.display();
        }
    }
}

class ColumnTable{
    private int[][] counts;
    private ArrayList<String> columnValues;
    private ArrayList<String> classValues;

    public ColumnTable(String[] column, String[] Y) {
        columnValues = DataHelper.unique(column);
        classValues = DataHelper.unique(Y);
        counts = new int[columnValues.size()][classValues.size()];

        for (int row = 0; row < column.length; row++) {
            String columnValue = column[row];
            String classValue = Y[row];
            int countRow = columnValues.indexOf(columnValue);
            int countColumn = classValues.indexOf(classValue);

            counts[countRow][countColumn]++;
        }
    }

    public double getProbability(String columnValue, String classValue) {
        int rowIndex = columnValues.indexOf(columnValue);
        int colIndex = classValues.indexOf(classValue);
        int count = counts[rowIndex][colIndex];
        int sum = 0;

        for (int row = 0; row< columnValues.size(); row++) {
            sum += counts[row][colIndex];
        }
        return 1.0 * count / sum;
    }

    public void display() {
        for (int row = 0; row < counts.length; row++) {
            System.out.print(columnValues.get(row) + ": ");
            System.out.println(Arrays.toString(counts[row]));
        }
    }
}

class ClassTable{
    private int[] counts;
    private ArrayList<String> classValues;

    public ClassTable(String[] Y){
        classValues = DataHelper.unique(Y);
        counts = new int[classValues.size()];

        for (String classValue : Y) {
            int classIndex = classValues.indexOf(classValue);
            counts[classIndex]++;
        }
    }
    public ArrayList<String> getClassValues() {
        return classValues;
    }

    public double getProbability(String classValue) {
        int index = classValues.indexOf(classValue);
        int count = counts[index];
        int sum = 0;
        for (int column = 0; column < classValues.size(); column++) {
            sum += counts[column];
        }
        return 1.0 * count / sum;
    }
}