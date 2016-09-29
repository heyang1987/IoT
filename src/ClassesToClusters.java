import java.io.*;
import weka.core.*;
import weka.clusterers.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This class shows how to perform a "classes-to-clusters"
 * evaluation like in the Explorer using EM. The class needs as
 * first parameter an ARFF file to work on. The last attribute is
 * interpreted as the class attribute. 
 * <p/>
 * This code is based on the method "startClusterer" of the 
 * "weka.gui.explorer.ClustererPanel" class and the 
 * "evaluateClusterer" method of the "weka.clusterers.ClusterEvaluation" 
 * class.
 *
 * @author  FracPete (fracpete at waikato dot ac dot nz)
 */
public class ClassesToClusters {
  public static void main(String[] args) throws Exception {
    // load data
    Instances data1 = new Instances(new BufferedReader(new FileReader(args[0])));
    Remove filter1 = new Remove();
    filter1.setAttributeIndices("1");
    filter1.setInputFormat(data1);
    Instances data = Filter.useFilter(data1, filter1);

    data.setClassIndex(data.numAttributes() - 1);

    // generate data for clusterer (w/o class)
    Remove filter = new Remove();
    filter.setAttributeIndices("" + (data.classIndex() + 1));
    filter.setInputFormat(data);
    Instances dataClusterer = Filter.useFilter(data, filter);

    // train clusterer
    //EM clusterer = new EM();
    HierarchicalClusterer clusterer = new HierarchicalClusterer();
    String[] options = new String[5];
    options[0] = "-N";
    options[1] = "2";
    options[2] = "-L";
    options[3] = "SINGLE";
    options[4] = "-P";
    // set further options for EM, if necessary...
    clusterer.setOptions(options);
    clusterer.buildClusterer(dataClusterer);

    // evaluate clusterer
    ClusterEvaluation eval = new ClusterEvaluation();
    eval.setClusterer(clusterer);
    eval.evaluateClusterer(data);

    // print results
    System.out.println(eval.clusterResultsToString());
  }
}
