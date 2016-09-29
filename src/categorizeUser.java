import java.io.File;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;

public class categorizeUser {

	public static void categorizeUserToCluster(String[] args) throws Exception {
		//DataSource source1 = new DataSource("docs/2clusters.arff");
		//DataSource source1 = new DataSource("docs/3clusters.arff");
		DataSource source1 = new DataSource("docs/5clusters.arff");
		
		DataSource source2 = new DataSource("docs/intel_result6_4clusters.arff");
		
		Instances data1 = source1.getDataSet();
		Instances data2 = source2.getDataSet();
		
		//data1.setClassIndex(data1.numAttributes()-1);
		//data2.setClassIndex(data2.numAttributes()-1);
		
		for (int i = 0; i < data1.numInstances(); i++) {
			//System.out.println(data1.instance(i).stringValue(data1.numAttributes()-1));
			for (int j = 14*i; j < 14*i + 14; j++) {
				data2.instance(j).setValue(data2.numAttributes()-1, data1.instance(i).stringValue(data1.numAttributes()-1));
			}
			//data2CurrentIndex = j;
		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data2);
		//saver.setFile(new File("docs/intel_result6_2clusters.arff"));
		//saver.setFile(new File("docs/intel_result6_3clusters.arff"));
		saver.setFile(new File("docs/intel_result6_5clusters.arff"));
		saver.writeBatch();
		System.out.println("Success!");
	}
}
