import java.io.FileWriter;
import java.util.ArrayList;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class oneArraySingleUserValidation {

	public static int classIndex;
	
	public static void main(String[] args) throws Exception {
		String filename= "docs/OneClusterSingleUserValidation.txt";
	    FileWriter fw = new FileWriter(filename,true);
		int index = 0;
		ArrayList<Integer> userArray = arffFunctions.getUserIDArray("docs/intel_result6_always.arff");
		DataSource source = new DataSource("docs/intel_result6_always.arff");
		Instances allUser = source.getDataSet();
		classIndex = allUser.numAttributes()-1;
		allUser.setClassIndex(classIndex);
		for(index = 0; index < userArray.size(); index++){
			int userID = userArray.get(index);
			ArrayList<Integer> restUserArray = new ArrayList<Integer>(userArray);
			ArrayList<Integer> singleUser = new ArrayList<Integer>();
			
			singleUser.add(userID);
			arffFunctions.generateArff(singleUser, "docs/intel_arff_header_always.txt", "singleUser.arff");
			DataSource sourceSingleUser = new DataSource("docs/singleUser.arff");
			Instances singleUserInstances = sourceSingleUser.getDataSet();
			singleUserInstances.setClassIndex(classIndex);
			
			restUserArray.remove(index);
			arffFunctions.generateArff(restUserArray, "docs/intel_arff_header_always.txt", "restUser.arff");
			DataSource sourceRestUser = new DataSource("docs/restUser.arff");
			Instances restUserInstances = sourceRestUser.getDataSet();
			restUserInstances.setClassIndex(classIndex);
			
			FilteredClassifier cls = wekaFunctions.train(restUserInstances, classIndex);
			double accuracy = wekaFunctions.eval(cls, restUserInstances, singleUserInstances);

			System.out.println("One clusters' accuracy for user -- " + userID + ": " + accuracy);
			
			fw.write(userID + "," + accuracy + "\n");
		}
		fw.close();
	}
}
