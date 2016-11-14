/**
 * @author Yang
 */

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class BottomUp {
	
	public static DataSource source;
	public static int classIndex;
	
	public static void main(String[] args) throws Exception {
		
		int counter1 = 0;
		int counter2 = 0;
		source = new DataSource("docs/intel_result6.arff");
		Instances inst = source.getDataSet();
		classIndex = inst.numAttributes()-1;
		inst.setClassIndex(classIndex);
		
		System.out.println(inst.numInstances());
		
		for (int i = 0; i < inst.numInstances(); i = i + 14) {
			int userID = (int)inst.instance(i).value(0);
			Instances user = new Instances(inst, i, 14);
			FilteredClassifier cls = wekaFunctions.train(user);
			//double accuracy = wekaFunctions.eval(cls,user, user);
			double accuracy = wekaFunctions.evalCrossValidation(cls, user);
			System.out.println("user#: " + userID);
			//System.out.println("Node number of classifier: " + cls.numElements());
			System.out.println("Classifier: " + cls);
			System.out.println("Accuracy: " + accuracy);
			if ( cls.numElements() == 1 ){
				counter1 ++;
			}
			if ( accuracy == 100 ){
				counter2++;
			}
		}
		System.out.println("One Node Tree#: " + counter1);
		System.out.println("100 Percent Accuracy classifier#: " + counter2);
    
	}
}
