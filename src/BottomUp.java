/**
 * @author Yang
 */

import java.util.ArrayList;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class BottomUp {
	
	public static DataSource source;
	public static int classIndex;
	public static ArrayList<Integer> array1 = new ArrayList<>();
	public static ArrayList<Integer> array2 = new ArrayList<>();
    public static ArrayList<Integer> array3 = new ArrayList<>();
	
	public static void main(String[] args) throws Exception {
		
		int counter1 = 0;
		int counter2 = 0;
		int c1=0,c2=0,c3=0,c4=0,c5=0,c6=0;
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
			String abm=cls.graph();  //can store this in a string and use string functions to parse it; put in if conditions to determine the clusters
            //System.out.println(abm);
			//System.out.println("Classifier: " + cls);
			System.out.println("Accuracy: " + accuracy);
			if ( cls.numElements() == 1 ){
				counter1 ++;
			}
			if ( accuracy == 100 ){
				counter2++;
			}
			String t1="N0 [label=\"no";
            String t2="N0 [label=\"yes";

           
            if(abm.contains(t1))
            {
                c1++;
                array1.add(userID);
            }
            if(abm.contains(t2))
            {
                c2++;
                array2.add(userID);
            }
            if(abm.contains(t1)==false && abm.contains(t2)==false)
            {
                c3++;
                array3.add(userID);
            }
		}
	    
	    System.out.println("Single 'NO' node trees :" +c1);
	    System.out.println("Single 'YES' node trees :" +c2);
	    System.out.println("Multi-node trees :" +c3);
	    
	    System.out.println("Users with multi-node trees :" +array3);
	
	    arffFunctions.generateArff(array1, "docs/intel_arff_header.txt", "bottomsup1.arff");
	    arffFunctions.generateArff(array2, "docs/intel_arff_header.txt", "bottomsup2.arff");
	    arffFunctions.generateArff(array3, "docs/intel_arff_header.txt", "bottomsup3.arff");
		System.out.println("One Node Tree#: " + counter1);
		System.out.println("100 Percent Accuracy classifier#: " + counter2);
    
	}
}
