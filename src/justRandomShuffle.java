import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class justRandomShuffle {
	
	private static int classIndex;
	private static double accuracy1Max = 0;
	private static double accuracy2Max = 0;
	
	public static ArrayList<Integer> getUserIDArray() throws Exception {
		ArrayList<Integer> userArray = new ArrayList<Integer>();
		DataSource source = new DataSource("docs/intel_result6.arff");
		Instances data = source.getDataSet();

		for (int i = 0; i < data.numInstances(); i= i + 14) {
			userArray.add((int)data.instance(i).value(0));
		}
		//System.out.println(userArray);
		return userArray;
	}
					
	// APPENDING AND GENERATING .ARFF FILES
	public static void generateArff(ArrayList<Integer> array, String fileName) throws IOException{
	    String source = "docs"+File.separator+"intel_arff_header.txt";
	    String dest = "docs"+ File.separator + fileName;
	    File fin = new File(source);
	    File fout = new File(dest);
	    
	    PrintWriter writer = new PrintWriter(fout);
	    FileInputStream fis = new FileInputStream(fin);
	    BufferedReader in = new BufferedReader(new InputStreamReader(fis));

	    String aLine = null;
	    while ((aLine = in.readLine()) != null) {
	        //Process each line and add output to Dest.txt file
	    	writer.println(aLine);
	    }
	    	    
	    // close the buffer reader
	    in.close();
	    for (int i : array) {
		    fin = new File("users"+File.separator+i+".txt");
		    fis = new FileInputStream(fin);
		    in = new BufferedReader(new InputStreamReader(fis));
		    while ((aLine = in.readLine()) != null) {
		        // output to Dest.txt file
		    	writer.println(aLine);
		    }
		    in.close();
	    }
	    // close buffer writer
	    writer.close();
	}
	
	// EVALUATION METHOD
	public static double trainAndEval(Instances train, Instances test) throws Exception
	{
		train.setClassIndex(classIndex);
		test.setClassIndex(classIndex);
		
		Remove rm = new Remove();
		rm.setAttributeIndices("1");  // REMOVING ID ATTRIBUTE AS THAT WON'T BE INPUT TO THE CLASSIFIER
		// classifier
		J48 j48 = new J48();
		j48.setUnpruned(true);        // using an unpruned J48
		// meta-classifier
		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(j48);
		// train
		fc.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(fc, test);
		return eval.pctCorrect();
		
	}
	
	public static void main(String[] args) throws Exception {
		
		ArrayList<Integer> userArray = getUserIDArray();
		// create array list object
		ArrayList<Integer> array1 = new ArrayList<Integer>();
		ArrayList<Integer> array2 = new ArrayList<Integer>();
		double accuracy1 = 0;
		double accuracy2 = 0;

		 
		for (int expTimes = 0; expTimes < 20000; expTimes++){
			userArray = getUserIDArray();
			// shuffle the list
			Collections.shuffle(userArray);
			//System.out.println("iteration: " + (expTimes+1));
			//System.out.println("ArrayU: " + userArray);
			array1 = new ArrayList<Integer>(userArray.subList(0, 100));
			array2 = new ArrayList<Integer>(userArray.subList(100, 200));
			Collections.sort(array1);
			Collections.sort(array2);
			
			//System.out.println("Array1: " + array1);
			//System.out.println("Array2: " + array2);
		
			generateArff(array1, "model1.arff");
			generateArff(array2, "model2.arff");
			
			DataSource source = new DataSource("docs/intel_result6.arff");
			DataSource source1 = new DataSource("docs/model1.arff");
			DataSource source2 = new DataSource("docs/model2.arff");
			
			Instances test = source.getDataSet();
			Instances data1 = source1.getDataSet();
			Instances data2 = source2.getDataSet();
			
			classIndex = test.numAttributes()-1;
			test.setClassIndex(classIndex);
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			accuracy1 = trainAndEval(data1, test);
			accuracy2 = trainAndEval(data2, test);
			//System.out.println("Array1's accuracy: " + accuracy1);
			//System.out.println("Array2's accuracy: " + accuracy2);
			//System.out.println((accuracy1 > accuracy1Max));
			//System.out.println((accuracy2 > accuracy2Max));
			if (  (accuracy1 > accuracy1Max) == true && (accuracy2 > accuracy2Max) == true ){
				accuracy1Max = accuracy1;
				accuracy2Max = accuracy2;
			}
			
			//System.out.println("Array1's Max accuracy: " + accuracy1Max);
			//System.out.println("Array2's Max accuracy: " + accuracy2Max);
			System.out.println(expTimes);
		}
		System.out.println("Final Array1's Max accuracy: " + accuracy1Max);
		System.out.println("Final Array2's Max accuracy: " + accuracy2Max);
			
	}

}
