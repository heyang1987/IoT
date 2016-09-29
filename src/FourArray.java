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

public class FourArray {
	
	private static int classIndex;
	//private static int equaAccurayUsers = 0;

	public static void main(String[] args) throws Exception {
		
		ArrayList<Integer> userArray = getUserIDArray();
		// create array list object
		ArrayList<Integer> array1 = new ArrayList<Integer>();
		ArrayList<Integer> array2 = new ArrayList<Integer>();
		ArrayList<Integer> array3 = new ArrayList<Integer>();
		ArrayList<Integer> array4 = new ArrayList<Integer>();
		 
		// shuffle the list
		Collections.shuffle(userArray);
		//System.out.println(userArray.size());
		array1 = new ArrayList<Integer>(userArray.subList(0, 50));
		array2 = new ArrayList<Integer>(userArray.subList(50, 100));
		array3 = new ArrayList<Integer>(userArray.subList(100, 150));
		array4 = new ArrayList<Integer>(userArray.subList(150, 200));
		
		Collections.sort(array1);
		Collections.sort(array2);
		Collections.sort(array3);
		Collections.sort(array4);
		
		DataSource source = new DataSource("docs/intel_result6.arff");
		DataSource source1 = new DataSource("docs/model1.arff");
		DataSource source2 = new DataSource("docs/model2.arff");
		DataSource source3 = new DataSource("docs/model3.arff");
		DataSource source4 = new DataSource("docs/model4.arff");
		
		Instances test = source.getDataSet();
		classIndex = test.numAttributes()-1;
		test.setClassIndex(classIndex);
		
		for (int expTimes = 0; expTimes < 200; expTimes++){
			System.out.println("Iteration: " + expTimes);
			
			System.out.println("Array1: " + array1);
			System.out.println("Array2: " + array2);
			System.out.println("Array3: " + array3);
			System.out.println("Array4: " + array4);
			

			generateArff(array1, "model1.arff");
			generateArff(array2, "model2.arff");
			generateArff(array2, "model3.arff");
			generateArff(array2, "model4.arff");
			

			Instances data1 = source1.getDataSet();
			Instances data2 = source2.getDataSet();
			Instances data3 = source3.getDataSet();
			Instances data4 = source4.getDataSet();
			

			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			data3.setClassIndex(classIndex);
			data4.setClassIndex(classIndex);
			

			System.out.println("Array1's accuracy: " + trainAndEval(data1, data1));
			System.out.println("Array2's accuracy: " + trainAndEval(data2, data2));
			System.out.println("Array3's accuracy: " + trainAndEval(data3, data3));
			System.out.println("Array4's accuracy: " + trainAndEval(data4, data4));
			
			// print the accuracy of the two models in general
			ArrayList<Integer> array1Backup = new ArrayList<Integer>(array1);
			ArrayList<Integer> array2Backup = new ArrayList<Integer>(array2);
			ArrayList<Integer> array3Backup = new ArrayList<Integer>(array3);
			ArrayList<Integer> array4Backup = new ArrayList<Integer>(array4);
			array1.clear();
			array2.clear();
			array3.clear();
			array4.clear();
			
			for (int i = 0; i < test.numInstances(); i = i + 14) {
				int userID = (int)test.instance(i).value(0);
				//System.out.println("#"+(i+14)/14+" ID:"+userID);
				double accuracy1 = 0;
				double accuracy2 = 0;
				double accuracy3 = 0;
				double accuracy4 = 0;
				
				accuracy1 = trainAndEval(data1, new Instances(test,i, 14));
				accuracy2 = trainAndEval(data2, new Instances(test,i, 14));
				accuracy3 = trainAndEval(data3, new Instances(test,i, 14));
				accuracy4 = trainAndEval(data4, new Instances(test,i, 14));
					
				//System.out.println(accuracy1);
				//System.out.println(accuracy2);
				//System.out.println(accuracy3);
				//System.out.println(accuracy4);

				if (array1Backup.contains(userID)){
					//System.out.println(userID + " Belongs to: Array1");
					if (accuracy1 >= Math.max( Math.max(accuracy2, accuracy3), accuracy4) )
						array1.add(userID);
					else if (accuracy4 >= Math.max(accuracy1, accuracy2))
						array4.add(userID);
					else if (accuracy3 >= Math.max(accuracy2, accuracy4))
						array3.add(userID);
					else if (accuracy2 >= Math.max(accuracy1, accuracy4))
						array2.add(userID);
				}
				else if (array2Backup.contains(userID)){
					//System.out.println(userID + " Belongs to: Array2");
					if (accuracy2 >= Math.max( Math.max(accuracy4, accuracy3), accuracy1) )
						array2.add(userID);
					else if (accuracy3 >= Math.max(accuracy1, accuracy4))
						array3.add(userID);
					else if (accuracy4 >= Math.max(accuracy1, accuracy3))
						array4.add(userID);
					else if (accuracy1 >= Math.max(accuracy3, accuracy4))
						array1.add(userID);
				}
				else if (array3Backup.contains(userID)){
					//System.out.println(userID + " Belongs to: Array3");
					if (accuracy3 >= Math.max( Math.max(accuracy2, accuracy4), accuracy1) )
						array3.add(userID);
					else if (accuracy2 >= Math.max(accuracy1, accuracy4))
						array2.add(userID);
					else if (accuracy4 >= Math.max(accuracy1, accuracy2))
						array4.add(userID);
					else if (accuracy1 >= Math.max(accuracy2, accuracy4))
						array1.add(userID);
				}
				else if (array4Backup.contains(userID)){
					//System.out.println(userID + " Belongs to: Array4");
					if (accuracy4 >= Math.max( Math.max(accuracy2, accuracy3), accuracy1) )
						array4.add(userID);
					else if (accuracy3 >= Math.max(accuracy1, accuracy2))
						array3.add(userID);
					else if (accuracy2 >= Math.max(accuracy1, accuracy3))
						array2.add(userID);
					else if (accuracy1 >= Math.max(accuracy2, accuracy3))
						array1.add(userID);
				}
				//System.out.println("Array1: " + array1);
				//System.out.println("Array2: " + array2);
				//System.out.println("Array3: " + array3);
				//System.out.println("Array4: " + array4);
		
				//System.out.println("===================");
			}
			System.out.println("========================================================================================================================================================");
			if ( array1.equals(array1Backup) && array2.equals(array2Backup) && array3.equals(array3Backup) && array4.equals(array4Backup)){
				System.out.println("Arrays converged in " + expTimes + " iterations");
				break;
			}
		}
		
	}
	
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
	
	public static double trainAndEval(Instances train, Instances test) throws Exception{
		train.setClassIndex(classIndex);
		test.setClassIndex(classIndex);
		
		Remove rm = new Remove();
		rm.setAttributeIndices("1");  // remove 1st attribute
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

}
