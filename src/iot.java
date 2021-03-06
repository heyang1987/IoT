import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class iot {
	
	private static int classIndex;
	private static int equaAccurayUser = 0;

	public static void main(String[] args) throws Exception {
		
		ArrayList<Integer> userArray = getUserIDArray();
		// create array list object
		ArrayList<Integer> array1 = new ArrayList<Integer>();
		ArrayList<Integer> array2 = new ArrayList<Integer>();
		
		//FilteredClassifier fc = new FilteredClassifier();
		FilteredClassifier fc1 = new FilteredClassifier();
		FilteredClassifier fc2 = new FilteredClassifier();
		 
		// shuffle the list
		Collections.shuffle(userArray);
		//System.out.println(userArray.size());
		array1 = new ArrayList<Integer>(userArray.subList(0, 100));
		array2 = new ArrayList<Integer>(userArray.subList(100, 200));
		Collections.sort(array1);
		Collections.sort(array2);

		
		DataSource source = new DataSource("docs/intel_result6.arff");	
		Instances test = source.getDataSet();
		classIndex = test.numAttributes()-1;
		test.setClassIndex(classIndex);
		
		//fc = train(test);
		//System.out.println("fc" + fc);
		
		for (int expTimes = 0; expTimes < 10; expTimes++){

			System.out.println("Iteration: " + expTimes);
			System.out.println("Array1: " + array1);
			System.out.println("Array2: " + array2);
			generateArff(array1, "model1.arff");
			generateArff(array2, "model2.arff");
			DataSource source1 = new DataSource("docs/model1.arff");
			DataSource source2 = new DataSource("docs/model2.arff");
			
			Instances data1 = source1.getDataSet();
			Instances data2 = source2.getDataSet();
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			fc1 = train(data1);
			fc2 = train(data2);
			//System.out.println("fc1:\n" + fc1);
			//System.out.println("fc2:\n" + fc2);
			
			System.out.println("Array1's size: " + data1.numInstances() + "\t" +
					"Array1's accuracy: " + evalCrossValidation(fc1, data1));
			System.out.println("Array2's size: " + data2.numInstances() + "\t" +
					"Array2's accuracy: " + evalCrossValidation(fc2, data2));

			ArrayList<Integer> array1Backup = new ArrayList<Integer>(array1);  //BACKUPS HAVE BEEN MADE JUST IN CASE WE NEED TO PUT IT BACK IN THE SAME ARRAY
			ArrayList<Integer> array2Backup = new ArrayList<Integer>(array2);
			array1.clear();
			array2.clear();
			
			for (int i = 0; i < test.numInstances(); i = i + 14) {
				int userID = (int)test.instance(i).value(0);
				//System.out.println("#"+(i+14)/14+" ID:"+userID);
				Instances user = new Instances(test, i, 14);
				double accuracy1 = 0;
				double accuracy2 = 0;
				
				accuracy1 = eval(fc1, data1, user);
				accuracy2 = eval(fc2, data2, user);
				//System.out.println(accuracy1);
				//System.out.println(accuracy2);
				//System.out.println();
				if (accuracy1 > accuracy2){
					array1.add(userID);
				}
				else if (accuracy1 < accuracy2){
					array2.add(userID);
				}
				else if (accuracy1 == accuracy2) {
					equaAccurayUser++;
					//System.out.println(array1Backup.contains(userID));
					if (array1Backup.contains(userID)){
						array1.add(userID);
					}
					else
						array2.add(userID);
				}		
				//System.out.println("===================");
			}
			System.out.println("EqualAccuracyUser#: " + equaAccurayUser);
			System.out.println("===================");
			equaAccurayUser = 0;
			if ( array1.equals(array1Backup) && array2.equals(array2Backup)){
				System.out.println("Arrays converged within " + expTimes + " iterations");
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
	
	
	public static FilteredClassifier train(Instances train) throws Exception
	{
		train.setClassIndex(classIndex);
		
		Remove rm = new Remove();
		rm.setAttributeIndices("1");  // REMOVING ID ATTRIBUTE AS THAT WON'T BE INPUT TO THE CLASSIFIER
		// classifier
		J48 j48 = new J48();
		//j48.setUnpruned(true);        // using an unpruned J48
		// meta-classifier
		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(j48);
		// train
		fc.buildClassifier(train);
		return fc;
		
	}
	
	public static double eval(FilteredClassifier fc, Instances train, Instances test)  throws Exception
	{
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(fc, test);
		return eval.pctCorrect();
	}

	public static double evalCrossValidation(FilteredClassifier fc, Instances data) throws Exception
	{
		Random random = new Random();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(fc, data, 10, random);
		return eval.pctCorrect();
	}


}
