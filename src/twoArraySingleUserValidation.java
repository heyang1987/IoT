/**
 * @author Yang
 */

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class twoArraySingleUserValidation {
	
	public static ArrayList<Integer> array1 = new ArrayList<>();
	public static ArrayList<Integer> array2 = new ArrayList<>();
	
	public static int classIndex;
	public static DataSource source;
	public static DataSource source1;
	public static DataSource source2;
	
	public static Instances test;
	public static Instances data1;
	public static Instances data2;
	
	public static double maxCorrectPercentage = 0;
	public static FilteredClassifier fc1 = new FilteredClassifier();
	public static FilteredClassifier fc2 = new FilteredClassifier();

	public static FilteredClassifier maxFc1 = new FilteredClassifier();
	public static FilteredClassifier maxFc2 = new FilteredClassifier();

	public static double Array1Accuracy = 0;
	public static double Array2Accuracy = 0;

	public static int finalArray1Size = 0;
	public static int finalArray2Size = 0;

	public void shuffle(ArrayList<Integer> userArray) throws Exception {
		
		int shuffleTimes = 0;
		int userArraySize = userArray.size();
		source = new DataSource("docs/intel_result6_always.arff");	
		test = source.getDataSet();
		classIndex = test.numAttributes()-1;
		test.setClassIndex(classIndex);
		
		System.out.println("2 arrays Start Shuffling...");                
		do{
			System.out.println("Shuffling array 1 and 2 Time: " + shuffleTimes);
			Collections.shuffle(userArray);
			array1 = new ArrayList<>(userArray.subList(0, Math.round(userArraySize/2)));
			array2 = new ArrayList<>(userArray.subList(Math.round(userArraySize/2), userArraySize));
			Collections.sort(array1);
			Collections.sort(array2);
			
			arffFunctions.generateArff(array1, "docs/intel_arff_header_always.txt", "model1.arff");
			arffFunctions.generateArff(array2, "docs/intel_arff_header_always.txt", "model2.arff");
			
			source1 = new DataSource("docs/model1.arff");
			source2 = new DataSource("docs/model2.arff");
			
			data1 = source1.getDataSet();
			data2 = source2.getDataSet();
			
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			
			fc1 = wekaFunctions.train(data1, classIndex);
			fc2 = wekaFunctions.train(data2, classIndex);
			            
			//System.out.println("Array1: " + array1);
			//System.out.println("Array2: " + array2);
			//System.out.println("Array3: " + array3);
			//System.out.println("Array4: " + array4.size());
			//System.out.println("Array1 size: " + array1.size());
			//System.out.println("Array2 size: " + array2.size());
			//System.out.println("Array3 size: " + array3.size());
			//System.out.println("Array4 size: " + array4.size());
			            
			//System.out.println("fc1 size: " + fc1.numElements());
			//System.out.println("fc2 size: " + fc2.numElements());
			
			shuffleTimes++;
		}while ( (fc1.numElements() == 1) && (fc2.numElements() == 1) );		
		System.out.println("2 arrays end Shuffling...");
	}
	

    public void converge() throws Exception{       
        for (int expTimes = 0; expTimes < 200; expTimes++){
		
			Array1Accuracy = wekaFunctions.evalCrossValidation(fc1, data1);
			Array2Accuracy = wekaFunctions.evalCrossValidation(fc2, data2);
//	        System.out.println("=============================================================================");
//			System.out.println("Iteration: " + expTimes);
//			System.out.println("fc1:\n " + fc1);
//			System.out.println("fc2:\n " + fc2);
//			System.out.println("fc1 size: " + fc1.numElements() + "\t" +
//					"Array1's size: " + data1.numInstances() + "\t" +
//					"Array1's accuracy: " + Array1Accuracy);
//			System.out.println("fc2 size: " + fc2.numElements() + "\t" +
//					"Array2's size: " + data2.numInstances() + "\t" +
//					"Array2's accuracy: " + Array2Accuracy);

			
            ArrayList<Integer> array1Backup = new ArrayList<>(array1);  //BACKUPS HAVE BEEN MADE JUST IN CASE WE NEED TO PUT IT BACK IN THE SAME ARRAY
            ArrayList<Integer> array2Backup = new ArrayList<>(array2);

            array1.clear();
            array2.clear();

            for (int i = 0; i < test.numInstances(); i = i + 14) {
                    int userID = (int)test.instance(i).value(0);
                    //System.out.println("#"+(i+14)/14+" ID:"+userID);
                    Instances user = new Instances(test, i, 14);
                    double accuracy1 = wekaFunctions.eval(fc1, data1, user);
                    double accuracy2 = wekaFunctions.eval(fc2, data2, user);

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
                            //System.out.println(array1Backup.contains(userID));
                            if (array1Backup.contains(userID)){
                                    array1.add(userID);
                            }
                            else
                                    array2.add(userID);
                    }	

                    //System.out.println("===================");
            }
			//System.out.println("EqualAccuracyUser#: " + equaAccurayUser);
            System.out.println("========================================================================================================================================================");
	        //equaAccurayUser = 0;
			if ( array1.equals(array1Backup) && array2.equals(array2Backup) ){
				double accuracy = data1.numInstances()*Array1Accuracy/280000 + data2.numInstances()*Array2Accuracy/280000;
//				System.out.println("*****************************************************************************");
//				System.out.println("Arrays converged within " + expTimes + " iterations");
//				System.out.println("Cumulated Correct Percentage: " +  accuracy);
//				System.out.println("*****************************************************************************");
				if (accuracy > maxCorrectPercentage){
					maxCorrectPercentage = accuracy;
					maxFc1 = fc1;
					maxFc2 = fc2;
					finalArray1Size = data1.numInstances();
					finalArray2Size = data2.numInstances();
					
				}
				break;
			}
			
            arffFunctions.generateArff(array1, "docs/intel_arff_header_always.txt", "model1.arff");
            arffFunctions.generateArff(array2, "docs/intel_arff_header_always.txt", "model2.arff");

            source1 = new DataSource("docs/model1.arff");
            source2 = new DataSource("docs/model2.arff");

            data1 = source1.getDataSet();
            data2 = source2.getDataSet();

            data1.setClassIndex(classIndex);
            data2.setClassIndex(classIndex);

            fc1 = wekaFunctions.train(data1, classIndex);
            fc2 = wekaFunctions.train(data2, classIndex);
		}
	}
	
	public static void main(String[] args) throws Exception {
		String filename= "docs/2ClusterSingleUserValidation200.txt";
	    FileWriter fw = new FileWriter(filename,true);
		int index = 0;
		ArrayList<Integer> userArray = arffFunctions.getUserIDArray("docs/intel_result6_always.arff");
		source = new DataSource("docs/intel_result6_always.arff");
		test = source.getDataSet();
		classIndex = test.numAttributes()-1;
		test.setClassIndex(classIndex);
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
			for (int i = 0; i < 200; i++){
				System.out.println("Shuffling times: " + i);
				twoArraySingleUserValidation twoArrayInstance = new twoArraySingleUserValidation();
				twoArrayInstance.shuffle(restUserArray);
				twoArrayInstance.converge();
			}
//    			System.out.println("*****************************************************************************");
//    			System.out.println("Final Statistics:\n");
//    			System.out.println("maxfc1:\n"+maxFc1+"maxfc2:\n"+maxFc2+"maxfc3:\n"+maxFc3);
//    			System.out.println("Array1's size: " +  finalArray1Size + "\t" + "accuracy: " + Array1Accuracy);
//    			System.out.println("Array2's size: " +  finalArray2Size + "\t" + "accuracy: " + Array2Accuracy);
//    			System.out.println("Array3's size: " +  finalArray3Size + "\t" + "accuracy: " + Array3Accuracy);
//    			System.out.println("Max Correct Percentage: " +  maxCorrectPercentage);
//    			System.out.println("*****************************************************************************");
			double acury1 = wekaFunctions.eval(maxFc1, singleUserInstances, singleUserInstances);
			double acury2 = wekaFunctions.eval(maxFc2, singleUserInstances, singleUserInstances);
			System.out.println("2 clusters' accuracy for user#: " + userID);
			System.out.println(acury1);
			System.out.println(acury2);
			
			fw.write(userID + "," + Math.max(acury1, acury2) + "\n");
		}
		fw.close();
	}
}