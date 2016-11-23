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

public class threeArraySingleUserValidation {
	
	public static ArrayList<Integer> array1 = new ArrayList<>();
	public static ArrayList<Integer> array2 = new ArrayList<>();
	public static ArrayList<Integer> array3 = new ArrayList<>();
	
	public static int classIndex;
	public static DataSource source;
	public static DataSource source1;
	public static DataSource source2;
	public static DataSource source3;
	
	public static Instances test;
	public static Instances data1;
	public static Instances data2;
	public static Instances data3;
	
	public static double maxCorrectPercentage = 0;
	public static FilteredClassifier fc1 = new FilteredClassifier();
	public static FilteredClassifier fc2 = new FilteredClassifier();
	public static FilteredClassifier fc3 = new FilteredClassifier();

	public static FilteredClassifier maxFc1 = new FilteredClassifier();
	public static FilteredClassifier maxFc2 = new FilteredClassifier();
	public static FilteredClassifier maxFc3 = new FilteredClassifier();

	public static double Array1Accuracy = 0;
	public static double Array2Accuracy = 0;
	public static double Array3Accuracy = 0;

	public static int finalArray1Size = 0;
	public static int finalArray2Size = 0;
	public static int finalArray3Size = 0;
	
	public void shuffle(ArrayList<Integer> userArray) throws Exception {

		int shuffleTimes = 0;
		int userArraySize = userArray.size();
		
		//System.out.println("3 arrays Start Shuffling...");
		do{
			//System.out.println("Shuffling array 1, 2, and 3 Time: " + shuffleTimes);
			Collections.shuffle(userArray);
			array1 = new ArrayList<>(userArray.subList(0, Math.round(userArraySize/3)));
			array2 = new ArrayList<>(userArray.subList(Math.round(userArraySize/3), Math.round(userArraySize*2/3)));
			array3 = new ArrayList<>(userArray.subList(Math.round(userArraySize*2/3), userArraySize));
			Collections.sort(array1);
			Collections.sort(array2);
			Collections.sort(array3);
			
			arffFunctions.generateArff(array1, "docs/intel_arff_header_always.txt", "model1.arff");
			arffFunctions.generateArff(array2, "docs/intel_arff_header_always.txt", "model2.arff");
			arffFunctions.generateArff(array3, "docs/intel_arff_header_always.txt", "model3.arff");
			
			source1 = new DataSource("docs/model1.arff");
			source2 = new DataSource("docs/model2.arff");
			source3 = new DataSource("docs/model3.arff");
			
			data1 = source1.getDataSet();
			data2 = source2.getDataSet();
			data3 = source3.getDataSet();
			
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			data3.setClassIndex(classIndex);
			
			fc1 = wekaFunctions.train(data1, classIndex);
			fc2 = wekaFunctions.train(data2, classIndex);
			fc3 = wekaFunctions.train(data3, classIndex);
			//System.out.println("Array1: " + array1);
			//System.out.println("Array2: " + array2);
			//System.out.println("Array3: " + array3);
			//System.out.println("Array1 size: " + array1.size());
			//System.out.println("Array2 size: " + array2.size());
			//System.out.println("Array3 size: " + array3.size());
			//System.out.println("fc1 size: " + fc1.numElements());
			//System.out.println("fc2 size: " + fc2.numElements());
			//System.out.println("fc3 size: " + fc3.numElements());
			shuffleTimes++;
		}while ( (fc1.numElements() == 1) && (fc2.numElements() ==1) && (fc3.numElements() ==1) );
		
		//System.out.println("Array1: " + array1);
		//System.out.println("Array2: " + array2);
		//System.out.println("Array3: " + array3);
		
		while ( (fc1.numElements() == 1) && (fc2.numElements() ==1)){
			//System.out.println("Shuffling array 1 and 2 Time: " + (shuffleTimes++));
			ArrayList<Integer> array = new ArrayList<>(array1);
			array.addAll(array2);
			Collections.shuffle(array);
			array1 = new ArrayList<>(array.subList(0, 66));
			array2 = new ArrayList<>(array.subList(66, array.size()));
			Collections.sort(array1);
			Collections.sort(array2);

			//System.out.println("Array1: " + array1);
			//System.out.println("Array2: " + array2);
			
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
			
			//System.out.println("Array1 size: " + array1.size());
			//System.out.println("Array2 size: " + array2.size());
			//System.out.println("Array3 size: " + array3.size());
			//System.out.println("fc1 size: " + fc1.numElements());
			//System.out.println("fc2 size: " + fc2.numElements());
			//System.out.println("fc3 size: " + fc3.numElements());
		}
		while ( (fc1.numElements() == 1) && (fc3.numElements() ==1)){
			//System.out.println("Shuffling array 1 and 3 Time: " + (shuffleTimes++));
			ArrayList<Integer> array = new ArrayList<>(array1);
			array.addAll(array3);
			Collections.shuffle(array);
			array1 = new ArrayList<>(array.subList(0, 66));
			array3 = new ArrayList<>(array.subList(66, array.size()));
			Collections.sort(array1);
			Collections.sort(array3);

			//System.out.println("Array1: " + array1);
			//System.out.println("Array3: " + array3);
			
			arffFunctions.generateArff(array1, "docs/intel_arff_header_always.txt", "model1.arff");
			arffFunctions.generateArff(array3, "docs/intel_arff_header_always.txt", "model3.arff");
			
			source1 = new DataSource("docs/model1.arff");
			source3 = new DataSource("docs/model3.arff");
			
			data1 = source1.getDataSet();
			data3 = source3.getDataSet();
			
			data1.setClassIndex(classIndex);
			data3.setClassIndex(classIndex);
			
			fc1 = wekaFunctions.train(data1, classIndex);
			fc3 = wekaFunctions.train(data3, classIndex);
//				System.out.println("Array1 size: " + array1.size());
//				System.out.println("Array2 size: " + array2.size());
//				System.out.println("Array3 size: " + array3.size());
			//System.out.println("fc1 size: " + fc1.numElements());
			//System.out.println("fc2 size: " + fc2.numElements());
			//System.out.println("fc3 size: " + fc3.numElements());
		}
		while ( (fc3.numElements() == 1) && (fc2.numElements() ==1)){
			//System.out.println("Shuffling array 2 and 3 Time: " + (shuffleTimes++));
			ArrayList<Integer> array = new ArrayList<>(array2);
			array.addAll(array3);
			Collections.shuffle(array);
			array2 = new ArrayList<>(array.subList(0, 66));
			array3 = new ArrayList<>(array.subList(66, array.size()));
			Collections.sort(array2);
			Collections.sort(array3);

			//System.out.println("Array2: " + array2);
			//System.out.println("Array3: " + array3);
			
			arffFunctions.generateArff(array2, "docs/intel_arff_header_always.txt", "model2.arff");
			arffFunctions.generateArff(array3, "docs/intel_arff_header_always.txt", "model3.arff");
			
			source2 = new DataSource("docs/model2.arff");
			source3 = new DataSource("docs/model3.arff");
			
			data2 = source2.getDataSet();
			data3 = source3.getDataSet();
			
			data2.setClassIndex(classIndex);
			data3.setClassIndex(classIndex);
			
			fc2 = wekaFunctions.train(data2, classIndex);
			fc3 = wekaFunctions.train(data3, classIndex);
			
//				System.out.println("Array1 size: " + array1.size());
//				System.out.println("Array2 size: " + array2.size());
//				System.out.println("Array3 size: " + array3.size());
			//System.out.println("fc1 size: " + fc1.numElements());
			//System.out.println("fc2 size: " + fc2.numElements());
			//System.out.println("fc3 size: " + fc3.numElements());
		}		
		//System.out.println("3 arrays End Shuffling...");
	}
			
		
	public void converge() throws Exception{
		Random randomNum = new Random();
		for (int expTimes = 0; expTimes < 200; expTimes++){
			
			Array1Accuracy = wekaFunctions.evalCrossValidation(fc1, data1);
			Array2Accuracy = wekaFunctions.evalCrossValidation(fc2, data2);
			Array3Accuracy = wekaFunctions.evalCrossValidation(fc3, data3);
			//System.out.println("=============================================================================");
			//System.out.println("Iteration: " + expTimes);
			//System.out.println("fc1:\n " + fc1);
			//System.out.println("fc2:\n " + fc2);
			//System.out.println("fc3:\n " + fc3);
//			System.out.println("fc1 size: " + fc1.numElements() + "\t" +
//					"Array1's size: " + data1.numInstances() + "\t" +
//					"Array1's accuracy: " + Array1Accuracy);
//			System.out.println("fc2 size: " + fc2.numElements() + "\t" +
//					"Array2's size: " + data2.numInstances() + "\t" +
//					"Array2's accuracy: " + Array2Accuracy);
//			System.out.println("fc3 size: " + fc3.numElements() + "\t" +
//					"Array3's size: " + data3.numInstances() + "\t" +
//					"Array3's accuracy: " + Array3Accuracy);
			
			ArrayList<Integer> array1Backup = new ArrayList<>(array1);  //BACKUPS HAVE BEEN MADE JUST IN CASE WE NEED TO PUT IT BACK IN THE SAME ARRAY
			ArrayList<Integer> array2Backup = new ArrayList<>(array2);
			ArrayList<Integer> array3Backup = new ArrayList<>(array3);
			
			array1.clear();
			array2.clear();
			array3.clear();
			
			for (int i = 0; i < test.numInstances(); i = i + 14) {
				int userID = (int)test.instance(i).value(0);
				//System.out.println("#"+(i+14)/14+" ID:"+userID);
				Instances user = new Instances(test, i, 14);
				double accuracy1 = 0;
				double accuracy2 = 0;
				double accuracy3 = 0;
				
				accuracy1 = wekaFunctions.eval(fc1, data1, user);
				accuracy2 = wekaFunctions.eval(fc2, data2, user);
				accuracy3 = wekaFunctions.eval(fc3, data3, user);
				//System.out.println(accuracy1);
				//System.out.println(accuracy2);
				//System.out.println();
				

				if (array1Backup.contains(userID)){
					//System.out.println(userID + " Belongs to: Array1");
					if (accuracy1 >= Math.max(accuracy2, accuracy3))
						array1.add(userID);
					else if (accuracy2 > accuracy3)
						array2.add(userID);
					else if (accuracy3 > accuracy2)
						array3.add(userID);
					else if (accuracy2 == accuracy3)
					{
						if (randomNum.nextInt() % 2 == 0)
							array2.add(userID);
						else
							array3.add(userID);
					}
				}
				else if (array2Backup.contains(userID)){
					//System.out.println(userID + " Belongs to: Array1");
					if (accuracy2 >= Math.max(accuracy1, accuracy3))
						array2.add(userID);
					else if (accuracy1 > accuracy3)
						array1.add(userID);
					else if (accuracy3 > accuracy1)
						array3.add(userID);
					else if (accuracy1 == accuracy3)
					{
						if (randomNum.nextInt() % 2 == 0)
							array1.add(userID);
						else
							array3.add(userID);
					}
				}
				else if (array3Backup.contains(userID)){
					//System.out.println(userID + " Belongs to: Array1");
					if (accuracy3 >= Math.max(accuracy2, accuracy1))
						array3.add(userID);
					else if (accuracy2 > accuracy1)
						array2.add(userID);
					else if (accuracy1 > accuracy2)
						array1.add(userID);
					else if (accuracy1 == accuracy2)
					{
						if (randomNum.nextInt() % 2 == 0)
							array1.add(userID);
						else
							array2.add(userID);
					}
				}
		
				//System.out.println("===================");
			}
			//System.out.println("EqualAccuracyUser#: " + equaAccurayUser);
			//equaAccurayUser = 0;
			if ( array1.equals(array1Backup) && array2.equals(array2Backup) && array3.equals(array3Backup)){
				double accuracy = data1.numInstances()*Array1Accuracy/280000 + data2.numInstances()*Array2Accuracy/280000 + data3.numInstances()*Array3Accuracy/280000;
				//System.out.println("*****************************************************************************");
				//System.out.println("Arrays converged within " + expTimes + " iterations");
				//System.out.println("Cumulated Correct Percentage: " +  accuracy);
				//System.out.println("*****************************************************************************");
				if (accuracy > maxCorrectPercentage){
					maxCorrectPercentage = accuracy;
					maxFc1 = fc1;
					maxFc2 = fc2;
					maxFc3 = fc3;
					finalArray1Size = data1.numInstances();
					finalArray2Size = data2.numInstances();
					finalArray3Size = data3.numInstances();
					
				}
				break;
			}
			arffFunctions.generateArff(array1, "docs/intel_arff_header_always.txt", "model1.arff");
			arffFunctions.generateArff(array2, "docs/intel_arff_header_always.txt", "model2.arff");
			arffFunctions.generateArff(array3, "docs/intel_arff_header_always.txt", "model3.arff");
			
			source1 = new DataSource("docs/model1.arff");
			source2 = new DataSource("docs/model2.arff");
			source3 = new DataSource("docs/model3.arff");
			
			data1 = source1.getDataSet();
			data2 = source2.getDataSet();
			data3 = source3.getDataSet();
			
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			data3.setClassIndex(classIndex);
			
			fc1 = wekaFunctions.train(data1, classIndex);
			fc2 = wekaFunctions.train(data2, classIndex);
			fc3 = wekaFunctions.train(data3, classIndex);
		}
	}
	
	public static void main(String[] args) throws Exception {
		String filename= "docs/3ClusterSingleUserValidation200.txt";
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
				threeArraySingleUserValidation threeArrayInstance = new threeArraySingleUserValidation();
				threeArrayInstance.shuffle(restUserArray);
				threeArrayInstance.converge();
			}
//			System.out.println("*****************************************************************************");
//			System.out.println("Final Statistics:\n");
//			System.out.println("maxfc1:\n"+maxFc1+"maxfc2:\n"+maxFc2+"maxfc3:\n"+maxFc3);
//			System.out.println("Array1's size: " +  finalArray1Size + "\t" + "accuracy: " + Array1Accuracy);
//			System.out.println("Array2's size: " +  finalArray2Size + "\t" + "accuracy: " + Array2Accuracy);
//			System.out.println("Array3's size: " +  finalArray3Size + "\t" + "accuracy: " + Array3Accuracy);
//			System.out.println("Max Correct Percentage: " +  maxCorrectPercentage);
//			System.out.println("*****************************************************************************");
			double acury1 = wekaFunctions.eval(maxFc1, singleUserInstances, singleUserInstances);
			double acury2 = wekaFunctions.eval(maxFc2, singleUserInstances, singleUserInstances);
			double acury3 = wekaFunctions.eval(maxFc3, singleUserInstances, singleUserInstances);
			System.out.println("3 clusters' accuracy for user#: " + userID);
			System.out.println(acury1);
			System.out.println(acury2);
			System.out.println(acury3);
			
			fw.write(userID + "," + Math.max(Math.max(acury1, acury2), acury3) + "\n");
		}
		fw.close();
	}

}
