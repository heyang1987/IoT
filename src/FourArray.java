/**
 * @author Yang
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FourArray {
	
	private static final int ROUNDS = 1;
	
	public static ArrayList<Integer> array1 = new ArrayList<Integer>();
	public static ArrayList<Integer> array2 = new ArrayList<Integer>();
	public static ArrayList<Integer> array3 = new ArrayList<Integer>();
	public static ArrayList<Integer> array4 = new ArrayList<Integer>();
	
	public static int classIndex;
	public static DataSource source;
	private DataSource source1;
	private DataSource source2;
	private DataSource source3;
	private DataSource source4;
	
	public static Instances test;
	private Instances data1;
	private Instances data2;
	private Instances data3;
	private Instances data4;
	
	public static double maxCorrectPercentage = 0;
	private FilteredClassifier fc1 = new FilteredClassifier();
	private FilteredClassifier fc2 = new FilteredClassifier();
	private FilteredClassifier fc3 = new FilteredClassifier();
	private FilteredClassifier fc4 = new FilteredClassifier();
	public FilteredClassifier maxFc1 = new FilteredClassifier();
	public FilteredClassifier maxFc2 = new FilteredClassifier();
	public FilteredClassifier maxFc3 = new FilteredClassifier();
	public FilteredClassifier maxFc4 = new FilteredClassifier();
	public double Array1Accuracy = 0;
	public double Array2Accuracy = 0;
	public double Array3Accuracy = 0;
	public double Array4Accuracy = 0;
	public int finalArray1Size = 0;
	public int finalArray2Size = 0;
	public int finalArray3Size = 0;
	public int finalArray4Size = 0;
	
	public void shuffle(ArrayList<Integer> userArray) throws Exception {
		
		int shuffleTimes = 0;
		int userArraySize = userArray.size();
		source = new DataSource("docs/intel_result6.arff");	
		test = source.getDataSet();
		classIndex = test.numAttributes()-1;
		test.setClassIndex(classIndex);
		
		System.out.println("4 arrays Start Shuffling...");
		shuffleTimes = 0;
		do{
			System.out.println("Shuffling array 1, 2, 3, and 4 Time: " + shuffleTimes);
			Collections.shuffle(userArray);
			array1 = new ArrayList<Integer>(userArray.subList(0, Math.round(userArraySize/4)));
			array2 = new ArrayList<Integer>(userArray.subList(Math.round(userArraySize/4), Math.round(userArraySize/2)));
			array3 = new ArrayList<Integer>(userArray.subList(Math.round(userArraySize/2), Math.round(userArraySize*3/4)));
			array4 = new ArrayList<Integer>(userArray.subList(Math.round(userArraySize*3/4), userArraySize));
			Collections.sort(array1);
			Collections.sort(array2);
			Collections.sort(array3);
			Collections.sort(array4);
			
			user.generateArff(array1, "model1.arff");
			user.generateArff(array2, "model2.arff");
			user.generateArff(array3, "model3.arff");
			user.generateArff(array4, "model4.arff");
			
			source1 = new DataSource("docs/model1.arff");
			source2 = new DataSource("docs/model2.arff");
			source3 = new DataSource("docs/model3.arff");
			source4 = new DataSource("docs/model4.arff");
			
			data1 = source1.getDataSet();
			data2 = source2.getDataSet();
			data3 = source3.getDataSet();
			data4 = source4.getDataSet();
			
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			data3.setClassIndex(classIndex);
			data4.setClassIndex(classIndex);
			
			fc1 = wekaFunctions.train(data1);
			fc2 = wekaFunctions.train(data2);
			fc3 = wekaFunctions.train(data3);
			fc4 = wekaFunctions.train(data4);
			//System.out.println("Array1: " + array1);
			//System.out.println("Array2: " + array2);
			//System.out.println("Array3: " + array3);
			//System.out.println("Array4: " + array4.size());
			//System.out.println("Array1 size: " + array1.size());
			//System.out.println("Array2 size: " + array2.size());
			//System.out.println("Array3 size: " + array3.size());
			//System.out.println("Array4 size: " + array4.size());
			System.out.println("fc1 size: " + fc1.numElements());
			System.out.println("fc2 size: " + fc2.numElements());
			System.out.println("fc3 size: " + fc3.numElements());
			System.out.println("fc4 size: " + fc4.numElements());
			shuffleTimes++;
		}while ( (fc1.numElements() == 1) && (fc2.numElements() == 1) && (fc3.numElements() == 1) && (fc4.numElements() == 1));			
//			System.out.println("Array1: " + array1);
//			System.out.println("Array2: " + array2);
//			System.out.println("Array3: " + array3);
//			System.out.println("Array4: " + array4);
		ArrayList<Integer> combined = new ArrayList<Integer>();
		if (fc1.numElements() != 1) {
			combined.addAll(array2);
			combined.addAll(array3);
			combined.addAll(array4);
			ThreeArray threeArrayInstance = new ThreeArray();
			threeArrayInstance.shuffle(combined);
			fc2 = threeArrayInstance.fc1;
			fc3 = threeArrayInstance.fc2;
			fc4 = threeArrayInstance.fc3;
		}
		else if (fc2.numElements() != 1) {
			combined.addAll(array1);
			combined.addAll(array3);
			combined.addAll(array4);
			ThreeArray threeArrayInstance = new ThreeArray();
			threeArrayInstance.shuffle(combined);
			fc1 = threeArrayInstance.fc1;
			fc3 = threeArrayInstance.fc2;
			fc4 = threeArrayInstance.fc3;
			data1 = threeArrayInstance.data1;
			data3 = threeArrayInstance.data2;
			data4 = threeArrayInstance.data3;
		}
		else if (fc3.numElements() != 1) {
			combined.addAll(array1);
			combined.addAll(array2);
			combined.addAll(array4);
			ThreeArray threeArrayInstance = new ThreeArray();
			threeArrayInstance.shuffle(combined);
			fc1 = threeArrayInstance.fc1;
			fc2 = threeArrayInstance.fc2;
			fc4 = threeArrayInstance.fc3;
			data1 = threeArrayInstance.data1;
			data2 = threeArrayInstance.data2;
			data4 = threeArrayInstance.data3;
		}
		else if (fc4.numElements() != 1) {
			combined.addAll(array1);
			combined.addAll(array2);
			combined.addAll(array3);
			ThreeArray threeArrayInstance = new ThreeArray();
			threeArrayInstance.shuffle(combined);
			fc1 = threeArrayInstance.fc1;
			fc2 = threeArrayInstance.fc2;
			fc3 = threeArrayInstance.fc3;
			data1 = threeArrayInstance.data1;
			data2 = threeArrayInstance.data2;
			data3 = threeArrayInstance.data3;
		}
		System.out.println("4 arrays End Shuffling...");
		
	}
	
	
	public void converge() throws Exception{
		Random randomNum = new Random();
			for (int expTimes = 0; expTimes < 200; expTimes++){
				
			Array1Accuracy = wekaFunctions.evalCrossValidation(fc1, data1);
			Array2Accuracy = wekaFunctions.evalCrossValidation(fc2, data2);
			Array3Accuracy = wekaFunctions.evalCrossValidation(fc3, data3);
			System.out.println("=============================================================================");
			System.out.println("Iteration: " + expTimes);
			//System.out.println("fc1:\n " + fc1);
			//System.out.println("fc2:\n " + fc2);
			//System.out.println("fc3:\n " + fc3);
			System.out.println("fc1 size: " + fc1.numElements() + "\t" +
					"Array1's size: " + data1.numInstances() + "\t" +
					"Array1's accuracy: " + Array1Accuracy);
			System.out.println("fc2 size: " + fc2.numElements() + "\t" +
					"Array2's size: " + data2.numInstances() + "\t" +
					"Array2's accuracy: " + Array2Accuracy);
			System.out.println("fc3 size: " + fc3.numElements() + "\t" +
					"Array3's size: " + data3.numInstances() + "\t" +
					"Array3's accuracy: " + Array3Accuracy);
			
			ArrayList<Integer> array1Backup = new ArrayList<Integer>(array1);  //BACKUPS HAVE BEEN MADE JUST IN CASE WE NEED TO PUT IT BACK IN THE SAME ARRAY
			ArrayList<Integer> array2Backup = new ArrayList<Integer>(array2);
			ArrayList<Integer> array3Backup = new ArrayList<Integer>(array3);
			
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
				System.out.println("*****************************************************************************");
				System.out.println("Arrays converged within " + expTimes + " iterations");
				System.out.println("Cumulated Correct Percentage: " +  accuracy);
				System.out.println("*****************************************************************************");
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
			user.generateArff(array1, "model1.arff");
			user.generateArff(array2, "model2.arff");
			user.generateArff(array3, "model3.arff");
			
			source1 = new DataSource("docs/model1.arff");
			source2 = new DataSource("docs/model2.arff");
			source3 = new DataSource("docs/model3.arff");
			
			data1 = source1.getDataSet();
			data2 = source2.getDataSet();
			data3 = source3.getDataSet();
			
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
			data3.setClassIndex(classIndex);
			
			fc1 = wekaFunctions.train(data1);
			fc2 = wekaFunctions.train(data2);
			fc3 = wekaFunctions.train(data3);
		}
	}
//		System.out.println("*****************************************************************************");
//		System.out.println("Final Statistics:\n");
//		System.out.println("maxfc1:\n"+maxFc1+"maxfc2:\n"+maxFc2+"maxfc3:\n"+maxFc3);
//		System.out.println("Array1's size: " +  finalArray1Size + "\t" + "accuracy: " + Array1Accuracy);
//		System.out.println("Array2's size: " +  finalArray2Size + "\t" + "accuracy: " + Array2Accuracy);
//		System.out.println("Array3's size: " +  finalArray3Size + "\t" + "accuracy: " + Array3Accuracy);
//		System.out.println("Max Correct Percentage: " +  maxCorrectPercentage);
//		System.out.println("*****************************************************************************");
//	}
	
	public static void main(String[] args) throws Exception {
		ArrayList<Integer> userArray = user.getUserIDArray();
		FourArray fourArrayInstance = new FourArray();
		fourArrayInstance.shuffle(userArray);
		System.out.println(fourArrayInstance.fc1);
		System.out.println(fourArrayInstance.array1.size());
		System.out.println(fourArrayInstance.array2.size());
		System.out.println(fourArrayInstance.array3.size());
		System.out.println(fourArrayInstance.array4.size());
	}

	


}
