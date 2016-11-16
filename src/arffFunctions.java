import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class arffFunctions {
	
	public static ArrayList<Integer> getUserIDArray(String file) throws Exception {
		ArrayList<Integer> userArray = new ArrayList<Integer>();
		DataSource source = new DataSource(file);
                //DataSource source = new DataSource("docs/.arff");
		Instances data = source.getDataSet();

		for (int i = 0; i < data.numInstances(); i= i + 14) {
			userArray.add((int)data.instance(i).value(0));
		}
		//System.out.println(userArray);
		return userArray;
	}
        
        public static boolean splitUser(String sourcefile) throws Exception {
		DataSource source = new DataSource(sourcefile);
		Instances data = source.getDataSet();
		
		for (int i = 0; i < data.numInstances(); i++) {
			String destFileName = "users/" + (int)data.instance(i).value(0) + ".txt";
			File file = new File(destFileName);
			if(!file.exists()){
				try {
		            if (file.createNewFile()) {  
		                System.out.println("Create " + destFileName + "--Success!");  
		            } else {  
		                System.out.println("Create " + destFileName + "--Fail!");  
		            }  
		        } catch (IOException e) {
		            e.printStackTrace();  
		            System.out.println("Create " + destFileName + "--Fail! " + e.getMessage());
		            return false;
		        }
			}
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(destFileName, true)));
		    out.println(data.instance(i));
		    out.close();
		}
		return true;
	}
	
	// APPENDING AND GENERATING .ARFF FILES
	public static void generateArff(ArrayList<Integer> array, String source, String fileName) throws IOException{
	    //String source = "docs"+File.separator+"intel_arff_header_a.txt";
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
}
