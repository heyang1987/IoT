import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class generateArffFile {
	public static void generateArff(ArrayList<Integer> array, String fileName) throws IOException{
	    String source = "docs\\intel_arff_header.txt";
	    String dest = "docs\\" + fileName;

	    File fin = new File(source);
	    File fout = new File(dest);
	    if (fout.exists()) {
	    	fout.delete();
	    }
	    FileInputStream fis = new FileInputStream(fin);
	    BufferedReader in = new BufferedReader(new InputStreamReader(fis));

	    FileWriter fstream = new FileWriter(dest, true);
	    BufferedWriter out = new BufferedWriter(fstream);

	    String aLine = null;
	    while ((aLine = in.readLine()) != null) {
	        //Process each line and add output to Dest.txt file
	        out.write(aLine);
	        out.newLine();
	    }

	    // do not forget to close the buffer reader
	    in.close();
	    for (int i : array) {
		    fin = new File("users\\"+i+".txt");
		    fis = new FileInputStream(fin);
		    in = new BufferedReader(new InputStreamReader(fis));
		    while ((aLine = in.readLine()) != null) {
		        //Process each line and add output to Dest.txt file
		        out.write(aLine);
		        out.newLine();
		    }
		    in.close();
	    }
	    // close buffer writer
	    out.close();
	}
}
