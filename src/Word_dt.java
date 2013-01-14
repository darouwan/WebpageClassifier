import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Word_dt {
	private ArrayList<String> enURLList = new ArrayList<String>();
	private ArrayList<String> frURLList = new ArrayList<String>();

	private ArrayList<String> enFeature = new ArrayList<String>();
	private ArrayList<String> frFeature = new ArrayList<String>();
	private ArrayList<String> feature = new ArrayList<String>();

	private ArrayList<String>[] allURLPartEN;
	private ArrayList<String>[] allURLPartFR;
	
	private ArrayList<String>[] allURLTrigramEN;
	private ArrayList<String>[] allURLTrigramFR;

	private ArrayList<int[]> en;
	private ArrayList<int[]> fr;

	private Instances instances;
	private static Filter filter = new StringToWordVector();
	private static Classifier classifier = new NaiveBayes();
	private static Classifier classifier2 = new J48();
	
	private ArrayList<String> testURL = new ArrayList<String>();
	
	
	public void readInTestFile(String fileName){
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(fileName));
			String sline = null;
			while ((sline = reader.readLine()) != null) {
				if (sline.length() > 0) {
					testURL.add(sline);
				}

			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public Instance createTestInstance(Instances data, int[] line)
			throws Exception {
		Instance testInstance = createInstanceByText(data, line);
		filter.input(testInstance);
		return filter.output();
	}

	public void testWord(String url,Classifier c) throws Exception {

		ArrayList<String> alURL = this.getURLparts(url);
		int[] iURL = this.getFeatureVector(alURL);

		Instance testInstance = createTestInstance(
				instances.stringFreeStructure(), iURL);
		
			
		
		double predicted = c.classifyInstance(testInstance);
		//System.out.println(predicted);
		String category = instances.classAttribute().value((int) predicted);
		System.out.println(category);
	}
	
	public void testTrigram(String url,Classifier c) throws Exception {

		ArrayList<String> alURL = this.getURLTrigram(url);
		int[] iURL = this.getFeatureVector(alURL);

		Instance testInstance = createTestInstance(
				instances.stringFreeStructure(), iURL);

		double predicted = c.classifyInstance(testInstance);
		//System.out.println(predicted);
		String category = instances.classAttribute().value((int) predicted);
		System.out.println(category);
	}
	

	public Instance createInstanceByText(Instances data, int[] line) {
		Instance instance = new Instance(feature.size() + 1);
		for (int j = 0; j < line.length; j++) {
			Attribute textAtt = instances.attribute(feature.get(j));
			//int index = textAtt.addStringValue(line[j] + "");
			instance.setValue(textAtt, line[j]);
		}

		instance.setDataset(instances);
		return instance;
	}

	public void buildClassifier() throws Exception {
		FastVector categories = new FastVector();
		categories.addElement("en");
		categories.addElement("fr");

		FastVector attributes = new FastVector();

		for (int i = 0; i < feature.size(); i++) {
			attributes.addElement(new Attribute(feature.get(i)));
		}

		attributes.addElement(new Attribute("category", categories));
		instances = new Instances("Weka", attributes, enURLList.size()
				+ frURLList.size());
		instances.setClassIndex(instances.numAttributes() - 1);

		for (int i = 0; i < en.size(); i++) {
			int[] line = en.get(i);
			Instance instance = createInstanceByText(instances, line);
			instance.setClassValue("en");

			instances.add(instance);
		}

		for (int i = 0; i < fr.size(); i++) {
			int[] line = fr.get(i);
			Instance instance = createInstanceByText(instances, line);
			instance.setClassValue("fr");

			instances.add(instance);
		}

		filter.setInputFormat(instances);
		Instances filteredInstances = Filter.useFilter(instances, filter);
		classifier.buildClassifier(filteredInstances);

	}
	
	public void buildClassifierJ48() throws Exception {
		FastVector categories = new FastVector();
		categories.addElement("en");
		categories.addElement("fr");

		FastVector attributes = new FastVector();

		for (int i = 0; i < feature.size(); i++) {
			attributes.addElement(new Attribute(feature.get(i)));
		}

		attributes.addElement(new Attribute("category", categories));
		instances = new Instances("Weka", attributes, enURLList.size()
				+ frURLList.size());
		instances.setClassIndex(instances.numAttributes() - 1);

		for (int i = 0; i < en.size(); i++) {
			int[] line = en.get(i);
			Instance instance = createInstanceByText(instances, line);
			instance.setClassValue("en");

			instances.add(instance);
		}

		for (int i = 0; i < fr.size(); i++) {
			int[] line = fr.get(i);
			Instance instance = createInstanceByText(instances, line);
			instance.setClassValue("fr");

			instances.add(instance);
		}

		filter.setInputFormat(instances);
		Instances filteredInstances = Filter.useFilter(instances, filter);
		classifier2.buildClassifier(filteredInstances);

	}

	public int[] getFeatureVector(ArrayList<String> al) {
		int[] f = new int[feature.size()];
		for (int i = 0; i < feature.size(); i++) {
			f[i] = 0;
		}

		for (int i = 0; i < al.size(); i++) {
			String s = al.get(i);
			for (int j = 0; j < feature.size(); j++) {
				if (s.equals(feature.get(j))) {
					f[j]++;
				}
			}
		}
		for (int i = 0; i < f.length; i++) {
			// System.out.print(f[i] + ",");
		}
		// System.out.println();
		return f;
	}

	public void getAllFeatureForWordVector() {
		en = new ArrayList<int[]>();
		fr = new ArrayList<int[]>();
		for (int i = 0; i < allURLPartEN.length; i++) {
			en.add(getFeatureVector(allURLPartEN[i]));
		}
		for (int i = 0; i < allURLPartFR.length; i++) {
			fr.add(getFeatureVector(allURLPartFR[i]));
		}
	}
	
	public void getAllFeatureForTrigram(){
		en = new ArrayList<int[]>();
		fr = new ArrayList<int[]>();
		for (int i = 0; i < allURLTrigramEN.length; i++) {
			en.add(getFeatureVector(allURLTrigramEN[i]));
		}
		for (int i = 0; i < allURLTrigramFR.length; i++) {
			fr.add(getFeatureVector(allURLTrigramFR[i]));
		}
	}

	public ArrayList<String> getURLparts(String url)
			throws UnsupportedEncodingException {
		ArrayList<String> al = new ArrayList<String>();
		String decodedURL = URLDecoder.decode(url, "utf-8");
		String[] part = decodedURL
				.split("\\p{Punct}|[0-9]|www|index|html|htm|http|https");
		for (int i = 0; i < part.length; i++) {
			if (part[i].length() >= 2) {
				al.add(part[i]);
			}
		}
		return al;
	}
	
	
	public ArrayList<String> getURLTrigram(String url) throws UnsupportedEncodingException{
		ArrayList<String> al = new ArrayList<String>();
		String decodedURL = URLDecoder.decode(url, "utf-8");
		String[] part = decodedURL
				.split("\\p{Punct}|[0-9]|www|index|html|htm|http|https");
		for (int i = 0; i < part.length; i++) {
			if (part[i].length() >= 2) {
				String s = "_"+part[i]+"_";
				for(int j = 0;j<part[i].length();j++){
					String t = s.substring(j, j+3);
					al.add(t);
					
				}
					
				
			}
		}
		return al;
	}

	public void getFeaturesForWord() throws UnsupportedEncodingException {
		for (int i = 0; i < enURLList.size(); i++) {
			ArrayList<String> al = getURLparts(enURLList.get(i));
			allURLPartEN[i] = al;
			for (int j = 0; j < al.size(); j++) {
				if (!feature.contains(al.get(j))) {
					feature.add(al.get(j));
					// System.out.println(al.get(j));
				}
			}
		}

		for (int i = 0; i < frURLList.size(); i++) {
			ArrayList<String> al = getURLparts(frURLList.get(i));
			allURLPartFR[i] = al;
			for (int j = 0; j < al.size(); j++) {
				if (!feature.contains(al.get(j))) {
					feature.add(al.get(j));
					// System.out.println(al.get(j));
				}
			}
		}

	}
	
	public void getFeaturesForTrigram() throws UnsupportedEncodingException{
		for (int i = 0; i < enURLList.size(); i++) {
			ArrayList<String> al = getURLTrigram(enURLList.get(i));
			allURLTrigramEN[i] = al;
			for (int j = 0; j < al.size(); j++) {
				if (!feature.contains(al.get(j))) {
					feature.add(al.get(j));
					// System.out.println(al.get(j));
				}
			}
		}

		for (int i = 0; i < frURLList.size(); i++) {
			ArrayList<String> al = getURLTrigram(frURLList.get(i));
			allURLTrigramFR[i] = al;
			for (int j = 0; j < al.size(); j++) {
				if (!feature.contains(al.get(j))) {
					feature.add(al.get(j));
					// System.out.println(al.get(j));
				}
			}
		}
		
		System.out.println("feature length = "+feature.size());
	}
	

	public void readinFile(String file) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String sline = null;
			while ((sline = reader.readLine()) != null) {
				if (sline.length() > 0) {
					String[] temp = sline.split(" ");
					if (temp[1].equals("en")) {
						enURLList.add(temp[0]);
					}
					if (temp[1].equals("fr")) {
						frURLList.add(temp[0]);
					}
				}

			}
			allURLPartEN = new ArrayList[enURLList.size()];
			allURLPartFR = new ArrayList[frURLList.size()];
			
			allURLTrigramEN = new ArrayList[enURLList.size()];
			allURLTrigramFR = new ArrayList[frURLList.size()];

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		//String url = "http://www.linternaute.com/dictionnaire/fr/definition/action/";
		Word_dt wd = new Word_dt();
		wd.readinFile("trainingFile");
		
		wd.getFeaturesForWord();
		//wf.getFeaturesForTrigram();
		
		wd.getAllFeatureForWordVector();
		//wf.getAllFeatureForTrigram();
		
		//wf.buildClassifier();
		wd.buildClassifierJ48();
		wd.readInTestFile("testingFile");
		
		for(int i=0;i<wd.testURL.size();i++){
			//System.out.print(i +":");
			wd.testWord(wd.testURL.get(i),classifier2);
			//wd.testTrigram(wd.testURL.get(i),classifier2);
		}
		
		
		
		
	}

}
