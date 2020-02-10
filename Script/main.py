import time, re, ast, numpy as np, pandas as p, cPickle as pickle
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive
from nltk import clean_html, SnowballStemmer, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

print ('Starting at '+time.strftime("%H:%M:%S", time.localtime())+'  ...')

## preprocessing 
# We have used two methods to preprocess boilerplate and url of example, namely Stemming and Lemmatization
def stemming(words_l, type="PorterStemmer", lang="english", encoding="utf8"):
	supported_stemmers = ["PorterStemmer","WordNetLemmatizer"]
	if type is False or type not in supported_stemmers:
		return words_l
	else:
		temp_list = []
		if type == "PorterStemmer": 
			#calling PorterStemmer
			stemmer = PorterStemmer()
			for word in words_l:
				temp_list.append(stemmer.stem(word).encode(encoding))
		if type == "WordNetLemmatizer":
			#calling WordNetLemmatizer
			wnl = WordNetLemmatizer()
			for word in words_l:
				temp_list.append(wnl.lemmatize(word).encode(encoding))
		return temp_list

# String and tokenize
def preprocess_boilerplate(str, stemmer_type="WordNetLemmatizer", lang="english", return_as_str=True, 
						do_remove_stopwords=True):
	temp_list = []
	words = []
	
	# Tokenizing
	# Tokenizers divide strings into lists of substrings.  For example, tokenizers can be used to find the words and punctuation in a string.
	sentences=[word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) for t in sent_tokenize(str.replace("'", ""))]
	
	for sentence in sentences:
		# Remove stopwords
		if do_remove_stopwords:
			#Stopwords usually have little lexical content, and their presence in a text fails to distinguish it from other texts.
			#examples: 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after'
			words = [w for w in sentence if w.lower() not in stopwords.words('english')]
		else:
			words = sentence
		# Stemming
		words = stemming(words, stemmer_type)
		
		# Convert to string
		if return_as_str:
			temp_list.append(" ".join(words))
		else:
			temp_list.append(words)
	if return_as_str:
		return " ".join(temp_list)
	else:
		return temp_list

# url cleaner (used to remove stop words, digits, etc), and then stemming function is called for further processing
def url_cleaner(url, stemmer_type="WordNetLemmatizer"):
	strip_list=['http', 'https', 'www', 'com', 'net', 'org', 'm', 'html', 'htm']
	url_list=[x for x in word_tokenize(" ".join(re.findall(r'\w+', url, flags = re.UNICODE | re.LOCALE)).lower()) if x not in strip_list and not x.isdigit() and x not in stopwords.words('english')]
	return " ".join(stemming(url_list, stemmer_type))

# extracting content of the page
def extract_content(str, stemmer_type="WordNetLemmatizer"):
	# Adjusting 'null' and extracting json

	#The "str" provided as function argument is filtered to contain only following Python literal structures: strings, numbers, tuples, lists, dicts, booleans, and None
	# Done for safely evaluating strings containing Python values from untrusted sources without the need to parse the values oneself. 
	try:
		json=ast.literal_eval(str)
	except ValueError:
		json=ast.literal_eval(str.replace('null', '"null"'))

	if ('body' in json and 'title' in json):
		return (preprocess_boilerplate(json['title'], stemmer_type), preprocess_boilerplate(json['body'], stemmer_type))
	elif ('body' in json):
		return ("", preprocess_boilerplate(json['body'], stemmer_type))
	elif ('title' in json):
		return (preprocess_boilerplate(json['title'], stemmer_type), "")
	else:
		return ("", "")

# creating TF-IDF matrix
def create_TF_IDF(train_data, test_data, model):
	# combine train and test data containing words
	cummulative_data = train_data + test_data

	# Learn the idf vector, to apply in transformation below
	print ("\nLearning IDF Vector...		started at "+time.strftime("%H:%M:%S", time.localtime()))
	model.fit(cummulative_data)

	# Transform matrix
	print ("\nTransforming data to create TF-IDF matrix...		started at "+time.strftime("%H:%M:%S", time.localtime()))
	# creating tf-idf matrix of dimentions (n_samples) x (n_features)
	cummulative_data = model.transform(cummulative_data)

	#seperating into train and test features, and returning 
	return (cummulative_data[:len(train_data)], cummulative_data[len(train_data):])


# This function returns highly frequent words in both categories(ephimeral/evergreen)
# the returned words are of not much importance in model being learned and can act as outliers, so are removed from dataset later
def get_high_frequence_words(words_list, yvalues):
	# y value 1 indicates evergrees website
	# y value 0 indicates non-evergreen(ephimeral) website
	all_words_present = []
	term_frequency_evergreen_dict = {}
	term_frequency_ephimeral_dict = {}
	word_count_evergreen = 0
	word_count_ephimeral = 0
	for doc in range(len(yvalues)):
		if(yvalues[doc] == 1):
			#evergreen website
			for word in words_list[doc].split():
				if(word not in all_words_present):
					all_words_present.append(word)
				if (word in term_frequency_evergreen_dict):
					term_frequency_evergreen_dict[word] += 1
				else:
					term_frequency_evergreen_dict[word] = 1
				word_count_evergreen += 1 #this is used for normalization later
		elif(yvalues[doc] == 0):
			#non-evergreen(ephimeral) website
			for word in words_list[doc].split():
				if(word not in all_words_present):
					all_words_present.append(word)
				if (word in term_frequency_ephimeral_dict):
					term_frequency_ephimeral_dict[word] += 1
				else:
					term_frequency_ephimeral_dict[word] = 1
				word_count_ephimeral += 1 #this is used for normalization later
		else:
			raise Exception("Y value is not 1 or 0 in train data")

	#normalizing term frequencies to convert them into fraction of frequency of occurance of words
	for i in term_frequency_evergreen_dict: #evergreen website
		term_frequency_evergreen_dict[i] /= (1.0*word_count_evergreen)
		# print term_frequency_evergreen_dict[i]

	for i in term_frequency_ephimeral_dict: #non-evergreen(ephimeral) website
		term_frequency_ephimeral_dict[i] /= (1.0*word_count_ephimeral)
		# print term_frequency_ephimeral_dict[i]
	
	#createng a list of words which have high frequency in both category's(evergreen/ephimeral) data 
	high_frequency_words = []
	frequency_threshold = 0.0001 # we came up with these values by printing above constructed dictionaries and looking at percentage of words have higher frequencies
	for word in all_words_present:
		if( (word in term_frequency_evergreen_dict) and (word in term_frequency_ephimeral_dict) ):
			if( (term_frequency_evergreen_dict[word] > frequency_threshold) and (term_frequency_ephimeral_dict[word] > frequency_threshold) ):
				high_frequency_words.append(word)
	return high_frequency_words

#fitting based on model: Regularized Logistic Regression, SVM, Naive Bayesian
def fit_train_and_test_data(train_data, test_data, y_train_data, model):
	if (model == "logit"):
		#create parameters, to implement Regularized Logistic Regression ( http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html )
		#Parameters Explained:
		#	penalty is set to "l2", to use l2 norm penalty
		#	dual is set to True, to use dual formulation in Logistic Regression
		#	tol is set to 0.0001, this is tolerance for stopping criteria
		#	C is set to 1.0, this is inverse of regularization strength. Smallar values specify stronger regularization
		#	fit_intercept is set to True, to add bias to decision function
		#	intercept_scaling is set to 1.0
		#	class_weight is set to None, to set weights of classes as 1
		#	random_state is set to None, this is seed of the pseudo random number generator to use when shuffling the data
		logistic_regression_parameters = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
		
		# transforming data, fitting logistic regression model
		print ( "\nApplying Logistic regression on train data...		started at "+time.strftime("%H:%M:%S", time.localtime()) )
		logistic_regression_parameters.fit(train_data,y_train_data)

		#transforming features in both train and test data to reduce it to have only most important features useful in learning model
		train_data = logistic_regression_parameters.transform(train_data)
		test_data = logistic_regression_parameters.transform(test_data)

		#Evaluating score on Train Data by taking mean of scores from 10 fold cross validation
		#Parameters Explained:
		#	we have used regularized Logistic Regression to fit the data, whose parameters are defined above
		#	data used to fit train_data
		#	target variable to predict is y_train_data
		#	cv is set to 10, to have 10 fold cross validation
		#	scoring is set to roc_auc(Receiver Operating Characteristic - Area Under Curve)
		print "\n10 Fold CV Score after transforming and doing Regularized Logistic Regularized on train data: ", np.mean(cross_validation.cross_val_score(logistic_regression_parameters, train_data, y_train_data, cv=10, scoring='roc_auc'))

		# Run logistic regression
		print ("\nTraining on full data and constructing output files...		started at "+time.strftime("%H:%M:%S", time.localtime()))
		logistic_regression_parameters.fit(train_data, y_train_data)

		#predicting probabilities of class to which sample belongs to...
		predicted_train = logistic_regression_parameters.predict_proba(train_data)[:,1]
		predicted_test = logistic_regression_parameters.predict_proba(test_data)[:,1]
		return (predicted_train, predicted_test)

	elif (model == "svm"):
		#create parameters, to implement Support Vector Machine ( http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html )
		#Parameters Explained:
		#	probability is set to True, to enable probability estimates
		#	max_iter is set to -1, to indicate no limit on number of iterations
		#	tol is set to 0.0001, this is tolerance for stopping criteria
		#	C is set to 1.0, Penalty parameter C of the error term
		#	class_weight is set to None, to set weights of classes as 1
		#	random_state is set to None, this is seed of the pseudo random number generator to use when shuffling the data
		svm_parameters = svm.SVC(probability = True, max_iter = 1000, tol=0.0001, C=1.0, class_weight=None, random_state=None)
		# transforming data, fitting logistic regression model
		print ( "\nApplying Support Vector Machine Learning on train data...		started at "+time.strftime("%H:%M:%S", time.localtime()) )
		svm_parameters.fit(train_data,y_train_data)

		#Evaluating score on Train Data by taking mean of scores from 10 fold cross validation
		#Parameters Explained:
		#	we have used support vector machine to fit the data, whose parameters are defined above
		#	data used to fit train_data
		#	target variable to predict is y_train_data
		#	cv is set to 10, to have 10 fold cross validation
		#	scoring is set to roc_auc(Receiver Operating Characteristic - Area Under Curve)
		print "\n10 Fold CV Score after transforming and doing Support Vector Machine Learning on train data: ", np.mean(cross_validation.cross_val_score(svm_parameters, train_data, y_train_data, cv=10, scoring='roc_auc'))

		# Run SVM
		print ("\nTraining on full data and constructing output files...		started at "+time.strftime("%H:%M:%S", time.localtime()))
		svm_parameters.fit(train_data,y_train_data)

		#predicting probabilities of class to which sample belongs to...
		predicted_train = svm_parameters.predict_proba(train_data)[:,1]
		predicted_test = svm_parameters.predict_proba(test_data)[:,1]
		return (predicted_train, predicted_test)

	elif (model == "naive"):
		#create parameters, to implement Multinomial Naive's bayesian ( http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB )
		#parameters Explained: 
		#       Alpha is set to 1 to allow smoothing, 0 for no smoothing and it can take any float values in between 0 and 1
		#       Fit prior is set to true, to learn class prior probabilities
		#       Array for class prior probabilities. If this is not specifies ( None ), then they are adjusted according to the data.
		naive_parameters = naive.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

		# transforming data, fitting logistic regression model
		print ( "Applying Multinomail Naive Bayesian on train data...               started at "+time.strftime("%H:%M:%S", time.localtime()))               
		naive_parameters.fit(train_data,y_train_data)


		#Evaluating score on Train Data by taking mean of scores from 10 fold cross validation
		#Parameters Explained:
		#       we have used Multinomial Naive Bayesian to fit the data, whose parameters are defined above
		#       data used to fit train_data
		#       target variable to predict is y_train_data
		#       cv is set to 10, to have 10 fold cross validation
		#       scoring is set to roc_auc(Receiver Operating Characteristic - Area Under Curve)
		print ("10 Fold CV Score after transforming and doing MultinomialNB on train data: ", np.mean(cross_validation.cross_val_score(naive_parameters, train_data, y_train_data, cv=10, scoring='roc_auc')))

		#Run Multinomial Naive Bayesian
		print ("Training on full data and constructing output files...          started at "+time.strftime("%H:%M:%S", time.localtime()))
		naive_parameters.fit(train_data,y_train_data)

		#predicting probabilities of class to which sample belongs to...
		predicted_train = naive_parameters.predict_proba(train_data)[:,1]
		predicted_test = naive_parameters.predict_proba(test_data)[:,1]
		return (predicted_train, predicted_test)

	else:
		raise Exception("Undefined model specified to use in classification")


if __name__ == "__main__":
	#loading train and test data
	print ("\nLoading input...\n")
	x_traindata = list(np.array(p.read_table('../data/train.tsv'))[:,2])
	x_testdata = list(np.array(p.read_table('../data/test.tsv'))[:,2])
	y_train = np.array(p.read_table('../data/train.tsv'))[:,-1] #last rwo consists of output values in train data
	y_train=y_train.astype(int)

	# loading url from both train and test data
	x_url_train = list(np.array(p.read_table('../data/train.tsv'))[:,0])
	x_url_test = list(np.array(p.read_table('../data/test.tsv'))[:,0])


	print ("\nPreprocessing boilerplate and url...		started at "+time.strftime("%H:%M:%S", time.localtime()))

	x_train_title_list = []
	x_train_body_list = []
	x_test_title_list = []
	x_test_body_list = []

	#using WordNetLemmatizer for stemming boilerplate content 
	for temp_data in x_traindata:
			temp_title, temp_body=extract_content(temp_data, "WordNetLemmatizer")
			x_train_title_list.append(temp_title)
			x_train_body_list.append(temp_body)
	for temp_data in x_testdata:
			temp_title, temp_body=extract_content(temp_data, "WordNetLemmatizer")
			x_test_title_list.append(temp_title)
			x_test_body_list.append(temp_body)
			
	#here pickle module is used to print data obtained after stemming boilerplate data (https://docs.python.org/2/library/pickle.html) in serialized manner 
	pickle.dump(x_train_title_list, open('preprocessed_train_title.p', 'wb'))
	pickle.dump(x_train_body_list, open('preprocessed_train_body.p', 'wb'))
	pickle.dump(x_test_title_list, open('preprocessed_test_title.p', 'wb'))
	pickle.dump(x_test_body_list, open('preprocessed_test_body.p', 'wb'))

	x_train_url_list = []
	x_test_url_list = []
	
	#using WordNetLemmatizer for stemming url 
	for url in x_url_train:
	        x_train_url_list.append(url_cleaner(url, "WordNetLemmatizer"))
	for url in x_url_test:
	        x_test_url_list.append(url_cleaner(url, "WordNetLemmatizer"))

	#here pickle module is used to print data obtained after stemming url (https://docs.python.org/2/library/pickle.html) in serialized manner 
	pickle.dump(x_train_url_list, open('preprocessed_train_url.p', 'wb'))
	pickle.dump(x_test_url_list, open('preprocessed_test_url.p', 'wb'))

	#extracting features

	#create TF-IDF matrix parameters (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
	#Parameters Explained:
	#	min_df is set to 15, to ignore terms that have document frequency strictly lower then 15, When building the vocabulary.
	#	analyzer is chosed to 	be word, to have feature made of words
	#	ngram_range is set to (1,2), which specifies lower and upper boundary (to be 1 and 2 respectively) of the range of n-values for different n-grams to be extracted
	# 	use_idf is set to True, to enable inverse-document-frequency reweighting
	#	smooth_idf is set to True, to smooth idf weights by adding constant '1' is added to the numerator and denominator of the idf as if an extra document was seen containing every term in the collection exactly once, 
	#			which prevents zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.

	#	sublinear_tf is set to True, to apply sublinear tf scaling, ( i.e. replace term frequency with 1 + log(tf) )

	tf_idf_parameters = TfidfVectorizer(min_df=15, max_features=None, strip_accents='unicode', analyzer='word', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)


	# putting together url, title and body of every input data point to use in feature extraction later
	# these cummulative lists are used to ci=onstruct TF-IDF matrix and go further with classification
	x_train_cummulative = [x_train_url_list[temp] + ' ' + x_train_title_list[temp] + ' ' + x_train_body_list[temp] for temp in range(len(x_train_body_list))]
	x_test_cummulative = [x_test_url_list[temp] + ' ' + x_test_title_list[temp] + ' ' + x_test_body_list[temp] for temp in range(len(x_test_body_list))]

	# constructing new set of highly frequent words for both categories(ephimeral/evergreen)
	# These set of words have high frequency in the dataset and are of not much importance, as they could act as outliers in the model being learned

	high_frequency_words_to_ignore=set(get_high_frequence_words(x_train_cummulative, y_train))
	print "\nNumber of high frequency words in data: ", len(high_frequency_words_to_ignore)

	# removing high frequency words
	x_train_cummulative=[' '.join(word for word in temp.split() if word not in high_frequency_words_to_ignore) for temp in x_train_cummulative]
	x_test_cummulative=[' '.join(word for word in temp.split() if word not in high_frequency_words_to_ignore) for temp in x_test_cummulative]

	#here pickle module is used to print data obtained after stemming url (https://docs.python.org/2/library/pickle.html) in serialized manner 
	pickle.dump(high_frequency_words_to_ignore, open('high_frequency_words.p', 'wb'))
	pickle.dump(x_train_cummulative, open('modified_train_data_with_removed_high_frequncy_words.p', 'wb'))
	pickle.dump(x_test_cummulative, open('modified_test_data_with_removed_high_frequncy_words.p', 'wb'))

	# creating tf-idf matrix from train and test data available from above
	tfidf_x_train_cummulative, tfidf_x_test_cummulative=create_TF_IDF(x_train_cummulative, x_test_cummulative, tf_idf_parameters)

	#calling fit_train_and_test_data() function to fit the train data and predict probabilities of various classes
	predicted_train, predicted_test = fit_train_and_test_data(tfidf_x_train_cummulative, tfidf_x_test_cummulative, y_train, "logit")

	# Write out into files
	train_file_params = p.read_csv('../data/train.tsv', sep="\t", na_values=['?'], index_col=1)
	predicted_file = p.DataFrame(predicted_train, index=train_file_params.index, columns=['label'])
	predicted_file.to_csv('prediction_train_data.csv')

	test_file_params = p.read_csv('../data/test.tsv', sep="\t", na_values=['?'], index_col=1)
	test_file = p.DataFrame(predicted_test, index=test_file_params.index, columns=['label'])
	test_file.to_csv('prediction_test_data.csv')

	print ('Completed at '+time.strftime("%H:%M:%S", time.localtime())+'  ...')