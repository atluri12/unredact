# Unredactor

In this project the functioning of an unredactor is shown. The dataset taken contains the movie reviews from IMDB. Limited number of reviews are selected for this project to consider the time taken. The program takes the movie reviews as input for training and perform extraction of entities from the training dataset. In this project only the names are redacted and the rest of the entities are ignored. The program is trained on the given dataset and then the program is used on a test data set to perform unredact and predict the entities. The project is compatible with Python 3.7.2.

### unredact.py
The unredact.py file is executed from the *unredact* folder after cloning it into the local system using the following command

`pipenv run python redactor/unredact.py`

The functions **get_entity(text), doextraction(glob_text), get_entity_result(text), doextraction_result(glob_text)** are called with the respective arguments for the execution. The detailed Functioning of the functions mentioned is as follows:
* #### get_entity(text)
    The function *get_entity* takes one argument which is the *text* given by the doextraction method. The get_entity method is called by the doextraction method for every file in the glob directory of the given directory. The get_entity method performs the sentance tokenization using nltk package and then get the entities which have the label 'PERSON' which indicates the names of people. The output from the get_entity method is the vector which contains the features such as *word length, number of spaces, number of words, length of the first word and length of the second word*. These features are extracted for every entity in the get_entity method.
* #### doextraction(glob_text)
    The function *doextraction* takes one argument which is the *glob_text* given by as input which contains the directory of the dataset. This method takes every files present in given directory and  call the get_entity method for the text present in every file. This method returns the vector containing the features of the entities and the list of entities.
* #### get_entity_result(text)
    The function *get_entity_result* takes one argument which is the *text* given by the doextraction_result method. The get_entity_result method is called by the  doextraction_result method for every file in the glob directory of the given directory. The get_entity_result method performs the extraction of the redacted elements and extracts the features which are *word length, number of spaces, number of words, length of the first word and length of the second word*.
* #### doextraction_result(glob_text)
    The function *doextraction_result* takes one argument which is the *glob_text* given by as input which contains the directory of the dataset. This method redacts the entities present in every file using the full block character('\u2588'). The redacted data is given as input to call the get_entity_result method to get the features and the entites.

The entities and vectors extracted from the doextraction method for the training dataset are given as input for the DictVectorizer to convert them into DictVectors. This DictVectors produce the training vectors. A GaussianNB (Gaussian Naive Bayes) classifier is used to perform the predictions.
```python
classifier = GaussianNB()
classifier.fit(train, train_entities)
```
The test   data vectors are extracted using the doextraction_result method. This vector is converted using the DictVectorizer. The classifier is fit using the training data and the testing data is used to predict the data. The predicted names are the unredacted names output.
```python
test_dictvec = DictVectorizer()
test = test_dictvec.fit_transform(test_vector).toarray()
result = classifier.predict(test)
```

### setup.py and setup.cfg
The setup.py file is required for finding the packages within the project during the execution. It finds the packages automatically. 

The setup.cfg file is required for running the pytest command to perform tests on the program.

### Pipfile and Pipfile.lock
A virtual environment is created for the execution of this project. This environment is created using *pipenv*c command as following

`pipenv install --python 3.7.2`

`pipenv install sklearn`

`pipenv install nltk`

`pipenv install pytest`

### test_unredact.py
The test_unredact.py file contains the test cases designed to test the functioning of the project2. The test_unredact.py file when executed runs every test case with the project2 and returns the output of failed and passed test cases. The test_unredact.py file contains five test cases specified one each for every function in the unredact.py file. The test cases are as follows:
* #### test_get_entity()
    The *test_get_entity* test case is used to test the *get_entity* function of the unredact.py file. This function executes the get_entity function and returns the extracted entities which are names and the features related to each name. The returned features vector is checked if it contains 5 features.

    `assert len(vector[0]) == 5`

* #### test_doextraction()
    The *test_doextraction* function is used to test the *doextraction* funtion of the project2.py file. This function executes the function and retrieves the features vector and the names of the entities. The assert statement in the function checks if the returned vector is of length 5, i.e. it contains only 5 features.

* #### test_get_entity_result()
    The *test_get_entity_result* function is used to test the *get_entity_result* function of the unredact.py file. This function executes the get_entity_result function with the parameter being the text data which is redacted for names. The assert statement is used to check if the entities returned after extraction from the redacted data are in the form of '\u2588+\s?\u2588*\s?\u2588*'.
    ```python
    reg = re.compile('\u2588+\s?\u2588*\s?\u2588*')
        for a in name:
            assert re.match(reg, a)
    ```

* #### test_doextraction_result()
    The *test_doextraction_result* function is used to test the *doextraction_result* function of the unredact.py file. This function executes the function get_entity_result to get the features of the redacted entities in the data. The assert statement checks if the returned vector contains all the features of length 5.
