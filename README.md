# heart disease prediction (mini-project)

Contributer - Amogh Shalabh Srivastava

## About

The Following project takes in the information of the person (example age , chol value, etc) and then based on the details entered by the user the project will predict if the person have a Heart Disease or not . This type of softwares can be useful in the future as since the heart disease is a rising threat these days to such projects can be used to alert the person earlier on that he/she may be suffering from a heart disease at a earlier stage and thus help detect and avoid life threatning diseases at earlier stage.

## Approach

The project takes in the data of several people as 'heart.csv' which is then used to train out RandomForestClassifier model and then this model is used to work on no the values given by the user and with the hep of the training given to it the model predicts that if the person have heart disease or not.

## Files Description

1- heart.csv -> the CSV file we use as the train data for the model that contains a no. of entries of both person who have disease and of person with no disease.

2- Model_Creation.ipynb -> Jupiter Notebook where we use the dataset and train the model according to it and then save the models in folder named 'pickle'.

3- pickle (folder) -> This folder contains our main model from RandomForestClassifier 'model.pkl' and all other are the model from OneHotEncoder.

4- App.ipynb -> Jupiter Noytebook where we take the inputs from the user and use the models inside the pickle folder to precess data and then use the 'model.pkl' from same folder to get the output on the given input (accuracy of prediction 88.5%)

### Submitted to Digipodium
website- http://www.digipodium.com/
github- http://github.com/digipodium

