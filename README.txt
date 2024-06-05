This folder contains the code to reproduce the experiments of the UAI2024 contribution: "Normalizing Flows for Conformal Regression". 

1. Required Python packages
- numpy
- scikit-learn
- pandas
- pytorch

2. Data sets
Download the following data sets and copy them to "datasets". 
- bike_train.csv
- CASP.csv
- communities.data
- communities_attributes.csv
- Concrete_Data.csv
- ENB2012_data.xlsx
- Features_Variant_1.csv


3. Train the RF regressors
Before training and testing the CP models, you need to train the underlying RF regressors by running 

python ./trainRFregression.py

The script will save the (A, X, Y) data sets you need to run the CP experiments in the folder called "yax" and the RF MAEs in "results/synth" and "results/real" 

4. Train and test the NF models
The following command will run 5 training-testing experiments on each of the 10 data sets 

python ./allExp.py

The script will write the results in "results/synth" and "results/real".

5. Print the average scores by running  

python ./print_results.py
