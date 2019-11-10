import numpy as np
import pandas as pd
import glob

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# columns
categorical_columns = ["Proto","SrcAddr","Sport","DstAddr","Dport","State"]
numerical_columns =  ["Dur","TotPkts","SrcBytes","TotBytes"]

# reading all csvs in the folder and creating a dataframe
print("Combining the csv(s) to create a dataframe")

csvs = glob.glob("Input_Directory\*.csv")
dfs = [pd.read_csv(f) for f in csvs]
dataframe = pd.concat(dfs,ignore_index=True)

# Encode Label column to 0 and 1 for 0 being benign and 1 malicious
dataframe["Label"] =  np.where(dataframe["Label"].str.contains("Botnet", case=False, na=False), 1, 0)


# Transformers
# Numeric transformer for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Categorical transformer for categorical columns
categorical_transformer = Pipeline(steps=[(
    'imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Preprocessor
preprocessor = ColumnTransformer(
		transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])

# Pipeline with a SVM classifier
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(gamma="auto"))])

# Spliting data for training and testing
print("Training the SVM Model")
X = dataframe.drop("Label", axis=1)
y = dataframe["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)

# print the score
print("Testing the SVM Model")
print("SVM Model score: %.3f" % clf.score(X_test, y_test))