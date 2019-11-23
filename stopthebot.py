import numpy as np
import pandas as pd
import glob

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

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

# Classfiers
clf_svm = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(gamma="auto"))])
clf_gbt = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(max_depth=3))])


# Spliting data for training and testing
X = dataframe.drop("Label", axis=1)
y = dataframe["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf_svm.fit(X_train, y_train)
clf_gbt.fit(X_train, y_train)

# print the scores
print("Testing the SVM Model")
print("SVM Model score: %.3f" % clf_svm.score(X_test, y_test))
print("Testing the GBT Model")
print("GBT Model score: %.3f" % clf_gbt.score(X_test, y_test))