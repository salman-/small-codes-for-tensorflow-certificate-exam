import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from CustomColumnTransformer import CustomColumnTransformer

NUMBER_OF_ROWS = 10
dt = pd.DataFrame({
    "Sex": np.random.choice(['male', 'female'], NUMBER_OF_ROWS),
    "Age": np.random.randint(12, 88, NUMBER_OF_ROWS),
    "Salary": np.random.randint(1000, 100_000, NUMBER_OF_ROWS),
    "Height": np.random.randint(120, 200, NUMBER_OF_ROWS),
    "Smoker": np.random.choice(['Yes', 'No'], NUMBER_OF_ROWS),
    "Weight": np.random.randint(45, 120, NUMBER_OF_ROWS),
    "Married": np.random.choice([True, False], NUMBER_OF_ROWS),
    "Education": np.random.choice(['BSc', 'Msc', 'PhD'], NUMBER_OF_ROWS),
    "Cancer_Status": np.random.choice([0, 1], NUMBER_OF_ROWS)})

# print(dt)

numerical_features = ['Age', 'Salary', 'Height', 'Weight']
categorical_features = ['Sex', 'Smoker', 'Education']
binary_features = ['Married']

numerical_pipeline = Pipeline(steps=[("numerical_features", MinMaxScaler())])
categorical_pipeline = Pipeline(steps=[("categorical_features", OneHotEncoder())])
boolean_pipeline = Pipeline(
    steps=[('boolean_features', CustomColumnTransformer(CustomColumnTransformer.convert_boolean_to_int))])

print(dt)

ct = ColumnTransformer(transformers=[
    ('numerical_pipeline', numerical_pipeline, numerical_features),
    ('categorical_pipeline', categorical_pipeline, categorical_features),
    ('boolean_pipeline',boolean_pipeline,binary_features)], remainder='drop')

dt = ct.fit_transform(dt)

print(ct.get_feature_names_out())
print('========================= Data =========================')
print(dt[0])
