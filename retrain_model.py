import hopsworks
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import os
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

data = fs.get_feature_group('aqi_prediction', version=1)

query = data.select(["pm25", "pm10", "o3", "no2", "so2", "co", "aqi"])

feature_view = fs.get_or_create_feature_view(
    name='aqi_prediction_fv',
    version=1,
    query=query,
    labels=["aqi"]
)

TEST_SIZE = 0.25

td_version, td_job = feature_view.create_train_test_split(
    description = 'pollution dataset',
    data_format = 'csv',
    test_size = TEST_SIZE,
    write_options = {'wait_for_job': True}
)

X_train, X_test, y_train, y_test = feature_view.get_train_test_split(td_version)


losses = []
r2_scores = []

model = XGBRegressor(
    n_estimators = 933,
    max_depth = 3,
    learning_rate = 0.0659553078597774,
    reg_lambda = 0.49527985837464644
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kfold.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model.fit(X_train_fold, y_train_fold)

input_schema = Schema(X_train.values)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

model_schema.to_dict()


model_dir="aqi_prediction_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

joblib.dump(model, model_dir + '/xgboost_aqi_prediction_model.pkl')

mr = project.get_model_registry()

aqi_prediction_model = mr.python.create_model(
    name="aqi_prediction_model", 
    model_schema=model_schema,
    input_example=X_train.sample(), 
    description="AQI predictor")

aqi_prediction_model.save(model_dir)