To run the app follow the below steps:

- make a virtual environment with python version 3.10.12 and install requirements.txt

- Then activate the virtual environment and paste the following command

```bash 
streamlit run app.py
```
# Details about the files

### AQI_Prediction.ipynb
This notebook imports the historical data that needs to be processed . I have chosen one location i,e. mandir-marg, Delhi location. Mixing data of other locations might fill the dataset with unecessary values , so I decided to pick one location only.
Then necessary preprocessing for data transformation like checking for null and duplicate values and removing them, converting the data into necessary datatypes, outlier removal is done. The data is uploaded to feature store.
I did a comparative study of models and found XGBoost to perform well on my data. Necessary transformation is done using featurestore objects and the model is uploaded to hopsworks model registry

### update_data.py
This file contains code for fetching the data from hopsworks feature store, fetching new data from aqicn api, concatinating two datas, and pushing it back to hopsworks feature store.
It runs every 1 hour. I have used github actions to schedule the run.

### retrain_model.py:
This file contains code for fetching data from feature store , model from model registry , retrain the model and push the model to feature store.
It is scheduled to run every week using github actions.

## app.py
This file contains a streamlit app that contains necessary fields for taking user inputs. It  fetches the trained model from hopsworks model registry and predicts on the input data and shows the output in the screen.

To schedule the run of the scripts create a yml file inside .github/workflows and specify the schedule for running the workflow by using the schedule event.
