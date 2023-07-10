import pandas as pd
from pymongo import MongoClient
import hopsworks


# Connect to MongoDB
client = MongoClient('mongodb+srv://pratyushparida18:password%4018@cluster0.gewdlyg.mongodb.net/')
db = client['index_db']
collection = db['index_collection']


index_doc = collection.find_one()
index = index_doc['index'] 


results = requests.get("https://api.waqi.info/feed/delhi/mandir-marg/?token=4a95eecaa38924fb067d8e35a491ac2d386b0d64")
api_data = results.json()['data']
values = [api_data['iaqi'][param]['v'] for param in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']]
values.append(api_data['aqi'])
date = api_data['time']['s'].split()[0].replace('-', '/')
values.insert(0, date)
values.insert(0, index)

new_data = pd.DataFrame([values], columns=["index","date","pm25", "pm10", "o3", "no2", "so2", "co", "aqi"])
new_data['date'] = pd.to_datetime(new_data['date'])  # Convert 'date' column to timestamp
new_data['no2'] = new_data['no2'].astype(int)  # Convert 'no2' column to bigint
new_data['so2'] = new_data['so2'].astype(int)  # Convert 'so2' column to bigint
new_data['co'] = new_data['co'].astype(int)
new_data['o3'] = new_data['o3'].astype(int)
new_data['pm10'] = new_data['pm10'].astype(int)
new_data['pm25'] = new_data['pm25'].astype(int)

project = hopsworks.login()
fs = project.get_feature_store()

feature_group = fs.get_feature_group('aqi_prediction', version=1)  
feature_group.insert(new_data)



# Update the index value in the collection
collection.update_one({}, {"$set": {"index": index+1}}, upsert=True)
