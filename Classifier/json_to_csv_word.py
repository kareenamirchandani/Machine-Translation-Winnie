# This file takes the json file containing individual words and labels
# It converts this json file to a csv file

import json
import csv

# https://www.geeksforgeeks.org/convert-json-to-csv-in-python/

with open('sunbird_split_word.json') as json_file:
    jsondata = json.load(json_file)

# jsondata = jsondata.replace("\\ua78", "")

data_file = open('sunbird_split_word.csv', 'w', newline='')
csv_writer = csv.writer(data_file)

count = 0
for data in jsondata:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())

data_file.close()
