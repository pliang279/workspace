import json
 
# Opening JSON file
with open('config_files/9.json') as json_file:
    data = json.load(json_file)
    print(data)