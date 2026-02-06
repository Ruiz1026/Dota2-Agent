import requests

match_id = 8650430843
url1 = f"https://api.opendota.com/api/matches/{match_id}"
url = 'https://api.opendota.com/api/heroes'
resp = requests.get(url)
data = resp.json()
print(data)
"""
for key in data.keys():
    print(key)
    print(data[key])
    print("-"*100)
"""