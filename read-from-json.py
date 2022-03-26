import json
import pandas as pd

# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json

with open('./sarcasm.json','r') as dt:
  datastore = json.load(dt)

dt = pd.DataFrame(datastore)

dt