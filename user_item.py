import numpy as np
import pandas as pd
import datetime
import json

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


def save_dict(filename, dic):
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

def load_dict(filename):
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic
'''
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=r_cols,encoding='latin-1')

user_dict = {}
for i in range(1,6041):
    num = np.where(ratings['user_id']==i)
    list = []
    for j in num[0]:
        list.append(ratings.loc[j].values[1]-1)
    user_dict[i-1] = list
    print(i-1,user_dict[i-1])
    
save_dict("user_dict.json",user_dict)
'''