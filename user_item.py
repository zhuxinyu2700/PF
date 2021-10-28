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
