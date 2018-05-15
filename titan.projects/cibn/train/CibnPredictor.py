import os
import predictor as pd
import numpy as np

def get_current_model_path(model_dir):
	checkpoint_file_path = os.path.join(model_dir, "checkpoint")	
	with open(checkpoint_file_path, "r") as f:
		for line in f:
			token = line.split(":")[-1].split("/")
			return os.path.join(model_dir, token[-1].strip().strip('"'))


class CibnPredictor(object):
    def __init__(self):
        #model_dir = "/microservice/checkpoint"
        #checkpoint_path = get_current_model_path(model_dir)
        #checkpoint_path = '/tmp/cibn/b8f7a66c-b782-4743-bad8-2f7990cd6c31/model.ckpt-204' 
        checkpoint_path = get_current_model_path('.')
        self.model = pd.Predictor(checkpoint_path)

    def predict(self,X,features_names):
    	return np.reshape(self.model.predict(X), (1, -1))

if __name__ == "__main__":
    p = CibnPredictor()
    x = ["1 1:56707:1 2:110192:0%00484602595AC8DCF361014101FCBF15_282101", "1 1:45866:1 2:110192:0%00484602595AC8DCF361014101FCBF15_265206"]
    print p.predict(x, None)
