import numpy as np
import pickle


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 7)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    result2 = loaded_model.predict_proba(to_predict)
    print(result2[:,1].item())
    return result[0]


example_list = [45,10,1,1,1,2,3]
print(ValuePredictor(example_list))
