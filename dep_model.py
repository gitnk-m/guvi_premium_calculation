import pickle
import numpy as np

def model_loader(file_path):
    return pickle.load(open(file_path, 'rb'))





load_model_1 = model_loader("Premium_xgb_model.pkl")
load_model_2 = model_loader("Premium_xgb_model_split.pkl")

test1 = np.array([[19.0,0,10049.0,1,1.0,0,1,22.598761,2,2,2.0,17.0,372.0,5.0,2,0,3,2,2023,12,23]])
test2 = np.array([[39.0,0,31678.0,0,3.0,2,0,15.569731,0,1,1.0,12.0,694.0,2.0,0,1,1,2,2023,6,12]])
test3= np.array([[23.0,1,25602.0,0,3.0,1,1,47.177549,1,2,1.0,14.0,615.0,3.0,1,1,3,2,2023,9,30]])
print("model_1_test_1:",load_model_1.predict(test1))
print("model_2_test_1:",load_model_2.predict(test1))
print("model_1_test_2:",load_model_1.predict(test2))
print("model_2_test_2:",load_model_2.predict(test2))
print("model_1_test_3:",load_model_1.predict(test3))
print("model_2_test_3:",load_model_2.predict(test3))