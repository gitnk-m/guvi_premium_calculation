import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score

import scipy.stats as stats
import pickle


# Columns in the dataset
# continues=["Age", "Annual Income", "Health Score", "Credit Score","Policy_start_year", "Policy_start_month", "Policy_start_day", "Premium Amount"]
# continues=["Age", "Annual Income", "Health Score", "Credit Score","Policy_start_year", "Policy_start_month", "Policy_start_day"]
# categories = ["Gender", "Marital Status",	"Number of Dependents",	"Education Level", "Occupation", "Location", "Policy Type",	"Previous Claims", "Vehicle Age", "Insurance Duration",	"Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"]
continues=["Age", "Annual Income", "Number of Dependents", "Health Score", "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration", "Policy_start_year", "Policy_start_month", "Policy_start_day"]
categories = ["Gender", "Marital Status", "Education Level", "Occupation", "Location", "Policy Type", "Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"]

# Reading the files
def read_file(file_path):
    df = pd.read_csv(file_path)
    return df

# Handling Null Values
# def null_handler(df):
#     df["Age"] = df.groupby(
#                         "Education Level"
#                     )["Age"].transform(lambda x: x.fillna(x.median()))
#     df["Annual Income"] = df.groupby([
#                                         "Education Level", 
#                                         "Location",
#                                         "Property Type"
#                                     ])["Annual Income"].transform(lambda x: x.fillna(x.median()))
#     df["Marital Status"] = df.groupby(
#                                         pd.cut(df["Age"], [17, 25, 35, 50, 70, 100])
#                                     )["Marital Status"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
#     df["Number of Dependents"] = df.groupby([
#                                                 "Gender", 
#                                                 pd.cut(df["Age"], [17, 25, 35, 50, 70, 100]), 
#                                                 pd.qcut(df["Annual Income"], 5, duplicates="drop"), 
#                                                 "Marital Status"
#                                             ])["Number of Dependents"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
#     df["Occupation"] = df.groupby([
#                                     "Gender", 
#                                     pd.cut(df["Age"], [17, 25, 35, 50, 70, 100]), 
#                                     pd.qcut(df["Annual Income"], 5, duplicates="drop"), 
#                                     "Marital Status",
#                                     "Education Level",
#                                     "Location",
#                                     "Property Type"
#                                 ])["Occupation"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
#     df["Health Score"] = df.groupby([
#                                         "Gender", 
#                                         pd.cut(df["Age"], [17, 25, 35, 50, 70, 100]), 
#                                         "Smoking Status",
#                                     ])["Health Score"].transform(lambda x: x.fillna(x.median()))
#     df["Previous Claims"] = df.groupby([
#                                         pd.cut(df["Age"], [17, 25, 35, 50, 70, 100]), 
#                                         "Smoking Status",
#                                         pd.qcut(df["Annual Income"], 5, duplicates="drop"),
#                                         "Occupation",
#                                         "Policy Type",
#                                     ])["Previous Claims"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
#     df["Vehicle Age"] = df.groupby([
#                                         pd.cut(df["Age"], [17, 25, 35, 50, 70, 100]), 
#                                         pd.qcut(df["Annual Income"], 5, duplicates="drop"),
#                                         "Location",
#                                         "Property Type",
#                                     ])["Vehicle Age"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
#     df["Credit Score"] = df.groupby([
#                                         pd.cut(df["Age"], [17, 25, 35, 50, 70, 100]), 
#                                         pd.qcut(df["Annual Income"], 5, duplicates="drop"),
#                                         "Policy Type",
#                                     ])["Credit Score"].transform(lambda x: x.fillna(x.median()))
#     df["Insurance Duration"].fillna(df["Insurance Duration"].median(), inplace=True)
#     df["Customer Feedback"] = df.groupby([
#                                         pd.cut(df["Age"], [17, 25, 35, 50, 70, 100]), 
#                                         "Education Level",
#                                         "Location",
#                                         pd.qcut(df["Annual Income"], 5, duplicates="drop"),
#                                         "Policy Type",
#                                     ])["Customer Feedback"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
    
#     null_cleared_cols = [
#         "id",
#         "Age",
#         "Annual Income",
#         "Marital Status",
#         "Number of Dependents",
#         "Occupation",
#         "Health Score",
#         "Previous Claims",
#         "Vehicle Age",
#         "Credit Score",
#         "Insurance Duration",
#         "Customer Feedback",
#         "Premium Amount",
#     ]

# #   Not Null in the Train Data 
#     for col in df.columns:
#         if col not in null_cleared_cols:
#             if col in categories:
#                 df[col].fillna(df[col].mode()[0], inplace=True)
#             elif col == "Policy Start Date":
#                 df[col].fillna(method='ffill', inplace=True)
#             else:
#                 df[col].fillna(df[col].median(), inplace=True)
#             # df[col].fillna("Unknown", inplace=True)
    
#     return df


# Simplified Null Handler
# def null_handler(df):
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col].fillna(df[col].mode()[0], inplace=True)
#         elif col == "Policy Start Date":
#             df[col].fillna(method='ffill', inplace=True)
#         else:
#             df[col].fillna(df[col].median(), inplace=True)
#     return df


def null_handler(df):
    for col in df.columns:
        if col == "Policy Start Date" :
            df[col].fillna(method='ffill', inplace=True)
        elif col in categories:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df


# Feature Engineering
def feature_engineering(df):
    date = pd.to_datetime(df["Policy Start Date"], format="%Y-%m-%d %H:%M:%S.%f")
    df["Policy_start_year"] = date.dt.year
    df["Policy_start_month"] = date.dt.month
    df["Policy_start_day"] = date.dt.day
    df.drop(columns=["Policy Start Date"], inplace=True)
    df.set_index("id", inplace=True)
    return df

# Encoding Data
def encoding_data(df):
    enc = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = enc.fit_transform(df[col])
    return df

# standardization function
def standardize_data(df, data_type, scaler_obj=None):
    # scaler = StandardScaler()
    # if data_type == "train":
    #     df[continues] = scaler.fit_transform(df[continues])
    # else:
    #     df[continues] = scaler.transform(df[continues])
    # return df
    if scaler_obj is None:
        scaler_obj = StandardScaler()

    if data_type == "train":
        df[continues] = scaler_obj.fit_transform(df[continues])
    else:
        df[continues] = scaler_obj.transform(df[continues])

    return df, scaler_obj

# Pre-processing Function
def pre_process(file_path):
    df = read_file(file_path)
    df = null_handler(df)
    df = feature_engineering(df)
    df = encoding_data(df)
    df["Annual Income"]=stats.boxcox(df["Annual Income"], lmbda=0.5)
    df["Previous Claims"]=stats.boxcox(df["Previous Claims"], lmbda=0.5)
    return df

# Pickle Function
def pickeler(object, file_path):
    pickle.dump(object, open(file_path, 'wb'))

# Linear Regression Model
def linear_model(x_train, y_train, x_test, y_test):
    model = LinearRegression().fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Linear Regression Model Results")
    print("MAE : ",mean_absolute_error(y_test,y_pred))
    print("MSE : ",mean_squared_error(y_test,y_pred))
    print("R2 : ",r2_score(y_test,y_pred))

# Random Forest Regressor Model
def random_forest_model(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor(    
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1).fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Random Forest Regressor Model Results")
    print("MAE : ",mean_absolute_error(y_test,y_pred))
    print("MSE : ",mean_squared_error(y_test,y_pred))
    print("R2 : ",r2_score(y_test,y_pred))

# Decision Tree Regressor Model
def decision_tree_model(x_train, y_train, x_test, y_test):
    model = tree.DecisionTreeRegressor().fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Decision Tree Regressor Model Results")
    print("MAE : ",mean_absolute_error(y_test,y_pred))
    print("MSE : ",mean_squared_error(y_test,y_pred))
    print("R2 : ",r2_score(y_test,y_pred))

# XGB Regressor Model
def xgb_model(x_train, y_train, x_test, y_test):
    model = XGBRegressor().fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("XGB Regressor Model Results")
    print("MAE : ",mean_absolute_error(y_test,y_pred))
    print("MSE : ",mean_squared_error(y_test,y_pred))
    print("R2 : ",r2_score(y_test,y_pred))



### Reading and Pre-processing the data
train_df = pre_process("Dataset/train.csv")
test_df = pre_process("Dataset/test.csv")
test_y = read_file("Dataset/sample_submission.csv")


### train test split by instruction
x_train_a = train_df.drop(columns=["Premium Amount"])
y_train = train_df["Premium Amount"]
y_test = test_y["Premium Amount"]

### train test split by class
# x = train_df.drop(columns=["Premium Amount"])
# y = train_df["Premium Amount"]
# x_train_a , x_test_a , y_train , y_test = train_test_split(x,y,test_size=0.2)

### standardization
# x_train = standardize_data(x_train_a, data_type="test")
# x_test = standardize_data(x_test_a, data_type="test")
# x_test = standardize_data(test_df)

### GPT suggested standardization
x_train, scaler = standardize_data(x_train_a, data_type="train")
# x_test, _ = standardize_data(x_test_a, data_type="test", scaler_obj=scaler)
x_test, _ = standardize_data(test_df, data_type="test", scaler_obj=scaler)

### Model Training and Evaluation
# linear_model(x_train, y_train, x_test, y_test)
# random_forest_model(x_train, y_train, x_test, y_test)
# decision_tree_model(x_train, y_train, x_test, y_test)
# xgb_model(x_train, y_train, x_test, y_test)


### Model Selected in XGB based on the R2 Score while training the model by splitting the train data
pickeler(XGBRegressor().fit(x_train,y_train), "Premium_xgb_model.pkl")
