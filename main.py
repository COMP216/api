import flask
from flask import request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import os
import io
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn import metrics
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, plot_roc_curve
app = flask.Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flask_db.db'
db = SQLAlchemy(app)
# Load "model.pkl"
clf1 = joblib.load("D:/STUDY/SEM 5/COMP309/Group Project/model_clf1.pkl")
selectedCols = joblib.load(
    "D:/STUDY/SEM 5/COMP309/Group Project/selected_columns.pkl")
X = pd.read_csv("D:/STUDY/SEM 5/COMP309/Group Project/X.csv")
Y = pd.read_csv("D:/STUDY/SEM 5/COMP309/Group Project/Y.csv")


class FileContents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)


class ModelAttributes():
    scores = []
    meanScore = 0
    confusionMatrix = []


@app.route('/', methods=['GET'])
def home():
    return "API server is running"


@app.route('/api/v1/bicycletheft/getAttributes', methods=['GET'])
def getAttributes():
    # calculate scores
    scores = cross_val_score(linear_model.LogisticRegression(
        solver='lbfgs'), X, Y, scoring='accuracy', cv=10)

    # calculate confusion matrix
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    probs = clf1.predict_proba(X_test)
    prob = probs[:, 1]
    prob_df = pd.DataFrame(prob)
    prob_df['predict'] = np.where(prob_df[0] >= 0.05, 1, 0)
    Y_A = Y_test.values
    Y_P = np.array(prob_df['predict'])
    confusionMatrix = confusion_matrix(Y_A, Y_P)

    # istantiate model attribute object
    attributes = ModelAttributes()
    attributes.scores = [round(num, 3) for num in scores.tolist()]
    attributes.meanScore = round(scores.mean(), 3)
    attributes.confusionMatrix = confusionMatrix.tolist()
    return jsonify(attributes.__dict__)

# we need to show scor


@app.route('/api/v1/bicycletheft/predict', methods=['POST'])
def predict():
    # ============RUN PREDICTION ON INPUT FILE====================
    # Processing user input file
    inputFile = io.StringIO(request.data.decode('utf-8'))
    data_user_input = pd.DataFrame.from_dict(
        request.json, orient='index', dtype=None, columns=None)
    # print(data_user_input.dtypes)
    # Handle Missing data for input file
    data_user_input['Cost_of_Bike'] = pd.to_numeric(
        data_user_input['Cost_of_Bike'], errors='coerce')
    data_user_input['Cost_of_Bike'].fillna(
        data_user_input['Cost_of_Bike'].mean(), inplace=True)
    data_user_input['Bike_Make'].fillna('Missing', inplace=True)
    data_user_input['Bike_Model'].fillna('Missing', inplace=True)
    data_user_input['Bike_Type'].fillna('Missing', inplace=True)
    data_user_input['Bike_Colour'].fillna('Missing', inplace=True)
    data_user_input['Bike_Speed'].fillna(0, inplace=True)

    # Change output column to integer
    data_user_input['Status'] = (
        data_user_input['Status'] == 'RECOVERED').astype(int)
    # print(data_bicycle_theft['Status'].value_counts())

    # reduce categories
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'ASSAULT - RESIST/ PREVENT SEIZ', 'ASSAULT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'ASSAULT WITH WEAPON', 'ASSAULT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == "B&E W'INTENT", 'B&E', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'DRUG - POSS COCAINE (SCHD I)', 'DRUG', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'DRUG - TRAF CANNABIS (SCHD II)', 'DRUG', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'DRUG - TRAF OTHER (SCHD I)', 'DRUG', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'FRAUD - IDENTITY/PERS W-INT', 'FRAUD', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'FRAUD OVER', 'FRAUD', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'FRAUD UNDER', 'FRAUD', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'FTC PROBATION ORDER', 'FTC', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'FTC WITH CONDITIONS', 'FTC', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'MISCHIEF - ENDANGER LIFE', 'MISCHIEF', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'MISCHIEF - INTERFERE W-PROP', 'MISCHIEF', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'MISCHIEF TO VEHICLE', 'MISCHIEF', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'MISCHIEF UNDER', 'MISCHIEF', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'PUBLIC MISCHIEF', 'MISCHIEF', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'POSSESSION HOUSE BREAK INSTRUM', 'POSSESSION', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'POSSESSION PROPERTY OBC OVER', 'POSSESSION', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'POSSESSION PROPERTY OBC UNDER', 'POSSESSION', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'ROBBERY - HOME INVASION', 'ROBBERY', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'ROBBERY - MUGGING', 'ROBBERY', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'ROBBERY - OTHER', 'ROBBERY', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'ROBBERY - SWARMING', 'ROBBERY', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'ROBBERY WITH WEAPON', 'ROBBERY', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT FROM MOTOR VEHICLE OVER', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT FROM MOTOR VEHICLE UNDER', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT OF EBIKE OVER $5000', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT OF EBIKE UNDER $5000', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT OF MOTOR VEHICLE', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT OVER', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT OVER - BICYCLE', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT UNDER', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT UNDER - BICYCLE', 'THEFT', data_user_input['Primary_Offence'])
    data_user_input['Primary_Offence'] = np.where(
        data_user_input['Primary_Offence'] == 'THEFT UNDER - SHOPLIFTING', 'THEFT', data_user_input['Primary_Offence'])

    orginal_data = data_user_input

    # DATA MODELING
    cat_vars = ['Primary_Offence', 'Occurrence_Month',
                'Occurrence_Day', 'Division', 'Location_Type', 'Premise_Type',
                'Hood_ID']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(data_user_input[var], prefix=var)
        data_user_input1 = data_user_input.join(cat_list)
        data_user_input = data_user_input1

    # Adding features(columns) that exist in predictive model but don't exist in input file
    correctedCols = []
    for var in selectedCols:
        if var not in data_user_input.columns:
            correctedCols.append(var)

    dummy = pd.DataFrame(
        np.zeros((len(data_user_input.index), len(correctedCols))))
    dummy.columns = correctedCols
    data_user_input = data_user_input.join(dummy)
    #  Removee the original columns
    cat_vars = ['Primary_Offence', 'Occurrence_Year', 'Occurrence_Month',
                'Occurrence_Day', 'Occurrence_Time', 'Division', 'City',
                'Location_Type', 'Premise_Type', 'Bike_Make', 'Bike_Model',
                'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'Cost_of_Bike',
                'Hood_ID', 'event_unique_id', 'Neighbourhood', 'Occurrence_Date', 'City',
                'Occurrence_Time', 'Bike_Make', 'Bike_Model',
                'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'X', 'Y']
    data_user_input_vars = data_user_input.columns.values.tolist()
    to_keep = [i for i in data_user_input_vars if i not in cat_vars]
    data_user_input_final = data_user_input[to_keep]
    # Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
    data_input_X = data_user_input_final[selectedCols]
    data_input_Y = data_user_input_final['Status']
    data_input_X.fillna(0, inplace=True)
    prodProbs = clf1.predict_proba(data_input_X)
    prodPredicted = clf1.predict(data_input_X)
    prodProb = prodProbs[:, 1]
    prodProb_df = pd.DataFrame(prodProb)

    orginal_data['predict'] = np.where(prodProb_df[0] >= 0.05, 1, 0)
    orginal_data.to_csv(
        r'D:/STUDY/SEM 5/COMP309/Group Project/userOutputFile.csv', index=True)

    return "Run successfully"


if __name__ == '__main__':
    app.run(debug=True)
