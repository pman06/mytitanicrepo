
def prediction_model(pclass, sex,age,sibsp,parch,fare,embarked,title):
    import pickle
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_dir = os.path.join(BASE_DIR, 'titanic_model.sav')
    x= [[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    randomforest = pickle.load(open(file_dir, 'rb'))
