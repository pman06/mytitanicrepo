import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle # To store machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")

#This function accepts the name of an individual in our data and return the title of the individual
def get_title(name):
    if "." in name:
        return name.split(",")[1].split(".")[0].strip()
    else:
        return "unknown"

#Create a function for a shorter title

def short_title(x):
    title = x['Title']
    if title  in ["Col", "Major", "Capt", ""]:
        return "Officer"
    elif title in ["the Countess", "Lady", "Don", "Jonkheer", "Sir", "Dona"]:
        return "Royalty"
    elif title == "Mme":
        return "Mrs"
    elif title in ["Mlle", "Ms"]:
        return "Miss"
    else:
        return title


#Create a new column named "Title" and store the title of each "Name" record in the data

data["Title"] = data["Name"].map(lambda x : get_title(x))
#Now to have a more shorter title (group our titles to "Royalty","officer","Miss" etc)
data["Title"] = data.apply(short_title, axis =1)


data['Age'].fillna(data["Age"].median(), inplace= True)
data["Embarked"].fillna("S", inplace =True)
data['Fare'].fillna(data['Fare'].median(), inplace =True)
del data['Cabin']
data.drop("Name", axis =1, inplace= True)
data.drop("Ticket", axis=1, inplace =True)
data.Sex.replace(('male', 'female'), (0,1), inplace =  True)
data.Embarked.replace(('S', 'C', 'Q'), (0,1,2), inplace=True)
#Encode Title values to numbers
data.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace =True)

X = data.drop(['Survived','PassengerId'], axis =1)
Y = data['Survived']
xtrain, xtest,ytrain, ytest = train_test_split(X,Y, test_size = 0.1)

randomforest = RandomForestClassifier()
randomforest.fit(xtrain,ytrain)

filename = 'titanic_model.sav'
pickle.dump(randomforest, open(filename, 'wb'))
