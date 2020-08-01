from django.shortcuts import render
from .fake_model import predict
from . import ml_predict
def home(request):
    return render(request, 'index.html')

def result(request):
    #Take all form inputs
    input_pclass = int(request.POST['pclass'])
    input_sex = (lambda x: 0 if x=='male' else 1)(request.POST['sex'])
    input_age = int(request.POST['age'])
    input_sibsp = int(request.POST['sibsp'])
    input_perch = int(request.POST['perch'])
    input_embarked = int(request.POST['embarked'])
    input_fare = int(request.POST['fare'])
    input_title = int(request.POST['title'])

    #make predictions and generate text version of prediction
    prediction =(lambda x: 'Survived' if x==1 else "Not survived") (ml_predict.prediction_model(input_pclass,input_sex,input_age,input_sibsp,input_perch,input_embarked,input_fare,input_title))
    dict = {'prediction':prediction}
    return render(request, 'result.html', context = dict)
