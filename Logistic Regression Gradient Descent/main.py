import csv
import matplotlib.pyplot as plt
import numpy as np
trainedData=[]
trainedDataC1=[[],[]]
trainedDataC2=[[],[]]
testDataC1=[[],[]]
testDataC2=[[],[]]
with open("./train.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    trainedData.append(row)
trainedData.pop(0)
print(trainedDataC1)
Y=[]
X=[[],[]]
for i in trainedData:
  X[0].append(float(i[0]))
  X[1].append(float(i[1]))
  if (i[2]=='C1'):
    trainedDataC1[0].append(float(i[0]))
    trainedDataC1[1].append(float(i[1]))
    Y.append(1)
  elif (i[2]=='C2'):
    trainedDataC2[0].append(float(i[0]))
    trainedDataC2[1].append(float(i[1]))
    Y.append(0)
print("The data has been read, do you wish to see the scatter plot before we apply the logistic regression? ")
print("Press anything to procced (0 to cancel): ")
inp=input()
if (inp != '0'):
  f, axis = plt.subplots()
  axis.scatter(trainedDataC1[0],trainedDataC1[1])
  axis.scatter(trainedDataC2[0],trainedDataC2[1],color = 'hotpink')
  axis.set_xlabel('X1')
  axis.set_ylabel('X2')
  plt.show()

def gradient_descent(x, y, iterations, n,LearnRate=0.001):
  W1 = 0.1
  W0 = 0.01
  W2=0.1
  iterations = iterations
  LearnRate = LearnRate
  for i in range(iterations):
    finalPr = 1 / (1 + np.exp(-(W1 * x[0]) +-(W2 * x[1])+ -W0))
    gradient1 = -(2 / n) * sum(x[0] * (y - finalPr))
    gradient2 = -(2 / n) * sum(y - finalPr)
    gradient3= -(2 / n) * sum(x[1] * (y - finalPr))
    W1 = W1 - (LearnRate * gradient1)
    W2=W2 - (LearnRate * gradient3)
    W0 = W0 - (LearnRate * gradient2)
    print("Iteration",i," W0: ",W0,"W1: ",W1,"W2: ",W2)
  return W0, W1, W2
X=np.array(X)
print("Gradient Descent is going to run now,Press anything to procced...")
inp=input()
if (inp != '000'):
  Theta0,Theta1,Theta2=gradient_descent(X,Y,100000,len(Y))
  #Theta0,Theta1,Theta2=-0.05180900571531209,-1.1621171622480848,0.5978027589756287
  print("Theta0: ",Theta0,"Theta1: ",Theta1,"Theta2: ",Theta2)
  print("The logistic model obtained is =", Theta0, "+", Theta1, "x1", "+", Theta2, "x2")
print("Do you with to plot the decision boundary for the logistic regression model obtained?  ")
print("Press anything to proceed (0 to cancel): ")
def plotN(Theta0,Theta1,Theta2,DataC1,DataC2):
  f, axis = plt.subplots()
  axis.scatter(DataC1[0], DataC1[1])
  axis.scatter(DataC2[0], DataC2[1], color='hotpink')
  axis.set_xlabel('X1')
  axis.set_ylabel('X2')
  X1 = np.linspace(-0.5, 1)
  X2 = -(Theta0 + (Theta1 * X1)) / (Theta2)
  plt.plot(X1, X2)
  plt.show()
inp=input()
if (inp != '0'):
  plotN(Theta0,Theta1,Theta2,trainedDataC1,trainedDataC2)
###############################################################
#Reading the test data to calculate the accuracy ##
###############################################################
testData=[]
with open("./test.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    testData.append(row)
testData.pop(0)
YT=[]
XT=[[],[]]
for i in testData:
  XT[0].append(float(i[0]))
  XT[1].append(float(i[1]))
  if (i[2]=='C1'):
    testDataC1[0].append(float(i[0]))
    testDataC1[1].append(float(i[1]))
    YT.append(1)
  elif (i[2]=='C2'):
    testDataC2[0].append(float(i[0]))
    testDataC2[1].append(float(i[1]))
    YT.append(0)
print("Do you also wish to plot the test model?")
print("Press anything to procced (0 to cancel): ")
inp=input()
if (inp != '0'):
  XT = np.array(XT)
  plotN(Theta0,Theta1,Theta2,testDataC1,testDataC2)
#Accuracy functions that takes input X and Y.
def accuracy (X,Y,Theta0,Theta1,Theta2):
  count = 0
  for i in range(len(X[0])):
    Y2 = Theta0 + Theta1 * X[0][i] + Theta2 * X[1][i]
    p = 0
    if Y2 >= 0:
        p = 1
    if p == Y[i]:
        count+=1
  accuracy = count / len(Y)
  return accuracy
print("The accuracy for the trained data is: ",accuracy(X,Y,Theta0,Theta1,Theta2))
print("The accuracy for the test data is: ",accuracy(XT,YT,Theta0,Theta1,Theta2))
####################################################################################
#End of part1
####################################################################################
print("Gradient Descent for part 2 is going to run now,Press anything to proceed...")
inp=input()
if (inp != '000'):
  Xsq = np.square(X)
  Xsq=np.array(Xsq)
  Theta0,Theta1,Theta2=gradient_descent(Xsq,Y,100000,len(Y))
  print("Theta0: ", Theta0, "Theta1: ", Theta1, "Theta2: ", Theta2)
  print("The logistic model obtained is =", Theta0, "+", Theta1, "x1^2", "+", Theta2, "x2^2")
  #Theta0,Theta1,Theta2=-2.6446032684652816,4.6595123504592415,7.088818963290747
def accuracySQ (X,Y):
  count = 0
  for i in range(len(X[0])):
    Y2 = Theta0 + Theta1 * pow(X[0][i],2) + Theta2 * pow(X[1][i],2)
    p = 0
    if Y2 >= 0:
        p = 1
    if p == Y[i]:
        count+=1
  accuracy = count / len(Y)
  return accuracy
##########################
#Function to plot squared functions
##########################
def plotSQ(Theta0,Theta1,Theta2,DataC1,DataC2):
  f, axis = plt.subplots()
  axis.scatter(DataC1[0], DataC1[1])
  axis.scatter(DataC2[0], DataC2[1], color='hotpink')
  axis.set_xlabel('X1')
  axis.set_ylabel('X2')
  X1 = np.linspace(-1, 1, 100)
  X2 = np.linspace(-1, 1, 100)
  X1, X2 = np.meshgrid(X1, X2)
  F = Theta0 + Theta1 * pow(X1, 2) + Theta2 * pow(X2, 2)
  F = np.array(F)
  # F = F.reshape((len(X1), len(X2)))
  plt.contour(X1, X2, F, [0])
  # plt.plot(X1,X2,F)
  plt.show()
print("Do you with to plot the decision boundary for the logistic regression model obtained?  ")
print("Press anything to procced (0 to cancel): ")
inp=input()
if (inp != '0'):
  plotSQ(Theta0,Theta1,Theta2,trainedDataC1,trainedDataC2)
print("Do you also wish to plot the test data?")
print("Press anything to proceed (0 to cancel): ")
inp=input()
if (inp != '0'):
  XsqT = np.square(XT)
  XsqT = np.array(XsqT)
  plotSQ(Theta0,Theta1,Theta2,testDataC1,testDataC2)
print("The accuracy for the trained data is: ",accuracySQ(X,Y))
print("The accuracy for the test data is: ",accuracySQ(XT,YT))
###################################################################
###################################################################
print("The following is the linear regression using the sci-kit library...")
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
def logisticRegSK(X,Y,XT,YT,C):
  model = LogisticRegression(C=C, solver="lbfgs")
  X2T = XT.transpose()
  model.fit(X2T, YT)
  scoreT = model.score(X2T, YT)
  X2=X.transpose()
  model.fit(X2,Y)
  theta = model.coef_[0]
  print(model.intercept_[0],theta)
  print("Theta0: ",model.intercept_[0],"Theta1 and Theta2: ",theta)
  score = model.score(X2, Y)
  print("The accuracy of this train model: ",score)
  print("The accuracy of this test model: ", scoreT)
  return model.intercept_[0],model.coef_[0][0],model.coef_[0][1]
Theta=logisticRegSK(X,Y,np.array(XT),YT,0.001)
plotN(Theta[0],Theta[1],Theta[2],trainedDataC1,trainedDataC2)
plotN(Theta[0],Theta[1],Theta[2],testDataC1,testDataC2)
Theta=logisticRegSK(np.array(Xsq),Y,np.array(np.square(XT)),YT,1e2)
plotSQ(Theta[0],Theta[1],Theta[2],trainedDataC1,trainedDataC2)
plotSQ(Theta[0],Theta[1],Theta[2],testDataC1,testDataC2)



