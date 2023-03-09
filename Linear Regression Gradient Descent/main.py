#Ibrahim Nobani 1190278
import csv
import matplotlib.pyplot as plt
import numpy as np
Variables=[[],[],[],[],[]]
grades=[]
with open("./grades.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    grades.append(row)
grades.pop(0) #
print(grades)
def zeroVar(col,grades):
  avg = 0
  incr = 0
  for i in grades:
    #print(i[col])
    if(i[col]!='0'):
        avg+=int(i[col])
        incr+=1
  for i in grades:
    if (i[col] == '0'):
      i[col]=str(int(avg/incr))
  print(int(avg/incr))
  return grades

for i in range(0,5):
  (zeroVar(i,grades))
######################################
for i in grades:
  Variables[0].append(int(i[0]))
  Variables[1].append(int(i[1]))
  Variables[2].append(int(i[2]))
  Variables[3].append(int(i[3]))
  Variables[4].append(int(i[4]))
print("HW1:",Variables[0])
print("HW2:",Variables[1])
print("Midterm:",Variables[2])
print("Project:",Variables[3])
print("Final:",Variables[4])
print("The file has been read and its contents are shown above with all missing values set to average.")
print("Press anything (0 to cancel)To examine which of the input variables would be a good predictor for the final exam")
print("Using data visualization scatter plots of each input along with the final mark: ")
inp=input()
if (inp != '0'):
  f, axis = plt.subplots(nrows = 2, ncols = 2)

  axis[0][0].scatter(Variables[0],Variables[4])
  axis[0][0].set_xlabel('HW1')
  axis[0][0].set_ylabel('Final')

  axis[0][1].scatter(Variables[1],Variables[4])
  axis[0][1].set_xlabel('HW2')
  axis[0][1].set_ylabel('Final')

  axis[1][0].scatter(Variables[2],Variables[4])
  axis[1][0].set_xlabel('Midterm')
  axis[1][0].set_ylabel('Final')

  axis[1][1].scatter(Variables[3],Variables[4])
  axis[1][1].set_xlabel('Project')
  axis[1][1].set_ylabel('Final')
  plt.show()
#######################################################
######################################
def valuesSum (grades,num):
  sum=0
  for i in grades:
    sum+=int(i[num])
  return sum
#print(valuesSum(grades,2))
######################################
def valuesConvSum (grades,num,num1):
  sum=0
  for i in grades:
    sum+=(int(i[num])*int(i[num1]))
  return sum
#print(valuesConvSum(grades,2,4))
######################################
def valuesSquareSum (grades,num):
  sum=0
  for i in grades:
    sum+=(int(i[num])*int(i[num]))
  return sum
#print(valuesSquareSum(grades,2))
#####################################
def calculateW(grades,xn,yn,num):
  Dividend=(valuesConvSum(grades, xn, yn)-((valuesSum(grades,xn)*valuesSum(grades,yn))/num))
  Diviser=(valuesSquareSum(grades,xn)-((valuesSum(grades,xn)*valuesSum(grades,xn))/num))
  W1=Dividend/Diviser
  W0=(valuesSum(grades,yn)/num)- (W1*(valuesSum(grades,xn)/num))
  return W0,W1
print("To learn a linear model to predict the final exam, which of the following variables you selected based on the previous part:")
print("1-HW1","2-HW2","3-Midterm","4-Project")
print("Enter the variable number: ")
inp2=input()
while(int(inp2)<1 and int(inp2)>4):
  print("Enter a valid integer between 1 and 4 to procced")
  inp2=input()
print(calculateW(grades,int(inp2)-1,4,32))
W0,W1=calculateW(grades,int(inp2)-1,4,32)
FLR = W0 + W1 * np.array(Variables[2])
print("The linear model obtained is F(x)=",W0,"+",W1,"x")
print("Do you wish to plot this linear model along with the scatter plot?")
print("Press anything to procced (0 to cancel): ")
inp=input()
if (inp != '0'):
  x = np.linspace(0,80)
  fx=W0+(W1*x)
  plt.scatter(Variables[int(inp2)-1],Variables[4])
  plt.plot(x, fx)
  plt.xlabel('Midterm')
  plt.ylabel('Final', color='#1C2833')
  plt.show()
######################################
######################################
def gradient_descent(x, y, iterations, n,LearnRate=0.0001):
  W11 = 0.1
  W00 = 0.01
  iterations = iterations
  LearnRate = LearnRate
  for i in range(iterations):
    finalPr = (W11 * x) + W00
    gradient1 = -(2 / n) * sum(x * (y - finalPr))
    gradient2 = -(2 / n) * sum(y - finalPr)
    W11 = W11 - (LearnRate * gradient1)
    W00 = W00 - (LearnRate * gradient2)
    print("Iteration",i,"W1: ",W11,", W0: ",W00)
  return W11, W00
iterations=300000
print("Moving on to the gradient descent, would you like to enter the number of iterations (Initially set to 300000)?")
print("Press 0 to cancel or the number of iterations you wish to enter")
inpI=input()
if (inpI != '0'):
  iterations=int(inpI)
var1=np.array(Variables[int(inp2)-1])
w1, w0 = gradient_descent(var1, Variables[4],iterations,32)
FGD=w0+w1*np.array(Variables[2])
print("W1: ",w1,"W0: ",w0)
print("The linear model obtained is F(x)=",w0,"+",w1,"x")
print("Do you wish to plot this linear model along with the scatter plot?")
print("Press anything to procced (0 to cancel): ")
inp=input()
if (inp != '0'):
  plt.scatter(Variables[int(inp2)-1], Variables[4])
  x = np.linspace(0,80)
  fxGD = w1*x + w0
  plt.plot(x, fxGD)
  plt.xlabel("Midterm")
  plt.ylabel("Final")
  plt.show()
#####################################
#####################################
from sklearn import linear_model
print("The following is the linear regression using the sci-kit library...")
regr = linear_model.LinearRegression()
var1=(np.array(Variables[int(inp2)-1]))[..., np.newaxis]
var2=(np.array(Variables[4]))[..., np.newaxis]
regr.fit(var1, var2)
Y_axis_predict=regr.predict(var1)
#print("final:",var2)
#print(Y_axis_predict)
FSK = regr.intercept_[0] + regr.coef_[0][0] * np.array(Variables[2])
print("W1: ", regr.coef_)
print("W0: ", regr.intercept_)
print("The linear model obtained is F(x)=",regr.intercept_[0],"+",regr.coef_[0][0],"x")
print("Do you wish to plot this linear model along with the scatter plot?")
print("Press anything to procced (0 to cancel): ")
inp=input()
if (inp != '0'):
  plt.scatter(Variables[int(inp2)-1], Variables[4])
  x = np.linspace(0,80)
  fx2 = regr.coef_[0][0]*x + regr.intercept_[0]
  plt.plot(x, fx2)
  plt.xlabel("Midterm")
  plt.ylabel("Final")
  plt.show()

##########################################
#####################################
print("Now computing the error for each one of the above methods.")
def error(pred, Final,n):
    error = pow((Final - pred), 2)
    return np.sum(error)/n
print("For normal linear regression: ",error(FLR,np.array(Variables[4]),32))
print("For Gradient Descent: ",error(FGD,np.array(Variables[4]),32))
print("For SCIkit learn linear regression: ",error(FSK,np.array(Variables[4]),32))