import pandas
import joblib

from sklearn.linear_model import LinearRegression

ds=pandas.read_csv("SalaryData.csv")

#ds.info()

y=ds['Salary']
x=ds['YearsExperience'].values.reshape(-1,1)

#exp=int(input("Experience:"))
mind=LinearRegression()

mind.fit(x,y)

#exp=int(input("Enter your Experience")

#print(mind.predict([[exp]]))

joblib.dump(mind,'Salarypredictor.pk1')
