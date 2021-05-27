import joblib
exp=int(input("Work Experience:"))
mind=joblib.load('Salarypredictor.pk1')

print()
print("Your Salary will be:-> ",end='')
print(mind.predict([[exp]]))
