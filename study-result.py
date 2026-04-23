import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#step 1: create dataset
data ={
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "exam_marks": [35, 40, 50, 55, 60, 68, 75, 80]
}
df =pd.DataFrame(data) 

#step 2: show Dataset
print("dataset")
print(df)

#step 3: visualize data
plt.scatter(df["hours_studied"], df["exam_marks"])
plt.xlabel("hours_studied")
plt.xlabel("exam_marks")
plt.title("study hours vs exam marks")
plt.show()
 
 #step 4: define features and target
X = df[["hours_studied"]]
y= df["exam_marks"]

#step 5: Slipt data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("training data:")
print(X_train)
print("test data:")
print(X_test)
 
 #step 6: create and train model
model= LinearRegression()
model.fit(X_train, y_train)
 
 #step 7: prediction on test data
y_pred= model.predict(X_test)
print("actual _marks:", list(y_test))
print("predicted marks:", list(y_pred))

#step 8: evaluate
mse= mean_squared_error(y_test, y_pred)
print( "mean squred error:",mse)
 
#step 9: predict for new value
new_hours =[[15]]
predicted_mark = model.predict(new_hours)
print(f"if a students studies 15 hours, predicted marks:{predicted_mark[0]:.2f}")

#step 10:plot regression line
plt.scatter(X, y, label="actual data")
plt.plot(X, model.predict(X),label="regression line")
plt.xlabel("hours studied")
plt.ylabel("exam marks")
plt.title("linier regression")
plt.legend()
plt.show()
