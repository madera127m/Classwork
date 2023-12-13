




import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




df=pd.read_csv("auto-mpg.csv")

df.head()

y=np.array(df["displacement"]).reshape(398,1)

np.shape(y)

X=np.array(df["acceleration"]).reshape(398,1)

np.shape(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

fig, ax = plt.subplots()
ax.scatter(X_train,y_train)
ax.set_xlabel('Acceleration')
ax.set_ylabel('Displacement')
fig.savefig('xtrain_accel_vs_ytrain_disp.png')

lr=LinearRegression()
lr.fit(X_train,y_train)

minX=min(X_test)
maxX=max(X_test)

[miny, maxy] = lr.predict(np.array([minX,maxX]).reshape(-1,1))

miny,maxy

fig, ax = plt.subplots()
ax.plot([minX, maxX], [miny, maxy])
ax.set_xlabel('Acceleration')
ax.set_ylabel('Displacement')
fig.savefig('linear_regression.png')

r_squared = lr.score(X_test, y_test)
r_squared
with open('scriptnotes.txt', 'w') as notes_file:
        notes_file.write("Dataset Info:\n")
        notes_file.write(str(df.info()) + '\n\n')

        notes_file.write("Summary Statistics:\n")
        notes_file.write(str(df.describe()) + '\n\n')
        r_squared = lr.score(X_test, y_test)
        notes_file.write(f"R-squared score from lr.score of the test data: {r_squared:.4f}\n")

with open('report.txt', 'w') as report_file:
       
        report_file.write("The graphs shows a negative correlation between displacement and acceleration.\
                          \nThey are related in that greater engine displacement produces more power while less increases effiency.\
                          \nIn this case the r squared score was about .2. This means that the data match is very good this might \
                          because of some outliers in the data .")

