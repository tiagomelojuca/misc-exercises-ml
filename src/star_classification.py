# Exercise from https://www.youtube.com/watch?v=4HU3lqj0Cc8
from pch import *

df = load_dataframe('star_classification.csv')

X = df.loc[: , df.columns != 'class'].values
y = df['class'].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.preprocessing import StandardScaler
scale_obj = StandardScaler()
X = scale_obj.fit_transform(X.astype(float))

from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.15)

# from sklearn import decomposition
# pca = decomposition.PCA(n_components=10)
# XTrain = pca.fit_transform(XTrain)
# XTest = pca.transform(XTest)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=400)
start_time = time.perf_counter()
model.fit(XTrain, yTrain)
end_time = time.perf_counter()

print(f'Model trained in {end_time - start_time} seconds with score {model.score(XTest, yTest)}')

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, XTest, yTest)
plt.show()
