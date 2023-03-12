import numpy as np
from imblearn.under_sampling import TomekLinks
from joblib import load
from sklearn.preprocessing import StandardScaler

# Read Data
x_train = np.loadtxt('D:/ZSH/F/x_train.txt')
y_train = np.loadtxt('D:/ZSH/F/y_train.txt')
x_val1 = np.loadtxt('D:/ZSH/F/x_val.txt')
y_val = np.loadtxt('D:/ZSH/F/y_val.txt')

x_train = np.log10(x_train)
x_val1 = np.log10(x_val1)

# Balance Data
xru_train, yru_train = TomekLinks().fit_resample(x_train, y_train)

# StandardScaler
scaler = StandardScaler().fit(x_train)
xru_train = scaler.transform(xru_train)
xru_val = scaler.transform(x_val1)


model_svm = load('SVM.joblib')
model_mlp = load('MLP.joblib')


y_pred_svm = model_svm.predict(xru_val)
y_pred_mlp = model_mlp.predict(xru_val)
