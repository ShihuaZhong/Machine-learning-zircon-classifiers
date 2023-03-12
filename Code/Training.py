import numpy as np
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

x_train = np.loadtxt('D:/ZSH/F/x_train.txt')
y_train = np.loadtxt('D:/ZSH/F/y_train.txt')
x_val1 = np.loadtxt('D:/ZSH/F/x_val.txt')
y_val = np.loadtxt('D:/ZSH/F/y_val.txt')

x_train = np.log10(x_train)
x_val1 = np.log10(x_val1)

xru_train, yru_train = TomekLinks().fit_resample(x_train, y_train)

scaler = StandardScaler().fit(x_train)
xru_train = scaler.transform(xru_train)
xru_val = scaler.transform(x_val1)

MLP = MLPClassifier(hidden_layer_sizes=(100, 200, 100),
                    random_state=42,
                    max_iter=500,
                    solver='adam',
                    tol=0.001,
                    learning_rate='adaptive')
MLP.fit(xru_train, yru_train)
scores2ru = cross_val_score(MLP, xru_train, yru_train, cv=10, scoring='roc_auc_ovr')
scores2ru = np.append(scores2ru, [scores2ru.mean(), scores2ru.std()])
MLP.score(xru_train, yru_train)
yru_val_pred_mlp = MLP.predict(xru_val)
MLP_cmru = confusion_matrix(y_val, yru_val_pred_mlp)
MLP_f1ru = f1_score(y_val, yru_val_pred_mlp, average='macro')
AUCru = roc_auc_score(y_val, MLP.predict_proba(xru_val), multi_class='ovr')

SVM = SVC(kernel='rbf',
          C=16,
          gamma=0.5,
          cache_size=1000,
          class_weight=None,
          probability=True)
SVM.fit(xru_train, yru_train)
scores2ru = cross_val_score(SVM, xru_train, yru_train, cv=10, scoring='roc_auc_ovr')
scores2ru = np.append(scores2ru, [scores2ru.mean(), scores2ru.std()])
SVM.score(xru_train, yru_train)
yru_val_pred_svm = SVM.predict(xru_val)
SVM_cmru = confusion_matrix(y_val, yru_val_pred_svm)
SVM_f1ru = f1_score(y_val, yru_val_pred_svm, average='macro')
AUCru = roc_auc_score(y_val, SVM.predict_proba(xru_val), multi_class='ovr')
