import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# The data frame is read for the classifer
dataFrame = pd.read_csv("creditcard_m.csv")
print("Dataset is loaded\n")


# checks wheather there is any missing data in the data frame
missing_data = np.any(np.isnan(dataFrame))


if(missing_data == True):

    # If any missing data is exist, by imputing we can fix that
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(dataFrame)
    imputed_data = imr.transform(dataFrame.values)

    # Deviding the data set into x_value and y_value
    x_value = imputed_data[:, 1:29]
    y_value = imputed_data[:, 30]

else:

    x_value = dataFrame.iloc[:, 1:29].values
    y_value = dataFrame.iloc[:, 30].values

# Randomply splitting x_value and y_value into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.3, random_state=0)

# Standardizing the x_train, x_test
stdsc = StandardScaler()

x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.fit_transform(x_test)

#Reducing the features set into two principle components
pca = PCA(n_components=2)

x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.fit_transform(x_test_std)

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(x_train_pca, y_train)

prediction_rfc_train = rfc.predict(x_train_pca)
prediction_rfc = rfc.predict(x_test_pca)

count_train = 0
count = 0

for i in range(1, len(prediction_rfc_train)-1):
    if prediction_rfc_train[i] == y_train[i]:
        count_train = count_train + 1

for i in range(1, len(prediction_rfc)-1):
    if prediction_rfc[i] == y_test[i]:
        count = count + 1

accurary_rfc_train = count_train / len(prediction_rfc_train)
accurary_rfc = count / len(prediction_rfc)

print("Training accuracy of Random Forest: ", accurary_rfc_train * 100)
print("Testing accuracy of Random Forest: ", accurary_rfc * 100, end="\n\n")

precision_score_rfc = precision_score(y_test, prediction_rfc)
recall_score_rfc = recall_score(y_test, prediction_rfc)
f1_score_rfc = f1_score(y_test, prediction_rfc)
confusion_matrix_rfc = confusion_matrix(y_test, prediction_rfc)

print("recall score of Random forest classifier : ", recall_score_rfc, end="\n\n")
print("precision score of Random forest classifier : ", precision_score_rfc, end="\n\n")
print("f1 score of Random forest classifier : ", f1_score_rfc, end="\n\n")
#print("confusion matrix of support vector machine : ", confusion_matrix_svm)

fig_rfc = plt.figure()
fig_rfc.canvas.set_window_title("Confusion Matrix")
sns.heatmap(confusion_matrix_rfc, annot=True)
plt.title("Random Forest Classifer")
plt.xlabel("Prediction Class")
plt.ylabel("Actual Class")


svmc = svm.SVC(kernel='linear')

svmc.fit(x_train_pca, y_train)

prediction_svm_train = svmc.predict(x_train_pca)
prediction_svm = svmc.predict(x_test_pca)

count_train = 0
count = 0

for i in range(len(prediction_svm_train)):
    if prediction_svm_train[i] == y_train[i]:
        count_train = count_train + 1

for i in range(1, len(prediction_svm)-1):
    if prediction_svm[i] == y_test[i]:
        count = count + 1

accurary_svm_train = count_train / len(prediction_svm_train)
accurary_svm = count / len(prediction_svm)


print("\n\nTraing accuracy of SVM: ", accurary_svm_train * 100)
print("Testing accuracy of SVM: ", accurary_svm * 100, end="\n\n")

precision_score_svm = precision_score(y_test, prediction_svm)
recall_score_svm = recall_score(y_test, prediction_svm)
f1_score_svm = f1_score(y_test, prediction_svm)
confusion_matrix_svm = confusion_matrix(y_test, prediction_svm)

print("recall score of support vector machine : ", recall_score_svm, end="\n\n")
print("precision score of support vector machine : ", precision_score_svm, end="\n\n")
print("f1 score of support vector machine : ", f1_score_svm, end="\n\n")
#print("confusion matrix of support vector machine : ", confusion_matrix_svm)

fig_svm = plt.figure()
fig_svm.canvas.set_window_title("Confusion Matrix")
sns.heatmap(confusion_matrix_svm, annot=True)
plt.title("Support Vector Machine")
plt.xlabel("Prediction Class")
plt.ylabel("Actual Class")
plt.show()
plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('blue', 'red', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

dataFrame_test = pd.read_csv("creditcard_m.csv")

x_test_value = dataFrame_test.iloc[:, 1:29].values
y_test_value = dataFrame_test.iloc[:, 30].values

pca = PCA(n_components=2)

x_test_value_pca = pca.fit_transform(x_test_value)

y_prediction = rfc.predict(x_test_value_pca)


#ploting the testing dataset
for i in range(len(y_test_value)):
    if y_test_value[i] == 0:
        plt.plot(x_test_value_pca[i, 0], x_test_value_pca[i, 1], marker='o', color='b', label='class 0')
    else:
        plt.plot(x_test_value_pca[i, 0], x_test_value_pca[i, 1], marker='x', color='r', label='class 1')

plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.title('Test Features Plot')

plt.figure()

plt.title('Support Vector Machine Classifier')
plot_decision_regions(x_test_value_pca, y_test_value, classifier=svmc)

plt.xlabel('principle component 1')
plt.ylabel('principle component 2')

plt.legend(loc='upper left')

plt.figure()
plt.title('Random Forest Classifier')

plot_decision_regions(x_test_value_pca, y_test_value, classifier=rfc)

plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.legend(loc='upper left')

input("press a key to show the classifiction : ")
plt.show()

