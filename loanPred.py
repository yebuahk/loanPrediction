'''
Author : Kevin Yebuah
prediction on whether loan will be paid in full or not
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data.csv')

df.head()
df.describe()
df.info()

df.isnull().any().any()
df.isnull().sum().sum()

df.columns

X = df[['fico','int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
        'days.with.cr.line', 'revol.bal', 'revol.util']]

y = df['not.fully.paid']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() # add solver or multi-class for AUC
classifier.fit(X_train, y_train)

classifier.coef_
classifier.intercept_

y_pred = classifier.predict(pd.DataFrame(X_test))
y_pred

y_pred1 = classifier.predict_proba(X_test)
y_pred1 = pd.DataFrame(y_pred1)
y_pred1.columns =['fully pay', 'NotFullyPay']    # this has a a threshold of 0.5 by default

y_pred1[y_pred1['NotFullyPay']>0.80]




from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
cm = confusion_matrix(y_pred, y_test)
cm

pd.crosstab(y_pred, y_test)

total = sum(sum(cm))

accuracy = (cm[0, 0]+cm[1, 1])/total
print('Accuracy :', accuracy)
classifier.score(X_test, y_test)    # calculate accuracy automatically instead of manually

sensitivity = cm[0, 0]/(cm[0, 0]+cm[0, 1])
print('Accuracy :', sensitivity)

specificity = cm[1, 1]/(cm[1, 0]+cm[1, 1])
print('Accuracy :', specificity)

precision = cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Accuracy :', precision)


print(classification_report(y_test, y_pred))    # auto calculate all scores

''' average calculations e.g. 1.00 * 2406 + 0,00 * 468 / total '''

''' increasing threshold to improve sensitivity or specificity if score is low'''
def cutoff(x):
    x = np.where(x > 0.9, 0, 1)
    return x

y_pred2 = (cutoff(y_pred1.NotFullyPay))
cm2 = confusion_matrix(y_test, y_pred2)
cm2

accuracy2 = (cm2[0, 0]+cm2[1, 1])/total
print('Accuracy :', accuracy2)
classifier.score(X_test, y_test)    # calculate accuracy automatically instead of manually

sensitivity2 = cm2[0, 0]/(cm2[0, 0]+cm2[0, 1])
print('Accuracy :', sensitivity2)

''' after changing threshold to 0.9 in cutoff function there is an increase in specificity
this will affect other scores
'''
specificity2 = cm2[1, 1]/(cm2[1, 0]+cm2[1, 1])
print('Accuracy :', specificity2)

precision2 = cm2[0, 0]/(cm2[0, 0]+cm2[1, 0])
print('Accuracy :', precision2)


''' area under curve'''

fpr, tpr = roc_curve(y_test, y_pred1.iloc[:, 1])
df_roc = pd.DataFrame(dict(fpr=fpr, tpr=tpr))


AUC = auc(fpr, tpr)

plt.plot(fpr, tpr, label='auc = %0.2f' % AUC)
plt.legend(loc='lower right')
plt.plot([0.1], [0.1], '---r')  #Show mid area
plt.show()

