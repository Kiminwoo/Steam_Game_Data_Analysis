import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
import xlrd
import math
from scipy.stats import boxcox
from sklearn import preprocessing
import re
import seaborn as sns

# Feature selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score

# Feature selection 2

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature selection 3
from sklearn.feature_selection import RFE

# Feature selection 4
from sklearn.feature_selection import RFECV

# Scaling
# (1) StandardScaler code
from sklearn.preprocessing import StandardScaler

# (2) RobustScaler code
from sklearn.preprocessing import RobustScaler

# (3) MinMaxScaler code
from sklearn.preprocessing import MinMaxScaler

# (4) Normalizer code
from sklearn.preprocessing import Normalizer

# SVC 모델 학습시키기
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
DATA_PATH ="C:/Users/user/Desktop/Data/Data.xlsx"
Y_NAME ="game_price"

# game_owners만 int로 해준이유는 엑셀에 딕셔너리 형태로 저장을 시켜놓았었는데, 문자열로 저장되어 있었기 때문에
full_dataframe = pd.read_excel(DATA_PATH, sep=',',dtype={'game_owners':int})


print(type(full_dataframe))
print("\n* Data Shape : ", full_dataframe.shape)
print("\n* class : ", set(full_dataframe["game_price"].values))




full_dataframe.head(5)

# ---------------------------------------데이터의 기본 정보를 출력해 보는 것

# ----------------데이터 구조 파악 하기



# 결측치 확인
print("결측치 확인")
print(full_dataframe[full_dataframe.isnull().any(1)])


# # Data balance 확인 ( price의 비중을 시각화해서 확인해 보인다 )

sb.countplot(x=Y_NAME, data=full_dataframe)
plt.show()



# # Data balance 확인2 ( 카운트별로 갯수와 비중확인 )

labels_count = dict(game_price=0)
labels_count2 = {}


count = []
total = 0

print(labels_count)
print(*np.unique(full_dataframe[Y_NAME], return_counts=True))

for label in full_dataframe[Y_NAME].values:
    if label in labels_count2:
        labels_count2[label] +=1
    else:
        labels_count2[label] =1

for count in labels_count2.values():
    total += count
    print(total)

for label in labels_count2.items():
    print("{0: <15} 개수:{1}개\t데이터비중:{2:.3f}".format(*label, label[1]/total))

del labels_count2

# split data train 70% and test 30%
X_train, X_test, y_train, y_test = train_test_split(full_dataframe, full_dataframe[Y_NAME], test_size=0.3, random_state=42)


# Feature selection with correlation and random forest classificatio (1)

f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(full_dataframe.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)
plt.show()

# split data train 70% and test 30%

# random forest clssifier with n_estimators=10 (default)

clf_rf = RandomForestClassifier(random_state=43)
clr_rf = clf_rf.fit(X_train,y_train)


ac = accuracy_score(y_test,clf_rf.predict(X_test))
print("random forest clssifier")
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")
plt.show()

#  Univariate feature selection and random forest classification (2)

# find best scored 5 features

select_feature = SelectKBest(chi2, k=5).fit(X_train, y_train)

print("find best scored 5 features")
print('Score list : ', select_feature.scores_)
print('Feature list : ', X_train.columns)


# best 5 features

x_train2 = select_feature.transform(X_train)
x_test2 = select_feature.transform(X_test)

# random forest classifier with n_estimators=10 (default)

clf_rf_2 = RandomForestClassifier()
clr_rf_2 = clf_rf_2.fit(x_train2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test2))
print("Accuracy is : ", ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test2))
sns.heatmap(cm_2,annot=True,fmt="d")
plt.show()

#  Recursive feature elimination (RFE) with random forest ( 3 )

# Create the RFE object and rank each pixel

clf_rf_3 = RandomForestClassifier()
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(X_train, y_train)
print("RFE")
print("Chosen best 5 feature by rfe : ", X_train.columns[rfe.support_])



# Recursive feature elimination with cross validation and random forest classification ( 4 )

# The "accuracy" scoring is proportional to the number of correct classifications

clf_rf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

print("Recursive feature elimination")
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])



# Plot number of features VS cross-validation scores ( 5 )
# feature 수가 많아질수록 점수가 낮아짐 --> 이상함

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) +1), rfecv.grid_scores_)
plt.show()

# Tree based feature selection and random forest classification ( 6 )

clf_rf_5 = RandomForestClassifier()
clr_rf_5 = clf_rf_5.fit(X_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
#
print("Feature ranking:")
#
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# # Plot the feature importances of the forest

print("Plot the feature importance")
plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
              color="g", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Scaling

# (1) StandardScaler code

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
print("StandardScaler code")
print('스케일 조정 전 feature Min value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 feature Max value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 후 feature Min value : \n {}'.format(X_train_scale.min(axis=0)))
print('스케일 조정 후 feature Max value : \n {}'.format(X_train_scale.max(axis=0)))

# (2)
# RobustScaler code

scaler = RobustScaler()
X_train_scale2 = scaler.fit_transform(X_train)
print("RobustScaler code")
print('스케일 조정 전 feature Min value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 feature Max value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 후 feature Min value : \n {}'.format(X_train_scale2.min(axis=0)))
print('스케일 조정 후 feature Max value : \n {}'.format(X_train_scale2.max(axis=0)))

# (3)
#  MinMaxScaler code

scaler = MinMaxScaler()
X_train_scale3 = scaler.fit_transform(X_train)
print("MinMaxScaler code")
print('스케일 조정 전 features Min value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 features Max value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 후 features Min value : \n {}'.format(X_train_scale3.min(axis=0)))
print('스케일 조정 후 features Max value : \n {}'.format(X_train_scale3.max(axis=0)))


#  (4)
#  Normalizer code

scaler = Normalizer()
X_train_scale4 = scaler.fit_transform(X_train)

print("Normalizer code")
print('스케일 조정 전 feature Min value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 feature Max value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 후 feature Min value : \n {}'.format(X_train_scale4.min(axis=0)))
print('스케일 조정 후 feature Max value : \n {}'.format(X_train_scale4.max(axis=0)))


# 적용시켜 보기

# 적용 시키기 전 (1)

svc = SVC()
svc.fit(X_train, y_train)
print("적용전")
print('test accuracy : %3f' %(svc.score(X_test, y_test)))
#
# # 적용 시킨 후 (1)
scaler_min = StandardScaler()
X_train_scale5 = scaler_min.fit_transform(X_train)
X_test_scale = scaler_min.transform(X_test)
svc.fit(X_train_scale5, y_train)
print("적용후 ")
print('Scaled test accuracy : %.3f' %(svc.score(X_test_scale,y_test)))

# Scaling 후 결과값
# MinMaxScaler : 0.640 --> 0.262
# StandardScaler : 0.640 --> 0.932
# RobustScaler : 0.640 --> 0.843
# Normalizer : 0.640 --> 0.294




