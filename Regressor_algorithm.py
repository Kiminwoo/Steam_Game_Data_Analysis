import pandas as pd

import numpy as np

import statsmodels.api as sm


from sklearn import linear_model

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Evaluating the Algorithm
from sklearn import metrics

# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor,RandomForestRegressor

# AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

# Dummy estimators
from sklearn.dummy import DummyRegressor

# Gaussian Proecesses
from sklearn.gaussian_process import GaussianProcessRegressor


# neural_network.MLPRegressor
from sklearn.neural_network import MLPRegressor



# Excel read
# Excel 파일을 불러와서 full_dataframe으로 만든다.
# 이때 구분자를 ',' 로 해서 불러온다.
pd.set_option('display.max_columns', None)
DATA_PATH ="C:/Users/user/Desktop/Data/Data.xlsx"
full_dataframe = pd.read_excel(DATA_PATH, sep=',')


# import statsmodels.api as sm
# statsmodels.api를 이용해서 X,y 값을 도표로 수치화해서 볼 수 있게만들었다.

X = full_dataframe[["game_positive","game_negative","game_owners","game_initialprice","game_discount"]]
y = full_dataframe["game_price"]


Y_NAME = "game_price"

# Note the difference in argument order
model = sm.OLS(y,X).fit()

# --> model 생성


# make the predictions by the model
predictions = model.predict(X)

# 위에서 만든 model을 기반으로 Y를 예측


#Print out the statistics
model.summary()


# model의 내용을 보기위해서 summary를 사용

# OLS : Ordinary Least Squares
# Least Squears : 회귀 직선과의 거리의 제곱을 최소화하는 회귀직선에 적합하도록 하는 메소드
# DF : degrees of freedom ( 최종 결과값에서의 변화가 자유로운 통계 수 )

print(model.summary())

# 도표 출력
# --> model.summary를 통해 feature가 price에 미치는 영향을 알아본다.


# ---------------------------- LinearRegression

# split data train 70% and test 30%
X_train, X_test, y_train, y_test = train_test_split(full_dataframe, full_dataframe[Y_NAME], test_size=0.3, random_state=43)
# test_size = 0.3 을 하게되면 test_Data 가 30% 비중을 차지하게 되고 , 자동으로 train_data가 70%로 적용이 된다
# random_state=0으로 두는것은 위험.
# 그래서 random_state가 0일때와 43일때 두가지 경우를 구해보았다.



im = linear_model.LinearRegression()
# LinearRegression()을 사용하기 위해서 변수 im에 저장시킨다.

model = im.fit(X,y)
#  LinearRegression에 나의 데이터셋을 fit시킨다.

prediction_linear = im.predict(X)
# 알고리즘의 predict를 나의 데이터셋 X를 사용해서 예측한다.

# # 예측값과 실제값을 비교

plt.scatter(prediction_linear,y,s=10)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("coef :"+str(im.coef_))
print("score :"+str(im.score(X,y)))
print("intercept :"+str(im.intercept_))
print("Actual price  : "+ str(y))
print("predict price :"+str(prediction_linear))

# # 실질적으로 예측된 가격을 예측전의 가격과 비교해 보고싶어서 합쳐준다.
# # 일단 딕셔너리 형태로 저장
Result = {
        'Actual price' : y,
        'predict price' : prediction_linear
}

# # 저장되어진 딕셔너리형태를 Dataframe형태로 저장
Result_df = pd.DataFrame(Result)
print(Result_df)



#  --------------------------DecisionTreeRegressor

regressor = DecisionTreeRegressor()

# DecisionTreeRegressor() 함수를 만든다.

regressor.fit(X_train,y_train)

# 위에서 나누어준 훈련데이터를 DecisionTreeRegressor에 fit시킨다.

# To make predictions on the test set, uses the predict method

y_pred = regressor.predict(X_test)

# 위에서 나누어준 테스트데이터에 DecisionTreeRegressor로 예측을 한다.

df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)

# 알고리즘 적용전의 가격과 알고리즘 적용 후의 가격을 비교하고 싶어서 print로 출력.
# # mean_absolute_error 함수는 평균 절대 오류, 절대 오류 손실 또는 기대치 손실의 예상 값에 해당하는 위험 메트릭을 계산합니다.

print('DecisionTreeRegressor Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))

# # mean_squared_log_error 함수는 제곱 된 로그 (2 차) 오류 또는 손실의 예상 값에 해당하는 위험 메트릭을 계산합니다.
print('DecisionTreeRegressor Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))

# # 평균 제곱근 편차 또는 평균 제곱근 오차는 추정 값 또는 모델이 예측한 값과 실제 환경에서
# # 관찰되는 값의 차이를 다룰 떄 흔히 사용하는 측도이다.
# # 정밀도를 표현하는데 적합하다.
# # 각각의 차이값은 잔차라고도 하며, 평균 제곱근 편차들을 하나의 측도로 종합할 때 사용된다.
print('DecisionTreeRegressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('DecisionTreeRegressor Accuracy:', metrics.r2_score(y_test,y_pred))


# R-squared : 제곱근을 의미하는데 1에 가까울 정확한 값을 의미
print("DecisionTreeRegressor R-squared for Train: %.2f" %regressor.score(X_train,y_train))
print("DecisionTreeRegressor R-squared for Test: %.2f" %regressor.score(X_test,y_test))

# ------------------------------------------- Output

#random_state=0 일때
# DecisionTreeRegressor Mean Absolute Error: 0.7187215650591446
# DecisionTreeRegressor Mean Squared Error: 1396.9334622383985
# DecisionTreeRegressor Root Mean Squared Error: 37.375573069029976
# DecisionTreeRegressor Accuracy: 0.9984125399095579

# random_state=43일때
# DecisionTreeRegressor Mean Absolute Error: 2.4709963603275704
# DecisionTreeRegressor Mean Squared Error: 43819.10111464968
# DecisionTreeRegressor Root Mean Squared Error: 209.33012471846874
# DecisionTreeRegressor Accuracy: 0.9673605977859902
# DecisionTreeRegressor R-squared for Train: 1.00
# DecisionTreeRegressor R-squared for Test: 0.97

# -----------------------------Gradientboostingregressor

gbrt = GradientBoostingRegressor()
# GradientBoostingRegressor 선언
gbrt.fit(X_train,y_train)
# 나의 데이터를 GradientBoostingRegressor에 적용시킨다.
gbrt_y_pred = gbrt.predict(X_test)
# GradientBoostingRegressor 알고리즘을 통해 Y를 예측한다.

print("Feature importances")
print(gbrt.feature_importances_)
# GradientBoostingRegressor 메소드중 feature 중요도를 나타내는 것이 있었다.

print('Gradientboostingregressor Mean Absolute Error:', metrics.mean_absolute_error(y_test,gbrt_y_pred))
print('Gradientboostingregressor Mean Squared Error:', metrics.mean_squared_error(y_test,gbrt_y_pred))
print('Gradientboostingregressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,gbrt_y_pred)))
print('Gradientboostingregressor Accuracy:', metrics.r2_score(y_test,gbrt_y_pred))

# 예측 전의 Y 값과 예측 후의 Y값
gbrt_df = pd.DataFrame({'Actual':y_test, 'Predicted':gbrt_y_pred})
print(gbrt_df)
# 훈련데이터와 테스트데이터 각각의 알고리즘에 대한 정확도
print("R-squared for Train: %.2f" %gbrt.score(X_train,y_train))
print("R-squared for Test: %.2f" %gbrt.score(X_test,y_test))


# ------------------------------------------- Output
# random_state=0일때
# Gradientboostingregressor Mean Absolute Error: 0.7922192368407802
# Gradientboostingregressor Mean Squared Error: 16.667078349470625
# Gradientboostingregressor Root Mean Squared Error: 4.082533324967553
# Gradientboostingregressor Accuracy: 0.9999810597122775
# R-squared for Train: 1.00
# R-squared for Test: 1.00

# random_state=43일때

# Gradientboostingregressor Mean Absolute Error: 2.9070040004338233
# Gradientboostingregressor Mean Squared Error: 43876.08301832946
# Gradientboostingregressor Root Mean Squared Error: 209.46618585902942
# Gradientboostingregressor Accuracy: 0.9673181538465707
# R-squared for Train: 1.00
# R-squared for Test: 0.97

# -----------------------------------AdaBoostRegressor
abr = AdaBoostRegressor()
# AdaBoostRegressor 알고리즘 선언
abr.fit(X_train,y_train)
# AdaBoostRegressor 알고리즘에 나의 데이터를 적용시킨다.
abr_y_pred = abr.predict(X_test)
# AdaBoostRegressor 알고리즘을 사용하여서 Y값을 예측한다
#
print('AdaBoostRegressor Mean Absolute Error:', metrics.mean_absolute_error(y_test,abr_y_pred))
print('AdaBoostRegressor Mean Squared Error:', metrics.mean_squared_error(y_test,abr_y_pred))
print('AdaBoostRegressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,abr_y_pred)))
print('AdaBoostRegressor Accuracy:', metrics.r2_score(y_test,abr_y_pred))
abr_df = pd.DataFrame({'Actual':y_test, 'Predicted':abr_y_pred})
print(abr_df)
print("R-squared for Train: %.2f" %abr.score(X_train,y_train))
print("R-squared for Test: %.2f" %abr.score(X_test,y_test))


# ------------------------------------------- Output
# random_state=0일때
# AdaBoostRegressor Mean Absolute Error: 264.095934658306
# AdaBoostRegressor Mean Squared Error: 105842.9302694874
# AdaBoostRegressor Root Mean Squared Error: 325.3351045760162
# AdaBoostRegressor Accuracy: 0.8797212378397619
# R-squared for Train: 0.92
# R-squared for Test: 0.88

# random_state=43일때

# AdaBoostRegressor Mean Absolute Error: 262.2391348191739
# AdaBoostRegressor Mean Squared Error: 131324.9843951394
# AdaBoostRegressor Root Mean Squared Error: 362.38789217513795
# AdaBoostRegressor Accuracy: 0.9021803533758822
# R-squared for Train: 0.92
# R-squared for Test: 0.90

# -----------------------------------DummyRegressor

Dummy = DummyRegressor()
# DummyRegressor 알고리즘 선언

Dummy.fit(X_train,y_train)
# DummyRegressor 알고리즘에 나의 데이터를 적용시켜본다.

Dummy_y_pred = Dummy.predict(X_test)
# Dummy 알고리즘을 사용해서 Y값을 예측한다.

print('DummyRegressor Mean Absolute Error:', metrics.mean_absolute_error(y_test,Dummy_y_pred))
print('DummyRegressor Mean Squared Error:', metrics.mean_squared_error(y_test,Dummy_y_pred))
print('DummyRegressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,Dummy_y_pred)))
print('DummyRegressor Accuracy:', metrics.r2_score(y_test,Dummy_y_pred))
Dummy_df = pd.DataFrame({'Actual':y_test, 'Predicted':Dummy_y_pred})
print(Dummy_df)
print("R-squared for Train: %.2f" %Dummy.score(X_train,y_train))
print("R-squared for Test: %.2f" %Dummy.score(X_test,y_test))

# ------------------------------------------- Output
# random_state=0일 때

# DummyRegressor Mean Absolute Error: 603.3633884521669
# DummyRegressor Mean Squared Error: 880156.477329049
# DummyRegressor Root Mean Squared Error: 938.1665509540665
# DummyRegressor Accuracy: -0.00020030937269321925
# R-squared for Train: 0.00
# R-squared for Test: -0.00

# random_state=43일 때
# DummyRegressor Mean Absolute Error: 606.3160356253877
# DummyRegressor Mean Squared Error: 1342621.3396696597
# DummyRegressor Root Mean Squared Error: 1158.7153833749078
# DummyRegressor Accuracy: -7.432402441365227e-05
# R-squared for Train: 0.00
# R-squared for Test: -0.00

# ---------------------------------BaggingRegressor

bagg = BaggingRegressor()
# BaggingRegressor 알고리즘 선언

bagg.fit(X_train,y_train)
# BaggingRegressor 알고리즘에 나의 데이터를 적용시킨다.

bagg_y_pred = bagg.predict(X_test)
# BaggingRegressor 알고리즘을 사용해서 Y를 예측한다.

print('BaggingRegressor Mean Absolute Error:', metrics.mean_absolute_error(y_test,bagg_y_pred))
print('BaggingRegressor Mean Squared Error:', metrics.mean_squared_error(y_test,bagg_y_pred))
print('BaggingRegressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,bagg_y_pred)))
print('BaggingRegressor Accuracy:', metrics.r2_score(y_test,bagg_y_pred))

bagg_df = pd.DataFrame({'Actual':y_test, 'Predicted':bagg_y_pred})
print(bagg_df)
print("R-squared for Train: %.2f" %bagg.score(X_train,y_train))
print("R-squared for Test: %.2f" %bagg.score(X_test,y_test))

# ------------------------------------------- Output

# random_state=0일때
# BaggingRegressor Mean Absolute Error: 0.7531505914467703
# BaggingRegressor Mean Squared Error: 404.7560202456782
# BaggingRegressor Root Mean Squared Error: 20.11854915856703
# BaggingRegressor Accuracy: 0.9995400396326132
# R-squared for Train: 0.99
# R-squared for Test: 1.00

# random_state=43일때
# BaggingRegressor Mean Absolute Error: 2.7680391264786173
# BaggingRegressor Mean Squared Error: 52422.568587352136
# BaggingRegressor Root Mean Squared Error: 228.95975320425234
# BaggingRegressor Accuracy: 0.960952158814548
# R-squared for Train: 1.00
# R-squared for Test: 0.96

#  --------------------------RandomForestRegressor

RanDF = RandomForestRegressor()
# RandomForestRegressor 알고리즘 선언

RanDF.fit(X_train,y_train)
# RandomForestRegressor 알고리즘에 나의 데이터를 적용시킨다.

RanDF_y_pred = RanDF.predict(X_test)
# BaggingRegressor 알고리즘을 사용해서 Y값을 예측한다.


print('RandomForestRegressor Mean Absolute Error:', metrics.mean_absolute_error(y_test,RanDF_y_pred))
print('RandomForestRegressor Mean Squared Error:', metrics.mean_squared_error(y_test,RanDF_y_pred))
print('RandomForestRegressor Accuracy:', metrics.r2_score(y_test,RanDF_y_pred))
print('RandomForestRegressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,RanDF_y_pred)))
RanDF_df = pd.DataFrame({'Actual':y_test, 'Predicted':RanDF_y_pred})
print(RanDF_df)
print("RandomForestRegressor R-squared for Train: %.2f" %RanDF.score(X_train,y_train))
print("RandomForestRegressor R-squared for Test: %.2f" %RanDF.score(X_test,y_test))


# ------------------------------------------- Output
# random_state=0일때
# RandomForestRegressor Mean Absolute Error: 0.5275022747952686
# RandomForestRegressor Mean Squared Error: 430.61706778889925
# RandomForestRegressor Accuracy: 0.999510651417654
# RandomForestRegressor Root Mean Squared Error: 20.751314844821263
# RandomForestRegressor R-squared for Train: 0.99
# RandomForestRegressor R-squared for Test: 1.00

# random_state=43일때
# RandomForestRegressor Mean Absolute Error: 2.8778434940855315
# RandomForestRegressor Mean Squared Error: 57230.85101455866
# RandomForestRegressor Accuracy: 0.9573706279271497
# RandomForestRegressor Root Mean Squared Error: 239.22970345372806
# RandomForestRegressor R-squared for Train: 1.00
# RandomForestRegressor R-squared for Test: 0.96

# ------------------------ GaussianProcessRegressor

Gaussian = GaussianProcessRegressor()
# GaussianProcessRegressor 알고리즘 선언

Gaussian.fit(X_train,y_train)
# GaussianProcessRegressor 알고리즘에 나의 데이터를 적용시켜 본다.

Gaussian_y_pred = Gaussian.predict(X_test)
# GaussianProcessRegressor 알고리즘을 사용해서 Y값 예측한다.


print('GaussianProcessRegressor Mean Absolute Error:', metrics.mean_absolute_error(y_test,Gaussian_y_pred))
print('GaussianProcessRegressor Mean Squared Error:', metrics.mean_squared_error(y_test,Gaussian_y_pred))
print('GaussianProcessRegressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,Gaussian_y_pred)))
print('GaussianProcessRegressor Accuracy:', metrics.r2_score(y_test,Gaussian_y_pred))

Gaussian_df = pd.DataFrame({'Actual':y_test, 'Predicted':Gaussian_y_pred})
print(Gaussian_df)
print("R-squared for Train: %.2f" %Gaussian.score(X_train,y_train))
print("R-squared for Test: %.2f" %Gaussian.score(X_test,y_test))

# ------------------------------------------- Output
# random_state=0일때
# GaussianProcessRegressor Mean Absolute Error: 465.0475855343884
# GaussianProcessRegressor Mean Squared Error: 1095873.2812232838
# 평균오차제곱
# GaussianProcessRegressor Root Mean Squared Error: 1046.839663569968
# 평균제곱근편차
# GaussianProcessRegressor Accuracy: -0.24533855416145456
# R-squared for Train: 1.00
# R-squared for Test: -0.25

# random_state=43일때
# GaussianProcessRegressor Mean Absolute Error: 469.92498742359925
# GaussianProcessRegressor Mean Squared Error: 1559623.5228000334
# GaussianProcessRegressor Root Mean Squared Error: 1248.848879088272
# GaussianProcessRegressor Accuracy: -0.16171208829481176
# R-squared for Train: 1.00
# R-squared for Test: -0.16

# ------------------------ neural_network.MLPRegressor

MLPRe = MLPRegressor()
# MLPRegressor 알고리즘 선언

MLPRe.fit(X_train,y_train)
# MLPRegressor에 나의 데이터를 적용시킨다.

MLPRe_y_pred = MLPRe.predict(X_test)
# MLPRegressor 알고리즘을 통해 Y값을 예측한다.

print('MLPRe Mean Absolute Error:', metrics.mean_absolute_error(y_test,MLPRe_y_pred))
print('MLPRe Mean Squared Error:', metrics.mean_squared_error(y_test,MLPRe_y_pred))
print('MLPRe Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,MLPRe_y_pred)))
print('MLPRe Accuracy:', metrics.r2_score(y_test,MLPRe_y_pred))

MLPRe_df = pd.DataFrame({'Actual':y_test, 'Predicted':MLPRe_y_pred})
print(MLPRe_df)
print("MLPRe R-squared for Train: %.2f" %MLPRe.score(X_train,y_train))
print("MLPRe R-squared for Test: %.2f" %MLPRe.score(X_test,y_test))

# ------------------------------------------- Output

# MLPRe Mean Absolute Error: 111.23703423333136
# MLPRe Mean Squared Error: 2185990.942572628
# MLPRe Root Mean Squared Error: 1478.5097032392543
# MLPRe Accuracy: -0.6282725066433816
# MLPRe R-squared for Train: 0.78
# MLPRe R-squared for Test: -0.63
