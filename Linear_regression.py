import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Excel read
pd.set_option('display.max_columns', None)
DATA_PATH ="C:/Users/user/Desktop/Data/Data.xlsx"
full_dataframe = pd.read_excel(DATA_PATH, sep=',')

X = full_dataframe[["game_positive","game_negative","game_owners","game_initialprice","game_discount"]]
y = full_dataframe["game_price"]

# X = sm.add_constant()

# Note the difference in argument order
model = sm.OLS(y,X).fit()

# make the predictions by the model
predictions = model.predict(X)

#Print out the statistics
model.summary()
# OLS : Ordinary Least Squares
# Least Squears : 회귀 직선과의 거리의 제곱을 최소화하는 회귀직선에 적합하도록 하는 메소드
# DF : degrees of freedom ( 최종 결과값에서의 변화가 자유로운 통계 수 )
print(model.summary())
# --> model.summary를 통해 feature가 price에 미치는 영향을 알아본다.

im = linear_model.LinearRegression()
model = im.fit(X,y)
prediction_linear = im.predict(X)

# 예측값과 실제값을 비교
plt.scatter(prediction_linear,y,s=10)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("coef :"+str(im.coef_))
print("score :"+str(im.score(X,y)))
print("intercept :"+str(im.intercept_))
print("Actual price  : "+ str(y))
print("predict price :"+str(prediction_linear))

# 실질적으로 예측된 가격을 예측전의 가격과 비교해 보고싶어서 합쳐준다.
# 일단 딕셔너리 형태로 저장
Result = {
        'Actual price' : y,
        'predict price' : prediction_linear
}

# 저장되어진 딕셔너리형태를 Dataframe형태로 저장
Result_df = pd.DataFrame(Result)
print(Result_df)
