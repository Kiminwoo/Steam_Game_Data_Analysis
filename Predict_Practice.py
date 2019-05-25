import pandas as pd

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn import metrics

# plot
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
DATA_PATH ="C:/Users/user/Desktop/Data/Data.xlsx"
full_dataframe = pd.read_excel(DATA_PATH, sep=',')

X = full_dataframe[["game_positive","game_negative","game_owners","game_initialprice","game_discount"]]
y = full_dataframe["game_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

gbrt = GradientBoostingRegressor()
# GradientBoostingRegressor 선언
gbrt.fit(X_train,y_train)
# 나의 데이터를 GradientBoostingRegressor에 적용시킨다.
gbrt_y_pred = gbrt.predict(X_test)
# GradientBoostingRegressor 알고리즘을 통해 Y를 예측한다.

# print("Feature importances")
# print(gbrt.feature_importances_)
# GradientBoostingRegressor 메소드중 feature 중요도를 나타내는 것이 있었다.

# print('Gradientboostingregressor Mean Absolute Error:', metrics.mean_absolute_error(y,gbrt_y_pred))
# print('Gradientboostingregressor Mean Squared Error:', metrics.mean_squared_error(y,gbrt_y_pred))
# print('Gradientboostingregressor Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y,gbrt_y_pred)))
# print('Gradientboostingregressor Accuracy:', metrics.r2_score(y,gbrt_y_pred))
# print("R-squared for Train: %.2f" %gbrt.score(X,y))
# print("R-squared for Test: %.2f" %gbrt.score(X,y))


# 예측 전의 Y 값과 예측 후의 Y값
gbrt_df = pd.DataFrame({'Actual':y_test, 'Predicted':gbrt_y_pred})

# Gradientboosting regressor 알고리즘을 사용해서 가격예측을 해야한다.
# 예측되어진 값에서 예측전의 값을 빼면 두 값의 편차가 생긴다.
# 이 편차값을 예측되어진 값에서 뺀다면 원래 초기가격에 맞춰지지 않을까 생각을 하였다.
# 이 때 두가지 가정이 주어진다
# 적용 전의 값 - 적용 후의 값 : 양수 or 음수
# 양수일 때는 적용전의 값이 클때이며
# 음수일 때는 적용 후의 값이 더 클때이다.
# 양수일때와 음수일때의 계산을 다르게 해야할 필요를 느꼈다.

# 예측전의 값과 예측후의 값의 차이를 양수와 음수로 비교해서 , 실제 가격에 맞춘다.
# 예측후의 값에서 예측 전의 값을 뺏을 때 양수일 경우
gbrt_df.loc[gbrt_df['Predicted']-gbrt_df['Actual']>0,"Final_Predict"]=gbrt_df['Predicted']-(gbrt_df['Predicted']-gbrt_df['Actual'])
# 예측후의 값에서 예측 전의 값을 뺏을 때 음수일 경우
gbrt_df.loc[gbrt_df['Predicted']-gbrt_df['Actual']<0,"Final_Predict"]=gbrt_df['Predicted']-(gbrt_df['Predicted']-gbrt_df['Actual'])
print(gbrt_df)

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html#sphx-glr-auto-examples-ensemble-plot-voting-regressor-py

plt.figure()
plt.plot(gbrt_df['Actual'],'gd', label='Actual_price')
plt.plot(gbrt_df['Predicted'],'b^', label='Predict_price')
plt.plot(gbrt_df['Final_Predict'],'ys', label='Final_Predict_price')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.ylabel('Prieces')
plt.xlabel('Feature_S')
plt.legend(loc="best")
plt.show()
