import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBRegressor
import numpy as np

def main():
  train, train_pp = load_train()
  test, test_pp = load_test()
  sub = load_sub()
  colors = ['red', 'green', 'blue', 'purple', 'orange']

  page = st.sidebar.selectbox("Выберите страницу", ["Соревнование", "Модели"])
    
  if page == "Соревнование":
    st.title("House Prices - Advanced Regression Techniques")
    st.write("Задача данного соревнования является задачей регрессии. Предлагается по имеющимся семидесяти девяти признакам предсказать цену жилья в городе Эймс, штат Айова.")
    st.write("Признаки включают в себя тип зоны (_MSZoning_), наличие коммуникаций (_Utilities_), состояние дома (_OverallCond_), год постройки (_YearBuilt_) и многие другие.")
    st.write("Имеются две таблицы – тренировочная выборка train.csv размером 1460 строк и 81 столбец, тестовая выборка без значений целевого признака test.csv размером 1459 строк и 80 столбцов и примеры значений целевого признака sample_submission.csv размером 1459 строк и 2 столбца.")
    
    data = st.selectbox("Выберите набор данных", ["train", "test", "sub"])
    
    if data == "train":
      df = train
      target = train.SalePrice
      name = "Целевой признак тренировочной выборки"
    elif data == "test":
      df = test
      target = sub.SalePrice
      name = "Целевой признак тестовой выборки"
    else:
      df = sub
      target = sub.SalePrice
      name = "Целевой признак тестовой выборки"
    
    st.write(df)
    
    fig, graph = plt.subplots()
    plt.title(name)
    graph = sns.histplot(x=target, kde=True, color=colors[np.random.randint(0, 4)])
    st.pyplot(fig)
    
  elif page == "Модели":
    st.title("Модели предсказаний")
    st.write("На этой странице можно обратиться к одной из трёх моделей предсказаний, которые показали наилучший результат: _RandomForestRegressor_, _GradientBoostingRegressor_, _XGBRegressor_")
    model = st.selectbox("Выберите модель", ["Случайный лес", "Градиентный бустинг", "Экстремальный градиентный бустинг"])
    
    if model == "Случайный лес":
      clf = load_rf()
    
    if model == "Градиентный бустинг":
      clf = load_gbr()

    if model == "Экстремальный градиентный бустинг":
      clf = load_xgbr()

    st.write("Предсказания, полученные с помощью модели '" + model + "':")
    pred = clf.predict(test_pp.drop('SalePrice', axis=1))
    res = pd.DataFrame({
        "Id": np.arange(1461, 2920, 1),
        "SalePrice": np.expm1(pred)
    })
    st.write(res)

    fig, graph = plt.subplots()
    plt.title(model)
    graph = sns.histplot(x=np.expm1(pred), kde=True, color=colors[np.random.randint(0, 4)])
    st.pyplot(fig)

### data
@st.cache(allow_output_mutation=True)
def load_train():
  df = pd.read_csv("../data/train.csv")
  df_pp = pd.read_csv("../data/train-pp.csv")
  return df, df_pp

@st.cache
def load_test():
  df = pd.read_csv("../data/test.csv")
  df_pp = pd.read_csv("../data/test-pp.csv")
  return df, df_pp

@st.cache
def load_sub():
  df = pd.read_csv("../data/sample_submission.csv")
  return df

### models

#@st.cache
def load_rf():
  with open('../models/model_rf.pkl', 'rb') as pkl_file:
    rf = pickle.load(pkl_file)
  return rf

@st.cache
def load_gbr():
  with open('../models/model_gbr.pkl', 'rb') as pkl_file:
    gbr = pickle.load(pkl_file)
  return gbr

@st.cache(hash_funcs={XGBRegressor: id}) #https://github.com/streamlit/streamlit/issues/1456
def load_xgbr():
  with open('../models/model_xgbr.pkl', 'rb') as pkl_file:
    xgbr = pickle.load(pkl_file)
  return xgbr

if __name__ == "__main__":
  main()