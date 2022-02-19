import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

def ensemble(preds, weight=None):
    '''
    アンサンブルを行う関数
    加重平均可能
    Input: 
        preds: np.array or list
            全モデルの予測値
        w: np.array or list
            加重平均の重み
            Noneの時は単純平均
    Out:
        pred: np.array
            アンサンブルした予測値
    '''
    if weight == None:
        weight = np.ones(len(preds))
    preds_np = np.array(preds)
    weight_np = np.array(weight)
    return np.average(preds_np, weights=weight_np, axis=0)

def predict_proba_average(models, test_feature_df, d, weight=None):
    '''
    複数のモデルの予測値の平均を求める関数
    加重平均可能
    Input:
        models:
            学習済みのモデル
        test_feature_df: pd.DataFrame
            予測したいデータの特徴量
        d: int
            予測したいクラスを指定
        weight: np.array
            加重平均の重み
            Noneの時は単純平均
    Output:
        pred_prob: np.array
            複数のモデルの予測値の平均
    '''
    if weight == None:
        weight = np.ones(len(models))
    # k 個のモデルの予測確率 (preds) を作成. shape = (k, N_test, n_classes).
    preds = np.array([model.predict_proba(test_feature_df) for model in models])
    # k 個のモデルの平均を計算
    weight_np = np.array(weight)
    preds = np.average(preds, weights=weight_np, axis=0)
    # ほしいのは y=d の確率なので全要素の d 次元目を取ってくる
    preds = preds[:, d]

    return preds

def show_preds(preds, labels=None, figsize=(10, 6)):
    '''
    予測値の分布を表示する関数
    Input:
        preds: np.array or list
            予測値. 複数可
        labels: np.array str
            各予測値を予測したモデル名など. 複数可
            指定しない場合、連番で名付ける
        figsize: tuple
            描画サイズ
    '''
    if labels == None:
        labels = np.array([f'model{i+1}' for i in range(len(preds))] , dtype=np.str)
    fig, ax = plt.subplots(figsize=figsize)
    for pred, label in zip(preds, labels):
        sns.distplot(pred, ax=ax, label=label)
        ax.legend()
        ax.grid()
    plt.show()

import japanize_matplotlib

def visualize_importance(models, feat_train_df):
    """
    model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現する
    Input:
        models:
            List of models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importances_
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby("column")\
        .sum()[["feature_importance"]]\
        .sort_values("feature_importance", ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x="feature_importance", 
                  y="column", 
                  order=order, 
                  ax=ax, 
                  palette="viridis", 
                  orient="h")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()
    plt.show()

def min_max_df(df):
    '''
    データフレームを列ごとに正規化
    正規化: 最小値0, 最大値1
    Input:
        df: pd.DataFrame
            正規化したいデータフレーム
    Output:
        df_mm: pd.DataFrame
            正規化されたデータフレーム
    '''
    mm = preprocessing.MinMaxScaler()
    df_mm = mm.fit_transform(df)

    return df_mm

def std_df(df):
    '''
    データフレームを列ごとに標準化
    標準化: 平均0, 分散1
    Input:
        df: pd.DataFrame
            標準化したいデータフレーム
    Output:
        df_ss: pd.DataFrame
            標準化されたデータフレーム
    '''
    ss = preprocessing.StandardScaler()
    df_ss = ss.fit_transform(df)

    return df_ss