# reinforcement-learning-tutorial
強化学習の基本的な動きの解説用

Q学習 (Q-Learning) と SARSA という強化学習手法を、4*4のグリッドワールドと崖歩きというタスクで性能比較してみる

## 実行環境
|種別|内容|
|---|---|
|OS|macOS Mojave 10.14.6 <br> Windows10 Pro 1903|
|python|3.7|
|package|numpy <br> matplotlib <br> pandas <br> seaborn <br> gym|

## 実行方法
```
python main.py [gridWorld or cliffWalk]
```

## タスク説明
#### 4*4グリッドワールド
最も単純なToyTask。

左上のStartから右下のGoalまで行けばクリアとなる。

Goalに辿り着くと報酬 1 を獲得できる。

<img src="https://user-images.githubusercontent.com/27393111/75039313-b9443180-54fb-11ea-8c26-7ac4698263d6.png">

#### 崖歩き
Suttonの「強化学習」という本に記載されている崖歩きというタスク。

グレーのところ（崖）にいってしまうとStart地点に戻されて報酬 -100 が与えれる。

このタスクに関しては 1 回行動するごとに報酬 -1 が与えれる。

Goalに辿り着くと報酬 13 が与えれる。

つまり、崖に落ちることなくいかに最短ルートを通るということが求められるタスクである。

<img src="https://user-images.githubusercontent.com/27393111/75039364-d8db5a00-54fb-11ea-933f-620a08dd71b5.png">

## 結果
reward 軸は報酬の値で大きければ大きいほどいい

steps 軸はタスククリアまでの行動数の値で一般的に少なければ少ないほどいい
### 4*4グリッドワールド
だいたいどっちも一緒くらい

<img src="https://user-images.githubusercontent.com/27393111/74843608-82411500-536f-11ea-8867-fc4ad76c4f4b.png" width="600">
<img src="https://user-images.githubusercontent.com/27393111/74843705-a43a9780-536f-11ea-8d70-f444adf9f7d8.png" width="600">

### 崖歩き
最終的にQ学習のほうが優秀

<img src="https://user-images.githubusercontent.com/27393111/74843738-b3214a00-536f-11ea-9f0d-d43d6f6ec0d7.png" width="600">
<img src="https://user-images.githubusercontent.com/27393111/74843817-d0561880-536f-11ea-9767-089f1b593872.png" width="600">
