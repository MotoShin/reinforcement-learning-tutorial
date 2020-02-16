from abc import ABCMeta, abstractmethod


class BaseLearningMethod(metaclass=ABCMeta):
    """
    学習方法のinterfaceを記述するクラス
    """

    @abstractmethod
    def reset(self) -> None:
        """
        policyをリセットする
        :return: None
        """
        pass

    @abstractmethod
    def update(self,
               current_state: int,
               action: int,
               reward: float,
               next_state: int) -> None:
        """
        Q値の更新を行うメソッド
        :param current_state: 現在状態st int
        :param action: 現在状態での行動at int
        :param reward: 行動atの結果得られた報酬rt+1 int
        :param next_state: 次状態st+1 int
        :return: None
        """
        pass

    @abstractmethod
    def update_behavior_policy(self) -> None:
        """
        挙動方策のパラメータ更新
        :return: None
        """
        pass

    @abstractmethod
    def get_learning_method_name(self) -> str:
        """
        学習手法の名前を返す
        :return: str
        """
        pass
