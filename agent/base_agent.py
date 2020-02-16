from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):
    """
    Agentのinterfaceを記述するクラス
    """

    @abstractmethod
    def reset(self) -> None:
        """
        推定方策のresetを行うメソッド
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
    def act(self) -> int:
        """
        エージェントが行動を行うメソッド
        :return: エージェントが行う行動 int
        """
        pass

    @abstractmethod
    def update_policy(self, reward: int, next_state: int) -> None:
        """
        方策のアップデートを行い、エージェントの現在位置の更新を行うメソッド
        :param reward: エージェントの行動した結果得られた報酬 int
        :param next_state: エージェントが行動した結果である環境から渡される次状態 int
        :return: None
        """
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        エージェントのoutput用の文字列を返すメソッド
        :return:
        """
        pass

    @abstractmethod
    def set_start_state(self, start_state: int) -> None:
        """
        エージェントの初期状態のsetter
        :param start_state: int
        :return: None
        """
        pass
