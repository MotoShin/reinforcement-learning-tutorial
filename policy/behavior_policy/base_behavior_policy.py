from abc import ABCMeta, abstractmethod


class BaseBehaviorPolicy(metaclass=ABCMeta):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def choose(self, ary) -> int:
        pass
