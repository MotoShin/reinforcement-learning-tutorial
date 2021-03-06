import configparser
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.switch_backend('agg')


def df_to_numpy(data) -> dict:
    """
    pandasのデータからnumpyのデータに変換する
    """
    columns = list(data.columns)
    # np.arrayに変換
    numpy_data = np.array(data.values.flatten())
    # numpy_dataの形を、pandaで読み込んだデータフレームの形に変形する
    numpy_array = np.reshape(numpy_data, (data.shape[0], data.shape[1]))
    numpy_array = np.rot90(numpy_array, k=-1)
    numpy_array = np.flip(numpy_array, axis=1)

    data_dict = {}
    for i in range(len(columns)):
        if columns[i] == "Unnamed: 0":
            numpy_array[i] = numpy_array[i] + 1
            data_dict.update({"times": numpy_array[i]})
        else:
            data_dict.update({columns[i]: numpy_array[i]})

    return data_dict


class Util(object):
    """
    便利な関数書くクラス
    """

    @staticmethod
    def display_progress(simulations_number: int, display_simulation_number: int, algorithm_name: str):
        """
        実行しているコンソール画面に現在の進捗内容を表示する
        :param simulations_number: シミュレーションの総回数
        :param display_simulation_number: 現在のシミュレーションの回数
        :param algorithm_name: 現在実行しているアルゴリズムの名前
        :return: void
        """
        display_overall = 100
        reduction_coefficient = display_overall / simulations_number
        i = int(display_simulation_number * reduction_coefficient)
        pro_bar = ('=' * i) + (' ' * (display_overall - i))
        print('\r{0}: [{1}] {2}%'.format(algorithm_name, pro_bar, i / display_overall * 100.), end='')

    @staticmethod
    def make_config_parser():
        """
        configファイルの読むparserを作成する
        :return: void
        """
        parser = configparser.ConfigParser()
        file_path = os.path.abspath("./") + "/config/config.ini"
        if os.path.isfile(file_path):
            parser.read(file_path)
            return parser
        else:
            print("config file open failed.")
            return None

    @staticmethod
    def output_csv(name_and_data: dict, file_name: str):
        """
        結果をcsvとして出力する
        :param name_and_data: アルゴリズム名と成績が格納されているdict
        :param file_name: 出力するファイルの名前
        :return: void
        """
        split_str = file_name.split("/")
        if len(split_str) is not 1 and not os.path.isdir(split_str[len(split_str) - 2]):
            os.makedirs(os.path.abspath("./") + "/output/" + split_str[len(split_str) - 2], exist_ok=True)

        # windowsの方で空行が挿入されていたため、newlineを設定
        with open(os.path.abspath("./") + "/output/" + file_name, 'w', newline="") as f:
            writer = csv.writer(f)

            if len(name_and_data.keys()) == 1:
                for name in name_and_data.keys():
                    writer.writerow([name])
                    for one_data in name_and_data.get(name):
                        writer.writerow([one_data])
            else:
                keys = name_and_data.keys()
                all_data = None
                for key in keys:
                    data = name_and_data.get(key)
                    if all_data is None:
                        all_data = data
                    else:
                        all_data = np.append([all_data], [data], axis=0)

                all_data = all_data.T
                writer.writerow(keys)
                for line in all_data:
                    writer.writerow(line)

    @staticmethod
    def output_image(file_name: str) -> None:
        """
        画像を作るメソッド
        """
        sns.set()
        data = pd.read_csv("output/result/{}.csv".format(file_name))
        name_and_data = df_to_numpy(data)
        fig = plt.figure(figsize=(8, 5.5))
        ax = fig.add_subplot(111)

        for name in name_and_data.keys():
            ax.plot(name_and_data.get(name), label=name)
        ax.set_xlabel("Episode")
        ax.set_ylabel(file_name)
        plt.legend(frameon=False)

        fig.savefig("output/images/{}.png".format(file_name))
        plt.close()
