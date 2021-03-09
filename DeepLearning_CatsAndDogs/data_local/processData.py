import os
import shutil


def Process_data(local_data):
    if not os.path.exists("./data_local/train"):
        os.mkdir("./data_local/train")
    if not os.path.exists("./data_local/test"):
        os.mkdir("./data_local/test")
    for file in os.listdir(local_data):
        if int(file.split(".")[-2]) > 10000:
            shutil.copy(os.path.join(local_data, file), "./data_local/test")
        else:
            shutil.copy(os.path.join(local_data, file), "./data_local/train")


if __name__ == '__main__':
    data_local = "data_local/dogs-vs-cats-redux-kernels-edition/train"
    Process_data(data_local)

    print(len(os.listdir("./data_local/train")))
    print(len(os.listdir("./data_local/test")))


