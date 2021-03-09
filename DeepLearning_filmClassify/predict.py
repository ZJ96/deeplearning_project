import config
import jieba
import torch
from model import SentimentModel, pre_weight
from util import build_word_dict, set_seed
from train import *


def predict(comment_str, model, device, word2ix):
    model = model.to(device)
    seg_list = jieba.lcut(comment_str, cut_all=False)
    words_to_idx = []
    for w in seg_list:
        try:
            index = word2ix[w]
        except BaseException:
            index = 0  # 可能出现没有收录的词语，置为0
        words_to_idx.append(index)
    inputs = torch.tensor(words_to_idx).to(device)
    inputs = inputs.reshape(1, len(inputs))
    outputs, _ = model(inputs, [len(inputs), ])
    #print("outputs",outputs)
    pred = outputs.argmax(1).item()
    return pred


def main():
    Config = config.get_args()
    set_seed(Config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    word2ix, ix2word, max_len, avg_len = build_word_dict(Config.train_path)
    weight = torch.zeros(len(word2ix), Config.embedding_dim)
    model = SentimentModel(embedding_dim=Config.embedding_dim,
                           hidden_dim=Config.hidden_dim,
                           LSTM_layers=Config.LSTM_layers,
                           drop_prob=Config.drop_prob,
                           pre_weight=weight)
    model.load_state_dict(
        torch.load(
            Config.model_save_path),
        strict=True)  # 模型加载

    result = predict(Config.comment_str, model, device, word2ix)
    print(Config.comment_str)
    print(result)

if __name__ == '__main__':
    main()