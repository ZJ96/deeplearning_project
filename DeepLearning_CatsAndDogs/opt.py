class OPT(object):
    batch = 8
    train_print_num = 625
    epoch =50
    train_data_root = "./data_local/train"
    test_data_root = "./data_local/test"
    num_workers = 4

    lr = 1e-3
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0  # 损失函数

    log_file_name = "log/log.txt"

    is_use_gpu = True