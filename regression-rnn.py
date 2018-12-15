import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from rnn_model import RNNConfig, RNN
from data_loader import process_file, batch_iter

base_dir = 'data/'
f = 0.5
data_dir = str(f)+'.csv'
save_dir = 'checkpoints/rnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
f = 1.0
# v = 2.0
# val_dir = 'data/macro/val'
# vocab_dir = os.path.join(base_dir, 'vocab.txt')
#
# save_dir = 'checkpoints/textrnn'
# save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/rnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    # 读取数据
    print("Loading training  data: %.1f.csv" % f)
    tarin_dir = os.path.join(base_dir, data_dir)
    x_train, y_train = process_file(tarin_dir, config)
    x_normlize = x_train
    y_normlize = y_train

    # 开始训练
    print('Training and evaluating...')
    for epoch in range(config.epoches):
        x_batch, y_batch = batch_iter(x_normlize, y_normlize, config)
        for iteration in range(config.num_iterations):
            # print('Epoch: ', epoch + 1)
            # batch_train = batch_iter(x_train, y_train, config.batch_size)
            feed_dict = {
                model.xs: x_batch,
                model.ys: y_batch,
                # create initial state
            }
            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)
            # plotting
            # if v % 20 == 0:
            #     plt.figure(v)
            #     plt.plot(t[0, :], (v/10)*y_batch[0].flatten(), 'r', t[0, :], (v/10)*pred.flatten()[:config.time_steps], 'b--',
            #              t[0, :], (v/10) *x_batch[0].flatten(), 'k-.')
            #     plt.ylim((-16, 16))
            #     plt.draw()
            #     plt.pause(0.3)
            if iteration % 20 == 0:
                print('cost: ', round(cost, 4))
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, iteration)

    # 保存网络
    saver.save(sess=sess, save_path=save_path)



def test():
    # v = random.randint(start, end)
    # 读取数据
    print("Loading test  data: ")

    # tarin_data = os.path.join(base_dir, dir)
    tarin_dir = os.path.join(base_dir, data_dir)
    x_data, y_data = process_file(tarin_dir, config)
    # x_max =  max(abs(x_test))
    # y_max = max(y_train)
    # x_max = max(x_test)
    x_normlize = x_data
    y_normlize = y_data

    # 创建session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    x_batch, y_batch = batch_iter(x_normlize, y_normlize, config)
    x_test = x_batch[1].reshape(1, 1000, 1)
    y_test = y_batch[1].reshape(1, 1000, 1)
    feed_dict = {
            model.xs: x_test,
            model.ys: y_test,
            # create initial state
        }


    state, pred = sess.run(
        [model.cell_final_state, model.pred],
        feed_dict=feed_dict)

    # plotting
    # plt.figure(v)
    plt.plot(x_test.flatten(), 'r',  y_test.flatten(), 'b--', state.flatten(), 'k-.')
    plt.ylim((-16, 16))
    plt.draw()
    plt.pause(0.3)
    os.system("pause")



if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_rnn.py [train / test]""")

    print('Config RNN model...')
    config = RNNConfig()
    model = RNN(config)
    # train()
    if sys.argv[1] == 'train':
        train()
    else:
        test()
