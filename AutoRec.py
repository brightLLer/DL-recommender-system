import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from data_processing import read_movielens_data, create_user_item_matrix
from nn_model import AutoEncoder
import os, time


def parse_args():
    parser = argparse.ArgumentParser(description="An autoencoder to process recommmender system")

    parser.add_argument('-mo','--mode', default='train', help='train model or predict result', choices=['train', 'test'])
    parser.add_argument('-hs','--hidden_size', type=int, nargs='?', default=200,
                            help='The units of the hidden layer of the autoencoder')
    parser.add_argument('-es','--epochs', type=int, nargs='?', default=50,
                            help='The number of the training epochs')
    parser.add_argument('-bs','--batch_size', type=int, nargs='?', default=247,
                            help='The number of the training examples in one step per epoch')
    parser.add_argument('-lr','--learning_rate', type=float, nargs='?', default=0.007,
                            help='The learning rate used in gradient descent')
    parser.add_argument('-ga','--g_activation', nargs='?', default="sigmoid",
                            help='The activation funcation used in the hidden layer', 
                            choices=["identity", "sigmoid", "relu", "tanh"])
    parser.add_argument('-fa','--f_activation', nargs='?', default="identity",
                            help='The activation funcation used in the output layer',
                            choices=["identity", "sigmoid", "relu", "tanh"])
    parser.add_argument('-op','--optimizer', nargs='?', default="Adam",
                            help='the algorithm used to process graditent descent',
                            choices=["Adam","Adadelta","Adagrad","RMSProp","GradientDescent","Momentum"])
    parser.add_argument('-v','--validation', nargs='?', type=bool, default=True,
                            help='the algorithm used to process graditent descent',
                            choices=["Adam","Adadelta","Adagrad","RMSProp","GradientDescent","Momentum"])
    parser.add_argument('-f','--figure_save_path', default='./pictures/',
                        help='the saving path of the statistic data figure.')
    parser.add_argument('-ms','--model_save_path', default='./models/',
                        help='the saving path of the autoencoder model.when the mode is test,you must specify a model name')
    parser.add_argument('-st','--start_record_id', default=0, type=int,
                        help='the predict result is a table,specify the start row number.using when the mode is test')
    parser.add_argument('-ed','--end_record_id', default=20, type=int,
                        help='the predict result is a table,specify the end row number.using when the mode is test')
    return parser.parse_args()

if __name__ == '__main__':
    acts = {"identity": tf.identity, "sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "tanh": tf.nn.tanh}
    args = parse_args()
    mode = args.mode
    hidden_size = args.hidden_size
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    g_act = args.g_activation
    f_act = args.f_activation
    optimizer = args.optimizer
    validation = args.validation
    figure_save_path = args.figure_save_path
    model_save_path = args.model_save_path
    start = args.start_record_id
    end = args.end_record_id
    
    if g_act not in acts.keys():
        print("Name of g_activation is not valid.")
    if f_act not in acts.keys():
        print("Name of f_activation is not valid.")
    # print(hidden_size, epochs, batch_size, learning_rate, g_act, f_act, optimizer, out)
    print("Start reading data, please wait a few seconds....")
    movielens = read_movielens_data()
    train_pd, test_pd, num_users, num_items = movielens['train_set'], movielens['test_set'], movielens["num_users"], movielens['num_items']
    train_matrix = create_user_item_matrix(train_pd, num_users, num_items)
    test_matrix = create_user_item_matrix(test_pd, num_users, num_items)
    
    sess = tf.Session()
    auto_encoder = AutoEncoder(sess, num_users, hidden_size, epochs, batch_size, learning_rate, acts[f_act], acts[g_act], optimizer)
    if mode == 'train':
        print("Start training....")     
        if validation:
            historys = auto_encoder.train_model(train_matrix.T, test_matrix.T)
        else:
            historys = auto_encoder.train_model(train_matrix.T)
        print("finish training.")

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_save_path + 'model' + str(time.time()) + '.ckpt')
        print("model has been saved in " + save_path)
        sess.close()

        plt.figure(figsize=(10, 3))
        plt.subplot(121)
        plt.plot(historys['losses'], ls=':', lw=2, color='red', marker='^', label='loss')
        plt.legend(loc='best')
        plt.subplot(122)
        plt.plot(historys['train_rmses'], ls='-', lw=2, marker='*', label='train rmse')
        plt.plot(historys['test_rmses'], ls='--', lw=2, color='green', marker='+', label='test rmse')
        plt.legend(loc='best')
        plt.savefig(figure_save_path + 'statistics' + str(time.time()) + '.jpg')
        print("statistics data figure has been saved in " + figure_save_path + 'statistics' + str(time.time()) + '.jpg')
        plt.show()
    else:
        auto_encoder.predict(test_matrix.T, start, end, model_save_path)
        sess.close()