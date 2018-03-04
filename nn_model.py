import tensorflow as tf
import time
import numpy as np

class AutoEncoder(object):
    def __init__(self, sess, input_size, hidden_size, epochs, batch_size, start_lr, f_act, g_act, optimizer):
        self.sess = sess
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.start_lr = start_lr
        self.f_act = f_act
        self.g_act = g_act
        self.optimizer = optimizer
        self.R_pl = tf.placeholder(tf.float32, [None, self.input_size], name="R")
        self.keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob")
    def compute_rmse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def init_weight(self, shape, name, stddev=0.01):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
    
    def init_bias(self, shape, name):
        return tf.Variable(tf.zeros(shape), name=name)
    
    def inference(self):
        with tf.name_scope("Encoder_layer"):
            V = self.init_weight([self.input_size, self.hidden_size],"V")
            mu = self.init_bias([self.hidden_size], "mu")
            latent = self.g_act(tf.matmul(self.R_pl, V) + mu)
            latent_dropout = tf.nn.dropout(latent, self.keep_prob_pl)
        with tf.name_scope("Decoder_layer"):
            W = self.init_weight([self.hidden_size, self.input_size], "W")
            b = self.init_bias([self.input_size], "b")
            output = self.f_act(tf.matmul(latent_dropout, W) + b)
        with tf.name_scope("Loss"):
            R_nonzero_mask = tf.not_equal(self.R_pl, 0)
            R_observed = tf.boolean_mask(self.R_pl, R_nonzero_mask)
            output_observed = tf.boolean_mask(output, R_nonzero_mask)
            loss = tf.reduce_mean(tf.square(output_observed - R_observed))
        return loss, output, R_observed, output_observed
    
    def train_model(self, R_train, R_test=None, verbose=True, use_dropout=False):
        cost_historys = []
        train_rmse_historys = []
        test_rmse_historys = []
        loss, output, R_observed, output_observed = self.inference()
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.start_lr, global_step, 10, 0.96, staircase=False)
        if self.optimizer == "Adam":
            optim = tf.train.AdamOptimizer(lr)
        elif self.optimizer == "Adadelta":
            optim = tf.train.AdadeltaOptimizer(lr)
        elif self.optimizer == "Adagrad":
            optim = tf.train.AdadeltaOptimizer(lr)
        elif self.optimizer == "RMSProp":
            optim = tf.train.RMSPropOptimizer(lr)
        elif self.optimizer == "GradientDescent":
            optim = tf.train.GradientDescentOptimizer(lr)
        elif self.optimizer == "Momentum":
            optim = tf.train.MomentumOptimizer(lr, 0.9)
        train_op = optim.minimize(loss, global_step)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            t1 = time.time()
            R_shuffled = R_train[np.random.permutation(R_train.shape[0]), :]
            n_batch = R_train.shape[0] // self.batch_size \
                if R_train.shape[0] % self.batch_size == 0 else R_train.shape[0] // self.batch_size + 1
            cost = 0
            for k in range(n_batch - 1):
                R_batch = R_shuffled[k * self.batch_size: (k+1) * self.batch_size, :]
                if use_dropout:
                    keep_prob_feed = 0.5
                else:
                    keep_prob_feed = 1.0
                _, loss_per_batch = self.sess.run([train_op, loss], feed_dict={self.R_pl: R_batch, self.keep_prob_pl: keep_prob_feed})
                cost += loss_per_batch
            R_batch = R_shuffled[k * self.batch_size: , :]
            _, loss_per_batch = self.sess.run([train_op, loss], feed_dict={self.R_pl: R_batch, self.keep_prob_pl: keep_prob_feed})
            cost += loss_per_batch
            t = time.time() - t1

            cost /= n_batch
            cost_historys.append(cost)
            
            train_output, train_R = self.sess.run([output_observed, R_observed], feed_dict={self.R_pl: R_train, self.keep_prob_pl: 1.0})
            train_output_clip = train_output.clip(min=1, max=5)
            train_rmse = self.compute_rmse(train_R, train_output_clip)
            train_rmse_historys.append(train_rmse)
            
            if R_test is not None:
                test_output, test_R = self.sess.run([output_observed, R_observed], feed_dict={self.R_pl: R_test, self.keep_prob_pl: 1.0})
                test_output_clip = test_output.clip(min=1, max=5)
                test_rmse = self.compute_rmse(test_R, test_output_clip)
                test_rmse_historys.append(test_rmse)
            if verbose:
                if R_test is not None:
                    print("Epoch %3d, Loss %.4f, Train RMSE %.4f, Test RMSE %.4f, Time %.4f" % (epoch + 1, cost, train_rmse, test_rmse, t))
                else:
                    print("Epoch %3d, Loss %.4f, Train RMSE %.4f, Time %.4f" % (epoch + 1, cost, train_rmse, t))
        return dict([("losses", cost_historys), ("train_rmses", train_rmse_historys), ("test_rmses", test_rmse_historys)])
    
    def predict(self, R_test, start, end, model_save_path=None):
        _, output, _, _ = self.inference()
        if model_save_path:
            saver = tf.train.Saver()
            saver.restore(self.sess, model_save_path)
        out = self.sess.run(output, feed_dict={self.R_pl: R_test, self.keep_prob_pl: 1.0})
        result = [(loc[0]+1, loc[1]+1, R_test[loc[0], loc[1]], out[loc[0], loc[1]]) for loc in np.argwhere(R_test != 0)]
        if (end - start + 1) > len(result):
            print('out of the number of total results:' + str(len(result)))
        else:
            if start == end:
                line = result[start]
                print("User %4d, Item %4d, True rating %.1f, Predict rating %.3f" % (line[0], line[1], line[2], line[3]))
            else:
                for line in result[start: end]:
                    print("User %4d, Item %4d, True rating %.1f, Predict rating %.3f" % (line[0], line[1], line[2], line[3]))
        return out