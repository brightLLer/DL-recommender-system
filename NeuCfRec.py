import keras
from keras.layers import Embedding, Dense, Concatenate, Multiply, Input, Reshape, Flatten, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from data_processing import read_movielens_data
import numpy as np
import argparse
import os, time, sys

def parse_args():
    parser = argparse.ArgumentParser(description="A neural network combined with collaborative fitering to process recommender system")
    parser.add_argument('-m','--mode', nargs='?', default='neucf',
                            help='set the type of the neural network', choices=['neucf', 'neumf', 'cf_pred', 'mf_pred'])
    parser.add_argument('-h1','--hidden_size1', nargs='?', default=32, type=int, help='the units of the first hidden layer')
    parser.add_argument('-h2','--hidden_size2', nargs='?', default=16, type=int, help='the units of the second hidden layer')
    parser.add_argument('-h3','--hidden_size3', nargs='?', default=8, type=int, help='the units of the third hidden layer')
    parser.add_argument('-ed','--embed_dims', nargs='?', default=16, type=int, help='the dimesions of the embeddings used in neucf')
    parser.add_argument('-med','--mlp_embed_dims', nargs='?', default=16, type=int, 
                        help='the dimesions of the embeddings used in mlp of the neumf')
    parser.add_argument('-ged','--gmf_embed_dims', nargs='?', default=16, type=int, 
                        help='the dimesions of the embeddings used in gmf of the neumf')
    parser.add_argument('-a','--activation', nargs='?', default='relu', 
                        help='the activations used in all hidden layers', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-lr','--learning_rate', nargs='?', default=0.005, type=float, 
                        help='the learning rate used in gradient descent')
    parser.add_argument('-dr','--decay_rate', nargs='?', default=1e-4, type=float,
                        help='the decay rate used in learning rate decay')
    parser.add_argument('-es','--epochs', type=int, nargs='?', default=10,
                            help='The number of the training epochs')
    parser.add_argument('-bs','--batch_size', type=int, nargs='?', default=256,
                            help='The number of the training examples in one step per epoch')
    parser.add_argument('-ms','--model_save_path', nargs='?', default='./keras_models/',
                            help='The path to save the model,must contains weight filename when predict')
    parser.add_argument('-te','--tensorboard_event_path', nargs='?', default='./keras_logs/',
                            help='The path to save the tensorboard event')
    parser.add_argument('-sp','--statistics_save_path', nargs='?', default='./keras_historys/',
                            help='The path to save the statistics history data')
    parser.add_argument('-wl','--weight_load_path', nargs='?',
                            help='The path to load the weight of the model')
    parser.add_argument('-st','--start_record_id', default=0, type=int,
                        help='the predict result is a table,specify the start row number.using when the mode is cf_pred or mf_pred')
    parser.add_argument('-en','--end_record_id', default=20, type=int,
                        help='the predict result is a table,specify the end row number.using when the mode is cf_pred or mf_pred')
    return parser.parse_args()


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1, 5)
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def processing_data():
    movielens = read_movielens_data()
    train_set, test_set = movielens['train_set'], movielens['test_set']
    user_ids = train_set['user_id'].to_dense()[:, np.newaxis]
    item_ids = train_set['movie_id'].to_dense()[:, np.newaxis]
    y = train_set['rating'].to_dense()[:, np.newaxis]
    user_ids_test = test_set['user_id'].to_dense()[:, np.newaxis]
    item_ids_test = test_set['movie_id'].to_dense()[:, np.newaxis]
    y_test = test_set['rating'].to_dense()[:, np.newaxis]
    num_users, num_items = movielens['num_users'], movielens['num_items']
    return num_users, num_items, user_ids, item_ids, y, user_ids_test, item_ids_test, y_test


def create_neucf_model(num_users, num_items, embed_dims, mlp_hidden_layers, activation):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    user_embeddings = Embedding(num_users, embed_dims, name='user_embeddings')(user_input)
    item_embeddings = Embedding(num_items, embed_dims, name='item_embeddings')(item_input)
    user_embeddings = Reshape((-1,))(user_embeddings)
    item_embeddings = Reshape((-1,))(item_embeddings)
    input_con = Concatenate()([user_embeddings, item_embeddings])
    cf_layer = Dense(mlp_hidden_layers[0], activation=activation, name='CF1', input_shape=(embed_dims + embed_dims,))(input_con)
    cf_layer = Dense(mlp_hidden_layers[1], activation=activation, name='CF2')(cf_layer)
    cf_layer = Dense(mlp_hidden_layers[2], activation=activation, name='CF3')(cf_layer)
    outputs = Dense(1, name='output')(cf_layer)
    model = Model(inputs=[user_input, item_input], outputs=outputs)
    return model
    
def create_neumf_model(num_users, num_items, gmf_embed_dims, mlp_embed_dims, mlp_hidden_layers, activation):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    # GMF
    user_embedding_gmf = Embedding(num_users, gmf_embed_dims, name='user_embedding_gmf')(user_input)
    item_embedding_gmf = Embedding(num_items, gmf_embed_dims, name='item_embedding_gmf')(item_input)
    user_embedding_gmf = Flatten()(user_embedding_gmf)
    item_embedding_gmf = Flatten()(item_embedding_gmf)
    output_gmf = Multiply(name='element_wise_product')([user_embedding_gmf, item_embedding_gmf])
    # MLP
    user_embedding_mlp = Embedding(num_users, mlp_embed_dims, name='user_embedding_mlp')(user_input)
    item_embedding_mlp = Embedding(num_items, mlp_embed_dims, name='item_embedding_mlp')(item_input)
    user_embedding_mlp = Flatten()(user_embedding_mlp)
    item_embedding_mlp = Flatten()(item_embedding_mlp)
    input_mlp = Concatenate(axis=-1, name="concate_embeddings")([user_embedding_mlp, item_embedding_mlp])
    cf1 = Dense(mlp_hidden_layers[0], activation=activation, name='CF1')(input_mlp)
    cf2 = Dense(mlp_hidden_layers[1], activation=activation, name='CF2')(cf1)
    output_mlp = Dense(mlp_hidden_layers[2], activation=activation, name='CF3')(cf2)
    
    output_gmf_mlp = Concatenate(axis=-1, name="concate_models")([output_gmf, output_mlp])
    
    outputs = Dense(1, use_bias=False, name='output')(output_gmf_mlp)
    
    model = Model(inputs=[user_input, item_input], outputs=outputs)
    return model

if __name__ == '__main__':
    args = parse_args()
    mode = args.mode
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    hidden_size3 = args.hidden_size3
    embed_dims = args.embed_dims
    mlp_embed_dims = args.mlp_embed_dims
    gmf_embed_dims = args.gmf_embed_dims
    activation = args.activation
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    epochs = args.epochs
    batch_size = args.batch_size
    model_save_path = args.model_save_path
    tensorboard_event_path = args.tensorboard_event_path
    statistics_save_path = args.statistics_save_path
    weight_load_path = args.weight_load_path
    start = args.start_record_id
    end = args.end_record_id
    
    if activation not in ['sigmoid', 'relu', 'tanh']:
        print("Name of activation is not valid.")
    
    print('Start reading data, please wait a few seconds....')    
    num_users, num_items, user_ids, item_ids, y, user_ids_test, item_ids_test, y_test = processing_data()
    mlp_hidden_layers = [hidden_size1, hidden_size2, hidden_size3]
      
    if mode == 'cf_pred':
        model = create_neucf_model(num_users, num_items, embed_dims, mlp_hidden_layers, activation)
        model.load_weights(weight_load_path)
        y_pred = model.predict([user_ids_test, item_ids_test])
        for i in range(start, end + 1):
            print("User %4d, Item %4d, True rating %.1f, Predict rating %.3f" % 
                  (user_ids_test[i][0], item_ids_test[i][0], y_test[i][0], y_pred[i][0]))
        sys.exit(0)
    elif mode == 'mf_pred':
        model = create_neumf_model(num_users, num_items, gmf_embed_dims, mlp_embed_dims, mlp_hidden_layers, activation)
        model.load_weights(weight_load_path)
        y_pred = model.predict([user_ids_test, item_ids_test])
        for i in range(start, end + 1):
            print("User %4d, Item %4d, True rating %.1f, Predict rating %.3f" % 
                  (user_ids_test[i][0], item_ids_test[i][0], y_test[i][0], y_pred[i][0]))
        sys.exit(0)
    else:
        pass
    
    model_save_path = os.path.join(model_save_path, 'weights('+ mode +')' + str(time.time()))
    tensorboard_event_path = os.path.join(tensorboard_event_path, 'events('+ mode +')' + str(time.time()))
    statistics_save_path = os.path.join(statistics_save_path, 'historys(' + mode + ')' + str(time.time()))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(tensorboard_event_path):
        os.makedirs(tensorboard_event_path)
    if not os.path.exists(statistics_save_path):
        os.makedirs(statistics_save_path)
  
    print('Start training....')
    if mode == 'neucf':
        model = create_neucf_model(num_users, num_items, embed_dims, mlp_hidden_layers, activation)
    elif mode == 'neumf':
        model = create_neumf_model(num_users, num_items, gmf_embed_dims, mlp_embed_dims, mlp_hidden_layers, activation)
    else:
        pass
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=decay_rate), loss='mse', metrics=[rmse])
    
    checkpoint_path = os.path.join(model_save_path, 'weights-improvement.epoch_{epoch:02d}-val_loss_{val_loss:.4f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')
    tensorboard = TensorBoard(tensorboard_event_path, batch_size=batch_size, histogram_freq=2, write_grads=True)
    
    statistics = model.fit([user_ids, item_ids], y, batch_size=batch_size, epochs=epochs, 
                     validation_data=([user_ids_test, item_ids_test], y_test), callbacks=[checkpoint, tensorboard])
    history = statistics.history
    np.save(os.path.join(statistics_save_path, 'val_rmse.npy'), history['val_rmse'])
    np.save(os.path.join(statistics_save_path, 'rmse.npy'), history['rmse'])
    np.save(os.path.join(statistics_save_path, 'val_loss.npy'), history['val_loss'])
    np.save(os.path.join(statistics_save_path, 'loss.npy'), history['loss'])
    print('finish training.save model to %s, save event to %s, save statistics history data to %s' % 
          (model_save_path, tensorboard_event_path, statistics_save_path))
        
    
    
    
    
    
    