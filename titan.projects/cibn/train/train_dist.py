from __future__ import division
import os
import sys
import time
import logging
import tensorflow as tf
import numpy as np
import yaml
from collections import defaultdict
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

def get_conf(file_path):
    conf = {}
    with open(file_path, 'r') as f:
        _conf = yaml.load(f)
        conf.update(_conf['data'])
        conf.update(_conf['model'])
        conf.update(_conf['train'])
        conf.update(_conf['info'])
    return conf

def create_hparams(conf):
    """Create hparams."""
    config = conf.copy() 
    hparams = tf.contrib.training.HParams(
        # data
        train_file = config['train_file'] if 'train_file' in config else None,
        eval_file = config['eval_file'] if 'eval_file' in config else None,
        infer_file = config['infer_file'] if 'infer_file' in config else None,
        test_file = config['test_file'] if 'test_file' in config else None,
        feature_count=config['feature_count'] if 'feature_count' in config else None,
        field_count=config['field_count'] if 'field_count' in config else None,
        data_format=config['data_format'] if 'data_format' in config else None,
        infer_file_has_label=config['infer_file_has_label'] if 'infer_file_has_label' in config else True,
        PAIR_NUM=config['PAIR_NUM'] if 'PAIR_NUM' in config else None,
        DNN_FIELD_NUM=config['DNN_FIELD_NUM'] if 'DNN_FIELD_NUM' in config else None,
        n_user=config['n_user'] if 'n_user' in config else None,
        n_item=config['n_item'] if 'n_item' in config else None,
        n_user_attr=config['n_user_attr'] if 'n_user_attr' in config else None,
        n_item_attr=config['n_item_attr'] if 'n_item_attr' in config else None,

        # model
        dim=config['dim'] if 'dim' in config else None,
        layer_sizes=config['layer_sizes'] if 'layer_sizes' in config else None,
        activation=config['activation'] if 'activation' in config else None,
        dropout=config['dropout'] if 'dropout' in config else None,
        attention_layer_sizes=config['attention_layer_sizes'] if 'attention_layer_sizes' in config else None,
        attention_activation=config['attention_activation'] if 'attention_activation' in config else None,
        model_type=config['model_type'] if 'model_type' in config else None,
        method=config['method'] if 'method' in config else None,
        load_model_name=config['load_model_name'] if 'load_model_name' in config else None,
        load_model=config['load_model'] if 'load_model' in config else False,

        mu=config['mu'] if 'mu' in config else None,

        # train
        init_method=config['init_method'] if 'init_method' in config else 'tnormal',
        init_value=config['init_value'] if 'init_value' in config else 0.01,
        embed_l2=config['embed_l2'] if 'embed_l2' in config else 0.0000,
        embed_l1=config['embed_l1'] if 'embed_l1' in config else 0.0000,
        layer_l2=config['layer_l2'] if 'layer_l2' in config else 0.0000,
        layer_l1=config['layer_l1'] if 'layer_l1' in config else 0.0000,
        learning_rate=config['learning_rate'] if 'learning_rate' in config else 0.001,
        loss=config['loss'] if 'loss' in config else None,
        optimizer=config['optimizer'] if 'optimizer' in config else 'adam',
        epochs=config['epochs'] if 'epochs' in config else 10,
        batch_size=config['batch_size'] if 'batch_size' in config else 1,
        auto_stop_auc=config['auto_stop_auc'] if 'auto_stop_auc' in config else False,

        # show info
        show_step=config['show_step'] if 'show_step' in config else 1,
        save_epoch=config['save_epoch'] if 'save_epoch' in config else 5,
        metrics=config['metrics'] if 'metrics' in config else None,
        
        # save model
        model_dir=config['model_dir'] if 'model_dir' in config else None,
        cache_path=config['cache_path'] if 'cache_path' in config else None,
        res_dir=config['res_dir'] if 'res_dir' in config else None,

        #other flag
        cache_data_in_memory=config['cache_data_in_memory'] if 'cache_data_in_memory' in config else True
    )

    logger.info('create hparams: {0}'.format(hparams))
    return hparams

def extract_libffm_features(input_lines, has_label=True):
    """extract ffm features from lines"""
    labels = []
    features = []
    impression_ids = []

    start_index = 1 if has_label else 0

    for _ in input_lines:
        line = _.strip()
        if not line:
            continue
        tmp = line.strip().split('%')
        if len(tmp) == 2:
            impression_ids.append(tmp[1].strip())
        else:
            impression_ids.append('none')

        line = tmp[0]
        cols = line.strip().split(' ')
        label = float(cols[0].strip()) if has_label else 0
        #if label > 0:
        #    label = 1
        #else:
        #    label = 0
        cur_feature_list = []

        for word in cols[start_index:]:
            if not word.strip():
                continue
            tokens = word.strip().split(':')
            cur_feature_list.append( \
                [int(tokens[0]) -1, \
                    int(tokens[1]) -1, \
                    float(tokens[2])])
        features.append(cur_feature_list)
        labels.append(label)

    result = {}
    result['labels'] = labels
    result['features'] = features
    result['impression_ids'] = impression_ids
    return result

def convert_libffm_features_to_model_input(raw_features, feature_count, field_count):
    """convert features to fm input"""
    dim = feature_count 
    field_count = field_count 
    labels = raw_features['labels']
    features = raw_features['features']
    impression_ids = raw_features['impression_ids']

    instance_cnt = len(labels)

    fm_feat_indices = []
    fm_feat_values = []
    fm_feat_shape = [instance_cnt, dim]

    dnn_feat_indices = []
    dnn_feat_values = []
    dnn_feat_weights = []
    dnn_feat_shape = [instance_cnt * field_count, -1]

    for i in range(instance_cnt):
        m = len(features[i])
        dnn_feat_dic = {}
        for j in range(m):
            fm_feat_indices.append([i, features[i][j][1]])
            fm_feat_values.append(features[i][j][2])
            if features[i][j][0] not in dnn_feat_dic:
                dnn_feat_dic[features[i][j][0]] = 0
            else:
                dnn_feat_dic[features[i][j][0]] += 1
            dnn_feat_indices.append([i * field_count + features[i][j][0], \
                                        dnn_feat_dic[features[i][j][0]]])
            dnn_feat_values.append(features[i][j][1])
            dnn_feat_weights.append(features[i][j][2])
            if dnn_feat_shape[1] < dnn_feat_dic[features[i][j][0]]:
                dnn_feat_shape[1] = dnn_feat_dic[features[i][j][0]]
    dnn_feat_shape[1] += 1

    sorted_index = sorted(range(len(dnn_feat_indices)),
                            key=lambda k: (dnn_feat_indices[k][0], \
                                            dnn_feat_indices[k][1]))

    res = {}
    res['fm_feat_indices'] = np.asarray(fm_feat_indices, dtype=np.int64)
    res['fm_feat_values'] = np.asarray(fm_feat_values, dtype=np.float32)
    res['fm_feat_shape'] = np.asarray(fm_feat_shape, dtype=np.int64)
    res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)

    res['dnn_feat_indices'] = np.asarray(dnn_feat_indices, dtype=np.int64)[sorted_index]
    res['dnn_feat_values'] = np.asarray(dnn_feat_values, dtype=np.int64)[sorted_index]
    res['dnn_feat_weights'] = np.asarray(dnn_feat_weights, dtype=np.float32)[sorted_index]
    res['dnn_feat_shape'] = np.asarray(dnn_feat_shape, dtype=np.int64)
    res['impression_ids'] = impression_ids
    return res

def get_initializer(hparams):
    """get initializer"""
    if hparams.init_method == 'tnormal':
        init = tf.truncated_normal_initializer(stddev=hparams.init_value)
    elif hparams.init_method == 'uniform':
        init = tf.random_uniform_initializer(-hparams.init_value, hparams.init_value)
    elif hparams.init_method == 'normal':
        init = tf.random_normal_initializer(stddev=hparams.init_value)
    elif hparams.init_method == 'xavier_normal':
        init = tf.contrib.layers.xavier_initializer(uniform=False)
    elif hparams.init_method == 'xavier_uniform':
        init = tf.contrib.layers.xavier_initializer(uniform=True)
    elif hparams.init_method == 'he_normal':
        init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
    elif hparams.init_method == 'he_uniform':
        init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True)
    else:
        init = tf.truncated_normal_initializer(stddev=hparams.init_value)
    return init

def get_pred(logit, hparams):
    """get pred func"""
    if hparams.method == 'regression':
        pred = tf.identity(logit)
    elif hparams.method == 'classification':
        pred = tf.sigmoid(logit)
    else:
        raise ValueError("method must be regression or classification, but now is {0}".format(hparams.method))
    return pred

def activate(logit, activation):
    """activate"""
    if activation == 'sigmoid':
        return tf.nn.sigmoid(logit)
    elif activation == 'softmax':
        return tf.nn.softmax(logit)
    elif activation == 'relu':
        return tf.nn.relu(logit)
    elif activation == 'tanh':
        return tf.nn.tanh(logit)
    elif activation == 'elu':
        return tf.nn.elu(logit)
    elif activation == 'identity':
        return tf.identity(logit)
    else:
        raise ValueError("this activations not defined {0}".format(activation))

def compute_data_loss(hparams, labels, preds, logits):
    if hparams.loss == 'cross_entropy_loss':
        data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(logits, [-1]), labels=tf.reshape(labels, [-1])))
    elif hparams.loss == 'square_loss':
        data_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.reshape(preds, [-1]), tf.reshape(labels, [-1]))))
    elif hparams.loss == 'log_loss':
        data_loss = tf.reduce_mean(tf.losses.log_loss(predictions=tf.reshape(preds, [-1]),labels=tf.reshape(labels, [-1])))
    else:
        raise ValueError("this loss not defined {0}".format(hparams.loss))
    return data_loss

def l2_loss(state, hparams):
    loss = tf.zeros([1], dtype=tf.float32)
    # embedding_layer l2 loss
    for param in state.embed_params:
        loss = tf.add(loss, tf.multiply(hparams.embed_l2, tf.nn.l2_loss(param)))
    for param in state.layer_params:
        loss = tf.add(loss, tf.multiply(hparams.layer_l2, tf.nn.l2_loss(param)))
    return loss

def l1_loss(state, hparams):
    loss = tf.zeros([1], dtype=tf.float32)
    # embedding_layer l1 loss
    for param in state.embed_params:
        loss = tf.add(loss, tf.multiply(hparams.embed_l1, tf.norm(param, ord=1)))
    for param in state.layer_params:
        loss = tf.add(loss, tf.multiply(hparams.layer_l1, tf.norm(param, ord=1)))
    return loss

def compute_regular_loss(state, hparams):
    regular_loss = l2_loss(state, hparams) + l1_loss(state, hparams)
    regular_loss = tf.reduce_sum(regular_loss)
    return regular_loss

def get_train_optimizer(hparams, loss, global_step):
    """build train opt"""
    if hparams.optimizer == 'adadelta':
        train_opt = tf.train.AdadeltaOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'adagrad':
        train_opt = tf.train.AdagradOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'sgd':
        train_opt = tf.train.GradientDescentOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'adam':
        train_opt = tf.train.AdamOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'ftrl':
        train_opt = tf.train.FtrlOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'gd':
        train_opt = tf.train.GradientDescentOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'padagrad':
        train_opt = tf.train.ProximalAdagradOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'pgd':
        train_opt = tf.train.ProximalGradientDescentOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    elif hparams.optimizer == 'rmsprop':
        train_opt = tf.train.RMSPropOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    else:
        train_opt = tf.train.GradientDescentOptimizer(hparams.learning_rate).minimize(loss, global_step=global_step)
    return train_opt

def active_layer(logit, layer_idx, layer_keeps, activation):
    logit = dropout(logit, layer_idx, layer_keeps)
    logit = activate(logit, activation)
    return logit

def dropout(logit, layer_idx, layer_keeps):
    logit = tf.nn.dropout(x=logit, keep_prob=layer_keeps[layer_idx])
    return logit

# Deep wide model
def build_graph(state, hparams):

    state.keep_prob_train = 1 - np.array(hparams.dropout)
    state.keep_prob_test = np.ones_like(hparams.dropout)
    state.layer_keeps = tf.placeholder(tf.float32)

    with tf.variable_scope("DeepWide") as scope:
        with tf.variable_scope("embedding", initializer=state.initializer) as escope:
            state.embedding = tf.get_variable(name='embedding_layer',
                                             shape=[hparams.feature_count, hparams.dim],
                                             dtype=tf.float32)
            state.embed_params.append(state.embedding)
        logit = build_linear(state, hparams)
        logit = tf.add(logit, build_dnn(state, hparams))
        return logit

def build_train_feed_dict(state, input_data):
    """build train feed dict"""
    feed_dict = {
        state.labels: input_data['labels'],
        state.fm_feat_indices: input_data['fm_feat_indices'],
        state.fm_feat_values: input_data['fm_feat_values'],
        state.fm_feat_shape: input_data['fm_feat_shape'],
        state.dnn_feat_indices: input_data['dnn_feat_indices'],
        state.dnn_feat_values: input_data['dnn_feat_values'],
        state.dnn_feat_weights: input_data['dnn_feat_weights'],
        state.dnn_feat_shape: input_data['dnn_feat_shape'],
        state.layer_keeps: state.keep_prob_train
    }
    return feed_dict


def build_eval_feed_dict(state, input_data):
    """build eval feed dict"""
    feed_dict = {
        state.labels: input_data['labels'],
        state.fm_feat_indices: input_data['fm_feat_indices'],
        state.fm_feat_values: input_data['fm_feat_values'],
        state.fm_feat_shape: input_data['fm_feat_shape'],
        state.dnn_feat_indices: input_data['dnn_feat_indices'],
        state.dnn_feat_values: input_data['dnn_feat_values'],
        state.dnn_feat_weights: input_data['dnn_feat_weights'],
        state.dnn_feat_shape: input_data['dnn_feat_shape'],
        state.layer_keeps: state.keep_prob_test
    }
    return feed_dict


def build_linear(state, hparams):
    
    with tf.variable_scope("linear_part", initializer=state.initializer) as scope:
        w_linear = tf.get_variable(name='w',
                                   shape=[hparams.feature_count, 1],
                                   dtype=tf.float32)
        b_linear = tf.get_variable(name='b',
                                   shape=[1],
                                   dtype=tf.float32)
        x = tf.SparseTensor(state.fm_feat_indices,
                            state.fm_feat_values,
                            state.fm_feat_shape)
        linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
        state.layer_params.append(w_linear)
        state.layer_params.append(b_linear)
        tf.summary.histogram("linear_part/w", w_linear)
        tf.summary.histogram("linear_part/b", b_linear)
        return linear_output


def build_dnn(state, hparams):

    fm_sparse_indexs = tf.SparseTensor(state.dnn_feat_indices,
                                       state.dnn_feat_values,
                                       state.dnn_feat_shape)
    w_fm_sparse_weight = tf.SparseTensor(state.dnn_feat_indices,
                                         state.dnn_feat_weights,
                                         state.dnn_feat_shape)
    w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(state.embedding,
                                                        fm_sparse_indexs,
                                                        w_fm_sparse_weight,
                                                        combiner="sum")
    w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.field_count])
    last_layer_size = hparams.field_count * hparams.dim
    layer_idx = 0
    hidden_nn_layers = []
    hidden_nn_layers.append(w_fm_nn_input)
    layer_keeps = state.layer_keeps

    with tf.variable_scope("nn_part", initializer=state.initializer) as scope:
        for idx, layer_size in enumerate(hparams.layer_sizes):
            curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                              shape=[last_layer_size, layer_size],
                                              dtype=tf.float32)
            curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                              shape=[layer_size],
                                              dtype=tf.float32)
            tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx),
                                 curr_w_nn_layer)
            tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx),
                                 curr_b_nn_layer)
            curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                   curr_w_nn_layer,
                                                   curr_b_nn_layer)
            scope = "nn_part" + str(idx)
            activation = hparams.activation[idx]
            curr_hidden_nn_layer = active_layer(logit=curr_hidden_nn_layer,
                                                      #scope=scope,
                                                      layer_keeps=layer_keeps,
                                                      activation=activation,
                                                      layer_idx=idx)
            hidden_nn_layers.append(curr_hidden_nn_layer)
            layer_idx += 1
            last_layer_size = layer_size
            state.layer_params.append(curr_w_nn_layer)
            state.layer_params.append(curr_b_nn_layer)

        w_nn_output = tf.get_variable(name='w_nn_output',
                                      shape=[last_layer_size, 1],
                                      dtype=tf.float32)
        b_nn_output = tf.get_variable(name='b_nn_output',
                                      shape=[1],
                                      dtype=tf.float32)
        tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx),
                             w_nn_output)
        tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx),
                             b_nn_output)
        state.layer_params.append(w_nn_output)
        state.layer_params.append(b_nn_output)
        nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
        return nn_output


def calc_metrics(hparams, labels, preds):
    """Calculate metrics,such as auc, logloss, group auc"""
    res = {}

    for metric in hparams.metrics:
        if metric == 'auc':
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res['auc'] = round(auc, 4)
        elif metric == 'rmse':
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res['rmse'] = np.sqrt(round(rmse, 4))
        elif metric == 'logloss':
            # avoid logloss nan
            preds = [max(min(p, 1. - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res['logloss'] = round(logloss, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res

class ModelState(object):
    def __init__(self, hparams):
        self.fm_feat_indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.fm_feat_values = tf.placeholder(tf.float32, shape=[None])
        self.fm_feat_shape = tf.placeholder(tf.int64, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])

        self.dnn_feat_indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.dnn_feat_values = tf.placeholder(tf.int64, shape=[None])
        self.dnn_feat_weights = tf.placeholder(tf.float32, shape=[None])
        self.dnn_feat_shape = tf.placeholder(tf.int64, shape=[None])

        self.layer_params = []
        self.embed_params = []
        self.layer_keeps = None
        self.keep_prob_train = None
        self.keep_prob_test = None
        self.embedding = None
        self.initializer = get_initializer(hparams)

FLAGS = None

def main(_):

    logger.info("FLAGS: %s",FLAGS)

    train_output_dir = FLAGS.train_output_dir
    summary_output_dir = FLAGS.summary_output_dir if FLAGS.summary_output_dir else train_output_dir 

    train_file_path = FLAGS.train_file_path.split(",")

    conf = get_conf(FLAGS.conf_path) 
    conf['field_count'] = FLAGS.field_count
    conf['feature_count'] = FLAGS.feature_count
    conf['learning_rate'] = FLAGS.learning_rate

    hparams = create_hparams(conf)


    # Get ps and worker hosts
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)


    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

            train_dataset = tf.data.TextLineDataset(train_file_path).shard(len(worker_hosts), FLAGS.task_index).batch(FLAGS.batch_size)
            train_iterator = train_dataset.make_initializable_iterator()
            train_next_ele = train_iterator.get_next()

            state = ModelState(hparams)

            # Create the graph, etc.
            logit = build_graph(state, hparams)
            pred = get_pred(logit, hparams)

            data_loss = compute_data_loss(hparams, state.labels, pred, logit)
            regular_loss = compute_regular_loss(state, hparams)
            loss = tf.add(data_loss, regular_loss)

            global_step = tf.train.get_or_create_global_step()
            #global_step = tf.Variable(0, trainable=False, name='global_step')
            train_opt = get_train_optimizer(hparams, loss, global_step)
            saver = tf.train.Saver(max_to_keep=hparams.epochs)

            tf.summary.scalar("data_loss", data_loss)
            tf.summary.scalar("regular_loss", regular_loss)
            tf.summary.scalar("loss", loss)
            summary_op = tf.summary.merge_all()

            global_init_op = tf.global_variables_initializer()
            #local_init_op = tf.local_variables_initializer()

        num_train_steps = FLAGS.num_train_steps 
        #hooks=[tf.train.StopAtStepHook(last_step=10000000000)]
        hooks = []

        is_chief = (FLAGS.task_index == 0)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=FLAGS.train_output_dir,
                                               save_summaries_steps = None,
                                               save_summaries_secs = None,
                                               hooks=hooks) as sess:

            if is_chief: summary_writer = tf.summary.FileWriter(summary_output_dir, sess.graph)

            # Initialize the variables (like the epoch counter).
            sess.run(global_init_op)

            step = 0
            sess.run(train_iterator.initializer)
            epoch = 0
            cur_su_steps = 0

            while epoch < FLAGS.num_epochs:
                try:
                    train_lines = sess.run(train_next_ele)
                    #logger.info("---- %s", train_data)
                    train_features = extract_libffm_features(train_lines)
                    train_model_input = convert_libffm_features_to_model_input(train_features, FLAGS.feature_count, FLAGS.field_count)  
                    _, step_loss, step_data_loss, g_step, step_su = sess.run([train_opt, loss, data_loss, global_step, summary_op], feed_dict=build_train_feed_dict(state, train_model_input))
                    step += 1

                    if is_chief and (g_step - cur_su_steps) >= FLAGS.save_summaries_steps:
                        cur_su_steps = g_step
                        summary_writer.add_summary(step_su, g_step)
                        
                    if step % FLAGS.show_step == 0:
                        logger.info('step {0:d} , step_loss: {1:.4f}, step_data_loss: {2:.4f}, task_index: {3:d}, global_step: {4:d}'.format(step, step_loss, step_data_loss, FLAGS.task_index, g_step))
                    if num_train_steps > 0 and g_step > num_train_steps: break

                except tf.errors.OutOfRangeError:
                    logger.info('--- epoch %d is done, task_index: %d, step: %d, batch_size: %d', epoch, FLAGS.task_index, step, FLAGS.batch_size) 
                    sess.run(train_iterator.initializer)
                    epoch += 1 

            if is_chief: summary_writer.close()
        logger.info("Finished, is_chief: %s, time: %s", is_chief, time.ctime())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
    )

    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
    )

    parser.add_argument(
      "--batch_size",
      type=int,
      default=500,
      help="Batch size"
    )

    parser.add_argument(
      "--field_count",
      type=int,
      default=2,
      help="Field count"
    )

    parser.add_argument(
      "--feature_count",
      type=int,
      default=142773,
      help="Feature count"
    )

    parser.add_argument(
      "--num_epochs",
      type=int,
      default=1,
      help="Num of epochs"
    )

    parser.add_argument(
      "--conf_path",
      type=str,
      default="network.yaml",
      help="Config file path"
    )

    parser.add_argument(
      "--show_step",
      type=int,
      default=100,
      help="Show step"
    )

    parser.add_argument(
      "--save_summaries_steps",
      type=int,
      default=10,
      help="Save summaries steps"
    )

    parser.add_argument(
      "--num_records",
      type=int,
      default=0,
      help="Total records count in the training file"
    )

    parser.add_argument(
      "--num_train_steps",
      type=int,
      default=0,
      help="Total train steps"
    )


    parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      help="Learning rate"
    )

    parser.add_argument(
      "--train_file_path",
      type=str,
      default="",
      required=True,
      help="Train input file path"
    )

    parser.add_argument(
      "--eval_file_path",
      type=str,
      default="",
      help="Eval input file path"
    )

    parser.add_argument(
      "--train_output_dir",
      type=str,
      default="",
      required=True,
      help="Train output dir"
    )

    parser.add_argument(
      "--summary_output_dir",
      type=str,
      default="",
      help="Summary output dir"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
