from __future__ import division
import os
import sys
import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import argparse
from train_dist import \
    get_conf, \
    create_hparams, \
    build_graph, \
    get_pred, \
    extract_libffm_features, \
    convert_libffm_features_to_model_input, \
    build_train_feed_dict, \
    ModelState

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


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

FLAGS = None

def main(_):
 
    logger.info("FLAGS: %s",FLAGS)

    train_output_dir = FLAGS.train_output_dir
 
    # eval_file_path = ['hdfs://10.190.148.125:9000/path/test/train_data_val_100.ffm']
    eval_file_path = FLAGS.eval_file_path.split(",")

    #train_output_dir = '/home/neilbao/test/train'

    conf = get_conf(FLAGS.conf_path) 
    conf['field_count'] = FLAGS.field_count
    conf['feature_count'] = FLAGS.feature_count

    hparams = create_hparams(conf)

    eval_dataset = tf.data.TextLineDataset(eval_file_path).batch(FLAGS.batch_size)
    eval_iterator = eval_dataset.make_initializable_iterator()
    eval_next_ele = eval_iterator.get_next()

    state = ModelState(hparams)

    # Create the graph, etc.
    logit = build_graph(state, hparams)
    pred = get_pred(logit, hparams)

    saver = tf.train.Saver()
    glob_path = os.path.join(train_output_dir, 'model.ckpt*.index')
    ckpt_list = map(lambda c: c.strip('.index'), tf.gfile.Glob(glob_path))
    logger.info("ckpt list: %s", ckpt_list)
   
    final_eval_model = os.path.join(train_output_dir, "final_eval_model.txt") 
    last_eval_auc = 0

    for ckpt in ckpt_list:
        with tf.Session() as sess:
            logger.info("ckpt: %s", ckpt)
            saver.restore(sess,ckpt)
            sess.run(eval_iterator.initializer)
            pred_list = []
            label_list = []

            while True:
                try:
                    eval_lines = sess.run(eval_next_ele)
                    eval_features = extract_libffm_features(eval_lines)
                    eval_model_input = convert_libffm_features_to_model_input(eval_features, FLAGS.feature_count, FLAGS.field_count)  
                    _pred, _label = sess.run([pred, state.labels], feed_dict=build_train_feed_dict(state, eval_model_input))
                    pred_list.extend(np.reshape(_pred, -1))
                    label_list.extend(np.reshape(_label, -1))

                except tf.errors.OutOfRangeError:
                    break
            
            me = calc_metrics(hparams, label_list, pred_list)
            logger.info("eval: %s", me)
            if last_eval_auc < me['auc']:
                last_eval_auc = me['auc']
                with tf.gfile.GFile(final_eval_model, 'w') as myfile:
                    myfile.write(ckpt)        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

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
      "--eval_file_path",
      type=str,
      default="",
      help="Eval input file path"
    )

    parser.add_argument(
      "--train_output_dir",
      type=str,
      default="",
      help="Train output dir"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
