import tensorflow as tf
from train_dist import \
    get_conf, \
    create_hparams, \
    build_graph, \
    get_pred, \
    extract_libffm_features, \
    convert_libffm_features_to_model_input, \
    build_train_feed_dict, \
    ModelState 


class Predictor(object):
    def __init__(self, checkpoint_path):
        conf = get_conf('network.yaml') 
        hparams = create_hparams(conf)
        state = ModelState(hparams)
        logit = build_graph(state, hparams)
        pred = get_pred(logit, hparams)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        self.state = state
        self.hparams = hparams
        self.pred = pred
        self.sess = sess

    def predict(self, x):
        train_features = extract_libffm_features(x)
        train_model_input = convert_libffm_features_to_model_input(train_features, self.hparams.feature_count, self.hparams.field_count)
        pred = self.sess.run([self.pred], feed_dict=build_train_feed_dict(self.state, train_model_input))
        return pred


if __name__ == '__main__':
    print('hello')
    x = ["1 1:56707:1 2:110192:0%00484602595AC8DCF361014101FCBF15_282101", "1 1:45866:1 2:110192:0%00484602595AC8DCF361014101FCBF15_265206"]
    checkpoint_path = '/tmp/cibn/b8f7a66c-b782-4743-bad8-2f7990cd6c31/model.ckpt-204'
    model = Predictor(checkpoint_path)
    p = model.predict(x) 
    print(p)
