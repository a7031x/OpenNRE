from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_att(is_training):
    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)

    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    # PCNN. Appoint activation to whatever activation function you want to use.
    # There are three more encoders:
    #     framework.encoder.cnn
    #     framework.encoder.rnn
    #     framework.encoder.birnn
    x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
    # Selective attention. Setting parameter dropout_before=True means using dropout before attention. 
    # There are three more selecting method
    #     framework.selector.maximum
    #     framework.selector.average
    #     framework.selector.no_bag
    logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(logit)
        output = framework.classifier.output(logit)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(logit)
        framework.load_test_data()
        framework.test()

