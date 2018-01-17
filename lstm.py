import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

data_file = 'lovecraft.txt'
seed_text = 'Some words '
generate_length = 100

time_steps = 7
learning_rate = 0.001
train_epochs = 1000
batch_size = 64
n_units = 128

char_encoder = LabelEncoder()


def load_data(filename):
    with open(filename, 'r') as file:
        text = file.read()

    split_text = list(text)

    return char_encoder.fit_transform(split_text)[:200000], split_text


def make_sequence(data, steps=1):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(steps, 0, -1):
        cols.append(df.shift(i))
        names.append(f't-{i}')
    cols.append(df)
    names.append('t')
    sequence = pd.concat(cols, axis=1)
    sequence.columns = names
    sequence = sequence.dropna()
    return sequence


def progress_bar(epoch, step):
    percent = int((step * batch_size / train_size) * 100)
    sys.stdout.write('\r')
    sys.stdout.write(f"Epoch {epoch} - {percent}% - {int(time.time() - start_time)}s")
    sys.stdout.flush()


encoded_text, split_text = load_data(data_file)

vocab_size = len(list(set(split_text)))

train_size = int(len(encoded_text) * 0.99)
train_set = make_sequence(encoded_text[:train_size], time_steps)
dev_set = make_sequence(encoded_text[train_size:], time_steps)

print('Training set shape: ', train_set.shape)
print('Dev set shape: ', dev_set.shape)

x_train = train_set.loc[:, :'t-1'].values.reshape((-1, time_steps, 1))
y_train = train_set.loc[:, 't'].values.reshape((-1, 1))
x_dev = dev_set.loc[:, :'t-1'].values.reshape((-1, time_steps, 1))
y_dev = dev_set.loc[:, 't'].values.reshape((-1, 1))

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_dev shape: ', x_dev.shape)
print('y_dev shape: ', y_dev.shape)

X = tf.placeholder('float', [None, time_steps, 1])
Y = tf.placeholder('float', [None, 1])

W = {'out': tf.Variable(tf.random_normal([n_units, vocab_size]))}
b = {'out': tf.Variable(tf.random_normal([vocab_size]))}

saver = tf.train.Saver()


def RNN(x):
    x = tf.unstack(x, time_steps, 1)

    lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(n_units, forget_bias=1)
    lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(n_units, forget_bias=1)
    lstm_cell3 = tf.contrib.rnn.BasicLSTMCell(n_units, forget_bias=1)
    lstm_cell4 = tf.contrib.rnn.BasicLSTMCell(n_units, forget_bias=1)
    lstm_cell5 = tf.contrib.rnn.BasicLSTMCell(n_units, forget_bias=1)
    cells = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2, lstm_cell3, lstm_cell4, lstm_cell5])

    output, states = tf.nn.static_rnn(cells, x, dtype=tf.float32)

    return tf.matmul(output[-1], W['out']) + b['out']


logits = RNN(X)
prediction = tf.nn.softmax(logits)

onehot_labels = tf.reshape(tf.one_hot(tf.cast(Y, tf.int32), vocab_size), [-1, vocab_size])

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss_op)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()


def evaluate(sess, x, y, prefix):
    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x, Y: y})

    print(f'{prefix} | Loss: {loss} - Accuracy: {acc}')


def predict(sess, steps):
    char_seed = char_encoder.transform(list(seed_text))
    current_text = make_sequence(char_seed, time_steps).loc[:, :'t-1'].values.reshape((-1, time_steps, 1))

    output_sample = np.zeros(steps, dtype=np.int32)
    for i in range(steps):
        yhat = sess.run(tf.argmax(logits, 1), feed_dict={X: current_text})
        output_sample[i] = yhat[-1]
        char_seed = np.append(char_seed, yhat[-1])
        current_text = make_sequence(char_seed, time_steps).loc[:, :'t-1'].values.reshape((-1, time_steps, 1))

    output_sample = ''.join(char_encoder.inverse_transform(list(output_sample)))
    print(f'Sample text:\n{output_sample}')


with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "tmp/model.ckpt")

    try:
        for epoch in range(1, train_epochs + 1):
            start_time = time.time()
            for step in range(train_size // batch_size):
                batch_x = x_train[step * batch_size:step * batch_size + batch_size]
                batch_y = y_train[step * batch_size:step * batch_size + batch_size]

                sess.run(train, feed_dict={X: batch_x, Y: batch_y})

                progress_bar(epoch, step)

            if epoch == 1 or epoch % 10 == 0:
                evaluate(sess, x_train, y_train, '')

            if epoch == 1 or epoch % 25 == 0:
                evaluate(sess, x_dev, y_dev, 'DEV SET')
                predict(sess, generate_length)

    except KeyboardInterrupt:
        print('Training aborted')

    saver.save(sess, "tmp/model.ckpt")
    evaluate(sess, x_dev, y_dev, 'DEV SET')
    predict(sess, generate_length)
