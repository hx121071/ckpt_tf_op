import tensorflow as tf
v1 = tf.Variable(tf.random_normal([1,2]),name="v1")
v2 = tf.Variable(tf.random_normal([1,2]),name="v2")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"save/model.ckpt")
    print("v1",sess.run(v1))
    print("v2",sess.run(v2))
    print("Model Restore")
