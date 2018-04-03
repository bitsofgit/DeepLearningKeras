import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int32, shape=[3], name="x")
y = tf.placeholder(tf.int32, shape=[3], name="y")

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name = "prod_y")

final_div = tf.div(sum_x, prod_y, name = "final_div")
final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

with tf.Session() as sess:
    print("sum(x):" ,sess.run(sum_x, feed_dict={x:[100,200,300]}))
    print("prod(y):", sess.run(prod_y, feed_dict={y:[1,2,3]}))

writer = tf.summary.FileWriter("./PlaceHoldersGraph", sess.graph)

writer.close()

# Kind of like console.ReadLine()
input("Enter to exit...")