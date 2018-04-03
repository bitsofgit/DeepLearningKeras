import tensorflow as tf
import numpy as np

# y = Wx + b
W = tf.Variable([2.5, 4.0], tf.float32, name="var_W")
x = tf.placeholder(tf.float32, name="x")
b = tf.Variable([5.0,10.0], tf.float32, name="var_b")

y = W * x + b

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("Final Result:",sess.run(y, feed_dict={x:[10, 100]}))


number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()

result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print("Result: ", sess.run(result))
        print("Increment multiplier: new value: ", sess.run(multiplier.assign_add(1)))


writer = tf.summary.FileWriter("./VariablesGraph", sess.graph)
writer.close()

# Kind of like console.ReadLine()
input("Enter to exit...")