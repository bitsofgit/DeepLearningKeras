import tensorflow as tf

# suppresses level 1 and 0 warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# print version
print("Tensorflow version is : " + str(tf.__version__))

# verify session works
hello = tf.constant("Hello from tensorflow")
sess = tf.Session()
print(sess.run(hello))