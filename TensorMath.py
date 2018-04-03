import tensorflow as tf

x = tf.constant([100,200,300], name="x")
y = tf.constant([1,2,3], name="y")

sum_x = tf.reduce_sum(x,name="sum_x")
prod_y = tf.reduce_prod(y,name="prod_y")

final_div = tf.div(sum_x, prod_y, name="final_div")

final_mean =  tf.reduce_mean([sum_x, prod_y], name="final_mean")

sess = tf.Session()

print("x: ", sess.run(x))
print("y: ", sess.run(y))

print("sum_x: ", sess.run(sum_x))
print("prod_y: ", sess.run(prod_y))
print("final_div: ", sess.run(final_div))
print("final_mean: ", sess.run(final_mean))

writer = tf.summary.FileWriter("./TensorMathGraph", sess.graph)

writer.close()
sess.close()
# Kind of like console.ReadLine()
input("Enter to exit...")