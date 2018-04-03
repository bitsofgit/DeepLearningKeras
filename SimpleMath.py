import tensorflow as tf

a = tf.constant(6.5, name="const_a")
b = tf.constant(3.4, name="const_b")
c = tf.constant(3.0, name="const_c")
d = tf.constant(100.2, name="const_d")

square = tf.square(a, name="square_a")
power = tf.pow(b,c, name="pow_b_c")
sqrt = tf.sqrt(d, name="sqrt_d")

final_sum = tf.add_n([square, power, sqrt], name="final_sum")


sess = tf.Session()

print("Square of a: ", sess.run(square))
print("Power of b ^ c: ", sess.run(power))
print("Square root of d: ", sess.run(sqrt))
print("Final sum: ", sess.run(final_sum))

another_sum = tf.add_n([a,b,c,d,power], name="another_sum")


writer = tf.summary.FileWriter("./SimpleMathGraph", sess.graph)
writer.close()
sess.close()
# Kind of like console.ReadLine()
input("Enter to exit...")