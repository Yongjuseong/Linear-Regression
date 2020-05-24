import tensorflow as tf
xData = [1,2,3,4,5,6,7]
yData = [25000,55000,75000,110000,128000,155000,180000]
W = tf.Variable(tf.random_uniform([1],-100,100)) # weight random_uniform => making random value between -100 and 100
b = tf.Variable(tf.random_uniform([1],-100,100)) # bias
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W*X+b
cost = tf.reduce_mean(tf.square(H-Y))
a = tf.Variable(0.01) # learning rate => interval in gradient descent in each step
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5000):
    sess.run(train,feed_dict={X:xData,Y:yData})
    if i% 500 ==0:
        print(i,sess.run(cost,feed_dict={X:xData,Y:yData}),sess.run(W),sess.run(b))
print(sess.run(H,feed_dict={X:[8]}))





