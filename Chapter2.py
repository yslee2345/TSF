import tensorflow as tf
###Linear Regression ###


#H(x)=Wx + b
#X and Y data
x_train = [1,2,3]
y_train = [1,2,3]
W=tf.Variable(tf.random_normal([1]),name='weight') #값이 하나인 1차원 array
b=tf.Variable(tf.random_normal([1]),name='bias') #Variable은 텐서플로우가 자체적으로 변경시키는 값, trainable 한 variable값.
#Hypothesis
hypothesis = x_train * W + b
#cost/loss functuin
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #제곱의 평균을 구하는 것.
#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # 무엇을 minimize할것인가? -> cost를!
sess=tf.Session()
sess.run(tf.global_variables_initializer()) #텐서플로우의 variable을 사용하기 위해서는 다음과 같이 초기화 시켜주어야 함.
for step in range(10001):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))

######Placeholder############
#x_train, y_train 처럼 값을 직접 주지 않고.
import tensorflow as tf
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
hypothesis = X*W + b
cost=tf.reduce_mean(tf.square(hypothesis - Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost,W,b,train], feed_dict={X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]}) #train을 실행시킬때 feed dict를 통해 값을 넘겨줄 수 있다.
    if step % 20 == 0:
        print(step,cost_val,W_val,b_val)

