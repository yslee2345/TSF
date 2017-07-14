#Multi-variable linear regression

#Hypothesis
#H(x1,x2,x3) = x1w1+x2w2+x3w3 -> bias는 간략하게 하기 위햐여 생략해보자
import tensorflow as tf


x1_data = [73.,93.,89.,95.,73.]
x2_data = [80.,88.,91.,89.,66.]
x3_data = [75.,93.,30.,100.,70.]
y_data = [152.,185.,180.,196.,142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]),name='bias')
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,Y:y_data})
    if step % 10 == 0:
        print(step,"cost : ", cost_val, "\nPrediction\n", hy_val)


#################이 방법은 잘 쓰이지 않음 ##############################################


################# matrix를 이용한 방법 #######################
x_data = [[73.,80.,75.],
          [93.,88.,93.],
          [89.,91.,90.],
          [96.,98.,100.],
          [73.,66.,70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

#placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32,shape=[None,3]) #shape -> None : 행의 갯수는 아무렇게나 해도 되지만 / 3 : element는 3개씩이어야 함
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.matmul(X,W) + b #X와 Y의 행렬 내적

cost = tf.reduce_mean(tf.square(hypothesis - Y)) #편차의 제곱의 평균 최소화 를 비용으로

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost,hypothesis,train], feed_dict={X: x_data, Y:y_data})
    if step % 10 == 0:
        print(step,"Cost : ",cost_val, "\nPrediction:\n", hy_val )


############파일에서 데이터 읽어오기 ####################
import numpy as np
xy = np.loadtxt('D:\Python\TSF/file.csv',delimiter=',',dtype=np.float32)
x_data = xy[:, 0:-1] #전체 행을 가져오되 처음부터 마지막 한 열을 제외한 3개의 열을 가져오겠다.
y_data = xy[:, [-1]] #전체 행을 다 가져오되 마지막 열 하나만 가져오겠다.
y_data
X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.matmul(X,W) + b #X와 Y의 행렬 내적


