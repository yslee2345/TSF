#Logistic Classification


#Linear Regression 에서 배운 경사감소법으로는 비용 측정 불가.
#따라서 다른 비용 함수를 사용해야 한다.

#              -log(H(x)) : y=1
# C(H(x),y) =
#              -log(1-H(x)) : y=0

# if문을 쓰기에는 복잡해지니 다음과 같이 식을 작성한다.

# C(H(x),y) = -ylog(H(x)) - (1-y)log(1-H(x))

#만약 y가 1일 경우 c = -log(H(x))
#만약 y가 0일 경우 c = -log(1-H(x))

#위에서 도출한 Cost를 최소화 시켜야 한다.

# Cost(W) = -1/m 시그마 ylog(H(x)) + (1-y)log(1-H(x))

#cost function
import tensorflow as tf
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))

#minimize
a = tf.Variable(0.1) #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

################################### 예제 #######################
#Training Data
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32,shape=[None,2]) #shape에 주의하자. 행은 상관없고 열의 갯수는 지정 x1,x2니 2로 지정
Y = tf.placeholder(tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]),name='weight') #random_normal에서 첫번째 리스트는 들어오는 X가 2개이니 2이고 나가는 Y는 1이어야 하니 1인거다.
b = tf.Variable(tf.random_normal([1]),name='bias') # bias는 Y의 갯수만큼

#                     T
#                   -W X
#시그모이드 =1 1 + e
#               1
#  H(x) = -------------
#           시그모이드
# 이므로
#tf.div(1./1. + tf.exp(tf.matmul(X,W) + b ))를
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)  #이렇게 tf 내부의 sigmoid 함수를 통해 쉽게 구현할 수 있다.

# Cost(W) = -1/m 시그마 ylog(H(x)) + (1-y)log(1-H(x))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#Accuracy computation
#True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost,train], feed_dict={X: x_data, Y:y_data})
        if step % 200 == 0:
            print(step,cost_val)

    h, c, a = sess.run([hypothesis,predicted,accuracy],feed_dict={X: x_data, Y:y_data})
    print("\nHypothesis : ", h, "\nCorrect(Y): ",c,"\nAccuracy : ",a)


########### 당뇨병을 예측해보자 ###################
import numpy as np
xy =np.loadtxt(r'D:/Python/TSF/data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data = xy[0:700,0:-1]  #훈련 데이터 (라벨을 제외한 다른 모든 값들) 총 8개의 칼럼으로 이루어짐.
y_data = xy[0:700, [-1]] #라벨

x_test_data = xy[701:759,0:-1]
y_test_data = xy[701:759,[-1]]

X = tf.placeholder(tf.float32,shape=[None,8]) #칼럼이 8개니깐..
Y = tf.placeholder(tf.float32,shape=[None,1]) #아웃풋이 1개니깐..

W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + ( 1 - Y ) * tf.log(1-hypothesis) )
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {X: x_data, Y: y_data}
    feed_test = {X:x_test_data, Y:y_test_data}
    for step in range(10001):
        sess.run(train,feed_dict=feed)
        if step % 200 == 0:
            print(step,sess.run(cost,feed_dict=feed))

    h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict=feed_test)
    print('\nHypothesis : ', h, "\nCorrect(Y) : ",c , "\nAccuracy : ",a)

