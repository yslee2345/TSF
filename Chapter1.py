import tensorflow as tf
#########그래프를 통한  실행##########
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

node1=tf.constant(3.0,tf.float32) #tf.float32는 데이터 타입
node2=tf.constant(4.0)
node3=tf.add(node1,node2) #node3= node1 + node2 와 같은 의미이다.
print('node1 : ', node1, "node2 : ", node2) #-> 결과값이 나오는게 아니다.
sess=tf.Session() # 세션을 만들어서. 실행시켜야 한다.
print(sess.run([node1,node2]))
print(sess.run(node3))

#첫번째 해야 할 것이.
#1. 그래프를 빌드해야 한다.
#2. 그래프를 실행시킨다 sess.run
#3. 리턴
###############그래프를 통한 실행#########

##그래프를 만들어 놓고 값을 던져주고 싶다##
#노드를 placeholder로 만든다.
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node=a + b
print(sess.run(adder_node,feed_dict={a:3, b:4.5})) #여기서 값을 넘겨준다.
print(sess.run(adder_node,feed_dict={a:[1,3], b:[2,4]})) #여기서는 리스트의 같은 인덱스에 있는 값끼리 더해져서 결과가 출력된다.

