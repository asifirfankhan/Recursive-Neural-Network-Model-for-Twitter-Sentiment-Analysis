# Recursive-Neural-Network-Model-for-Twitter-Sentiment-Analysis

<h2>Why RNN Model</h2>

<p>RNN  model is especially useful for processing arbitrary sequences of inputs. In many cases I have sequential data and the order of data is very important for the prediction or decision making. In a normal sentence the order of the words are crucial, the reason for these is that the decision about the positive or negative sentence not only depends on a single word but also on the complete sequence of words. RNN model is proposed to take care of that kind of sequential data. In RNN there are many feed forward neural networks, but hidden layers of feed forwards are connected.</p>


<h2>How RNN Model works</h2>

<p>When n-gram is given to the RNN models, it is parsed into a binary tree and each leaf node, corresponding to a word, is represented as a vector. Recursive neural models will then compute parent vectors in a bottom up fashion using different types of compositionality functions g. The parent vectors are again given as features to a classifier.</p>
