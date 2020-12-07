# Top-K-Off-Policy-Correction-REINFORCE


Top-K Off-Policy Correction for a REINFORCE Recommender System 论文实现思路。

# 注意:
1. 本代码借鉴的是Session-based-RNN等序列模型用于推荐中的思路，而不是原论文中的CFN的网络来实现。如果采用原始的LSTM或者GRU的训练方式，一者，训练速度很慢，二来会丢失很多数据。采用Session-based-RNN的思路，可以利用每个用户的所有历史数据而不必截断。
2. 原论文中在训练的时候加入了label的信息。由于能力有限，实在不知道该如何实现这种思路，正常来说，输入数据是不应该含有标签的信息的。
3. 原论文中说训练pi网络用了reward>0的数据，训练beta网络用的是全部的(s,a)数据，在实现的过程中，并未完全遵照这种设定。主要原因是ml数据集只有reward>0的数据，
3. 使用的movielen数据集，观看loss的下降曲线，效果还ok，美中不足的是，ml数据集中自带reward，对强化学习来说，reward是很关键的一个因素，如果将该模型用于常规的推荐任务中，需要自己定义reward，比如将点击的item的reward设计为1，曝光未点击的设计为-1，但是这种设计思路会引来另一个问题。用户点击不同的item的reward其实是不一样的。

# 在使用tf.nn.sampled_softmax_loss求交叉熵的过程中，num_sampled参数和batch_size参数会影响最终的loss，经过测试，batch_size是num_sampled的20-100倍之间比较好，在该区间内，效果有差异，但差异不大
