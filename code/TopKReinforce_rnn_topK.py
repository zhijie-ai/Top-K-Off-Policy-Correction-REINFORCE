#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/6/15 下午6:44                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import math
import pickle
from tensorflow.contrib import rnn
import os

# 主网络和beta网络的实现
# topk修正后的概率
# 和TopKReinforce.py的区别是，在训练RNN的思路时是借鉴session-based RNN的思路

def cascade_model(p,k):
    return 1-(1-p)**k

# lambda比重
def gradient_cascade(p, k):
    return k*(1-p)**(k-1)

def load_data(path):
    from sklearn.model_selection import train_test_split
    import os

    def _discount_and_norm_rewards(rewards,gamma=.9):
        rewards = list(rewards)
        discounted_episode_rewards = np.zeros_like(rewards,dtype='float64')
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        # discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        # discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    ratings = pd.read_csv(path,index_col=None)
    ratings.sort_values(by=['userid','timestamp'],inplace=True)
    items = list(sorted(ratings.itemid.unique()))
    key_to_id_item = dict(zip(items,range(len(items))))
    ratings.itemid = ratings.itemid.map(key_to_id_item)
    ratings['rewards'] = ratings.groupby('userid')['rating'].transform(_discount_and_norm_rewards)

    # 切分训练集和测试集
    train_u,test_u = train_test_split(ratings.userid.unique(),test_size=0.2,random_state=1126)
    train = ratings[ratings.userid.isin(train_u)]
    test = ratings[ratings.userid.isin(test_u)]

    suffix = path[path.find('_')+1:path.find('.csv')]
    train_path = '../data/train_ratings_{}.csv'.format(suffix)
    test_path = '../data/test_ratings_{}.csv'.format(suffix)
    if not os.path.exists(train_path):
        train.to_csv(train_path,index=None)
        test.to_csv(test_path,index=None)
    return train,test,ratings.itemid.nunique()


class TopKReinforce():
    def __init__(self,sess,item_count,embedding_size=64,is_train=True,topK=1,
                 weight_capping_c=math.e**3,batch_size=128,epochs = 1000,hidden_size=1024,
                 gamma=0.95,model_name='reinforce_prior_rnn',num_sampled=50):
        self.sess = sess
        self.item_count=item_count
        self.embedding_size=embedding_size
        self.rnn_size = 128
        self.log_out = 'out/logs_rnn_topK'
        self.topK = topK
        self.weight_capping_c = weight_capping_c# 方差减少技术中的一种 weight capping中的常数c
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_size=hidden_size
        self.gamma = gamma
        self.model_name=model_name
        self.checkout = 'checkout/model_rnn_topK'
        self.kl_targ = 0.02
        self.num_sampled=num_sampled

        self.action_source = {"pi": "beta", "beta": "beta"}#由beta选择动作

        self._init_graph()

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # self.log_writer = tf.summary.FileWriter(self.log_out, self.sess.graph)


        if not is_train:
            self.restore_model()

    def __str__(self):
        dit = self.__dict__
        show = ['item_count','embedding_size','is_train','topK','weight_capping_c','batch_size','epochs','gamma','model_name','time_step','num_sampled']
        dict = {key:val for key,val in dit.items() if key in show}
        return str(dict)


    def weight_capping(self,cof):
        return tf.minimum(cof,self.weight_capping_c)

    # 注意，本论文的off-policy的实现和真正的off-policy有点不太一样。真正的off-policy的beta策略是需要和环境
    # 交互收集数据的。需要探索的过程。而此论文的off-policy中的beta策略不需要去收集收集。只需要提供计算概率的功能就ok了
    def choose_action(self, history):
        # Reshape observation to (num_features, 1)
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.PI, feed_dict = {self.input: history})
        action = list(map(lambda x:np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel()),prob_weights))

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        # action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

        # # exploration to allow rare data appeared
        # if random.randint(0,1000) < 1000:
        #     pass
        # else:
        #     action = random.randint(0,self.n_y-1)
        return action

    # 推理,针对一个用户
    def predict_one_session(self, history):
        state = np.zeros([1,self.rnn_size],dtype=np.float32)
        for i in history:
            alpha,state = self.sess.run([self.alpha,self.final_state],feed_dict={self.X:[i],self.state:state})
        # Reshape observation to (num_features, 1)
        # Run forward propagation to get softmax probabilities
        prob_weights = alpha
        # action = tf.arg_max(prob_weights[0])
        actions = tf.nn.top_k(prob_weights[0],self.topK)
        # tf.nn.in_top_k
        return actions['indices']

    def save_model(self,step):
        if not os.path.exists(self.checkout):
            os.makedirs(self.checkout)

        self.saver.save(self.sess,os.path.join(self.checkout,self.model_name),global_step=step,write_meta_graph=True)

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkout)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('HHHHHHHHHHHHHH',ckpt_name)
            self.saver.restore(self.sess,os.path.join(self.checkout,ckpt_name))

    def pi_beta_sample(self):
        # 1. obtain probabilities
        # note: detach is to block gradient
        beta_probs =self.beta
        pi_probs = self.PI

        # 2. probabilities -> categorical distribution.
        beta_categorical = tf.distributions.Categorical(beta_probs)
        pi_categorical = tf.distributions.Categorical(pi_probs)

        # 3. sample the actions
        # See this issue: https://github.com/awarebayes/RecNN/issues/7
        # usually it works like:
        # pi_action = pi_categorical.sample(); beta_action = beta_categorical.sample();
        # but changing the action_source to {pi: beta, beta: beta} can be configured to be:
        # pi_action = beta_categorical.sample(); beta_action = beta_categorical.sample();
        available_actions = {
            "pi": pi_categorical.sample(),
            "beta": beta_categorical.sample(),
        }
        pi_action = available_actions[self.action_source["pi"]]
        beta_action = available_actions[self.action_source["beta"]]

        # 4. calculate stuff we need
        pi_log_prob = pi_categorical.log_prob(pi_action)
        beta_log_prob = beta_categorical.log_prob(beta_action)

        return pi_log_prob, beta_log_prob, pi_probs

    def _init_graph(self):
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.int32,[None],name='input')
            self.label = tf.placeholder(tf.int32,[None],name='label')
            self.discounted_episode_rewards_norm = tf.placeholder(shape=[None],name='discounted_rewards',dtype=tf.float32)
            self.state = tf.placeholder(tf.float32,[None,self.rnn_size],name='rnn_state')

        cell = rnn.GRUCell(self.rnn_size)
        with tf.variable_scope('emb'):
            embedding = tf.get_variable('item_emb',[self.item_count,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding,self.X)

        print('AAAA',inputs)
        outputs,states_ = cell.__call__(inputs,self.state)# outputs为最后一层每一时刻的输出
        print(outputs.shape,states_)
        self.final_state = states_

        # state = tf.reshape(outputs,[-1,self.rnn_size])#bs*step,rnn_size,state
        state = outputs

        with tf.variable_scope('main_policy'):
            weights=tf.get_variable('item_emb_pi',[self.item_count,self.rnn_size])
            bias = tf.get_variable('bias',[self.item_count])
            self.pi_hat =tf.add(tf.matmul(state,tf.transpose(weights)),bias)
            self.PI =  tf.nn.softmax(self.pi_hat)# PI策略
            self.alpha = cascade_model(self.PI,self.topK)

        with tf.variable_scope('beta_policy'):
            weights_beta=tf.get_variable('item_emb_beta',[self.item_count,self.rnn_size])
            bias_beta = tf.get_variable('bias_beta',[self.item_count])
            self.beta = tf.add(tf.matmul(state,tf.transpose(weights_beta)),bias_beta)
            self.beta =  tf.nn.softmax(self.beta)# β策略

        label = tf.reshape(self.label,[-1,1])
        with tf.variable_scope('loss'):
            pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample()

            ce_loss_main =tf.nn.sampled_softmax_loss(
                weights,bias,label,state,self.num_sampled,num_classes=self.item_count)

            topk_correction =gradient_cascade(tf.exp(pi_log_prob),self.topK)# lambda 比值
            off_policy_correction = tf.exp(pi_log_prob)/tf.exp(beta_log_prob)
            off_policy_correction = self.weight_capping(off_policy_correction)
            # print('CCCCCCCC',self.PI.shape,self.beta.shape)
            # print('DDDDDDD',off_policy_correction.shape,topk_correction.shape,ce_loss_main.shape)# (?,) (?,) (?,)
            self.pi_loss = tf.reduce_mean(off_policy_correction*topk_correction*self.discounted_episode_rewards_norm*ce_loss_main)
            # tf.summary.scalar('pi_loss',self.pi_loss)

            self.beta_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights_beta,bias_beta,label,state,self.num_sampled,num_classes=self.item_count))
            # tf.summary.scalar('beta_loss',self.beta_loss)

        with tf.variable_scope('optimizer'):
            # beta_vars = [var for var in tf.trainable_variables() if 'item_emb_beta' in var.name or 'bias_beta' in var.name]
            beta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='beta_policy')
            self.train_op_pi = tf.train.AdamOptimizer(0.01).minimize(self.pi_loss)
            self.train_op_beta = tf.train.AdamOptimizer(0.01).minimize(self.beta_loss,var_list=beta_vars)

    def init(self,data):
        data.drop(['timestamp','rating'],axis=1,inplace=True)
        offset_sessions = np.zeros(data.userid.nunique()+1,dtype=np.int32)
        offset_sessions[1:] = data.groupby('userid').size().cumsum()
        return offset_sessions

    def train(self,data):

        pi = []
        beta=[]
        counter = 0

        offset_sessions = self.init(data)
        print(data.head(10))

        for epoch in range(self.epochs):
            state = np.zeros([self.batch_size,self.rnn_size],dtype=np.float32)
            session_idx_arr =np.arange(len(offset_sessions)-1)
            iters = np.arange(self.batch_size)

            maxiter = iters.max()
            start =offset_sessions[session_idx_arr[iters]]
            end =offset_sessions[session_idx_arr[iters]+1]

            finished=False
            while not finished:
                minlen =(end-start).min()
                out_idx = data.itemid.values[start]
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx =data.itemid.values[start+i+1]
                    rewards = data.rewards.values[start+i+1]
                    fetches =[self.final_state,self.PI,self.beta,self.pi_loss,self.beta_loss,
                              self.train_op_pi,self.train_op_beta]
                    feed_dict = {self.X:in_idx,self.label:out_idx,
                                 self.discounted_episode_rewards_norm:rewards,
                                 self.state:state}
                    state,pi_new,beta_new,pi_loss,beta_loss,_,_=self.sess.run(fetches,feed_dict)
                    print('num_sampled:{},ite:{},epoch:{},pi loss:{:.2f},beta loss:{:.2f},finished num of user:{}/{}'.format(self.num_sampled,counter,epoch,pi_loss,beta_loss,maxiter+1,data.userid.nunique()))

                    pi.append(pi_loss)
                    beta.append(beta_loss)
                    counter+=1


                start = start + minlen - 1
                mask =np.arange(len(iters))[(end-start)<=1]

                for idx in mask:
                    maxiter +=1
                    if maxiter >= len(offset_sessions)-1:
                        print('epoch finished!!!!!')
                        finished=True
                        break
                    # 用下一个session的数据接力
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] =offset_sessions[session_idx_arr[maxiter]+1]

                if len(mask):
                    state[mask]=0

        # 保存模型
        self.save_model(step=counter)
        return pi,beta

    # 取前10个后10个的均值
    def plot_pi(self,pi_loss,num=10):
        pi_loss_ = [np.mean(pi_loss[ind-num:ind+num]) for ind ,val in enumerate(pi_loss) if ind%1000==num]
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        plt.subplot(211)
        plt.plot(range(len(pi_loss)),pi_loss,label='pi-loss-{}'.format(self.num_sampled),color='g')
        plt.title('pi loss')

        plt.subplot(212)
        plt.plot(range(len(pi_loss_)),pi_loss_,label='pi-loss-{}'.format(self.num_sampled),color='g')
        plt.xlabel('Training Steps')
        plt.ylabel('loss')
        # plt.ylim(0,2000)
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('jpg/reinforce_top{}_pi_rnn_{}.jpg'.format(self.topK,self.num_sampled))

    def plot_beta(self,beta_loss,num=10):
        # pi_loss_ = [val for ind ,val in enumerate(pi_loss) if ind%5000==0]
        # beta_loss_ = [val for ind ,val in enumerate(beta_loss) if ind%5000==0]
        beta_loss_ = [np.mean(beta_loss[ind-num:ind+num]) for ind ,val in enumerate(beta_loss) if ind%1000==num]
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        plt.subplot(211)
        plt.plot(range(len(beta_loss)),beta_loss,label='beta-loss-{}'.format(self.num_sampled),color='r')
        plt.title('beta loss')

        plt.subplot(212)
        plt.plot(range(len(beta_loss_)),beta_loss_,label='beta-loss-{}'.format(self.num_sampled),color='r')
        plt.xlabel('Training Steps')
        plt.ylabel('loss')
        # plt.ylim(0,10)
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('jpg/reinforce_top{}_beta_rnn_{}.jpg'.format(self.topK,self.num_sampled))

    def predict_next_batch(self,session_ids,input_item_ids,batch):
        if not self.predict:
            self.current_session = np.ones(batch)*-1
            self.predict=True

        #session_ids 即为iter,如果此次运行的用户和上次运行的用户不一样，则说明有些用户是新加的，则需要将state置0
        session_change =np.arange(batch)[session_ids!=self.current_session] #session_ids中有些是已经结束了的,-1
        if len(session_change) > 0 :
            self.predict_state[session_change] = 0.0
            self.current_session = session_ids.copy()

        in_idx =input_item_ids
        fetches =[self.pi_hat,self.final_state]
        feed_dict ={self.X:in_idx,self.state:self.predict_state}
        preds,self.predict_state = self.sess.run(fetches,feed_dict)#(batch,n_items)
        preds = np.asarray(preds).T
        return pd.DataFrame(data=preds,index=range(preds.shape[0]))


    def evaluate_sessions_batch(self,test,cut_off=20,batch_size = 2048):
        self.predict=False
        offset_sessions = np.zeros(test['userid'].nunique()+1,dtype=np.int32)
        offset_sessions[1:] = test.groupby('userid').size().cumsum()
        evaluation_point_count = 0
        mrr,recall = 0.0,0.0
        if len(offset_sessions)-1 <batch_size:
            batch_size = len(offset_sessions)-1

        # 初始化state
        self.predict_state = np.zeros([batch_size,self.rnn_size],dtype=np.float32)

        iters =np.arange(batch_size).astype(np.int32)
        maxiter= iters.max()
        start =offset_sessions[iters]
        end =offset_sessions[iters+1]
        in_idx =np.zeros(batch_size,dtype=np.int32)
        np.random.seed(1126)
        while True:
            valid_mask =iters>=0
            if valid_mask.sum() ==0:
                break
            start_valid =start[valid_mask]
            minlen =(end[valid_mask]-start_valid).min()
            in_idx[valid_mask]=test['itemid'].values[start_valid]
            for i in range(minlen-1):
                out_idx =test['itemid'].values[start_valid+i+1]
                preds = self.predict_next_batch(iters,in_idx,batch_size)
                preds.fillna(0,inplace=True)
                # print('UUUUUUUUUUUUU',preds.shape,preds.head())#(3706, 256)
                in_idx[valid_mask]=out_idx
                ranks =(preds.values.T[valid_mask].T > np.diag(preds.loc[in_idx].values)[valid_mask]).sum(axis=0) + 1
                rank_ok =ranks<cut_off
                recall+= rank_ok.sum()
                mrr +=(1.0/ranks[rank_ok]).sum()
                evaluation_point_count += len(ranks)

            start =start+minlen-1
            mask =np.arange(len(iters))[(valid_mask)& (end-start<=1)]
            for idx in mask:
                maxiter +=1
                if maxiter>=len(offset_sessions)-1:
                    iters[idx]=-1
                else:
                    iters[idx] =maxiter
                    start[idx] =offset_sessions[maxiter]
                    end[idx] =offset_sessions[maxiter+1]

        return recall / evaluation_point_count, mrr / evaluation_point_count



if __name__ == '__main__':

    train,test,n_items = load_data('../data/ratings_20m.csv')
    print('num users of test set:{}'.format(test.userid.nunique()))
    t1 = time.time()
    import sys

    try:
        num_sampled = int(sys.argv[1])
    except:
        num_sampled = 50

    print('start model training.......{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))))
    with tf.Session() as sess:
        reinforce = TopKReinforce(sess,item_count=n_items,epochs=50,batch_size=4096,topK=1,num_sampled=num_sampled)
        print('model config :{}'.format(reinforce))
        pi_loss,beta_loss = reinforce.train(train)
        reinforce.plot_pi(pi_loss)
        reinforce.plot_beta(beta_loss)
        t2 = time.time()
        print('model training end~~~~~~{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t2))))
        print('time cost :{} m'.format((t2-t1)/60))

        print('evaluating..................')
        res = reinforce.evaluate_sessions_batch(test)
        t2_ = time.time()
        print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
        print('evaluate_sessions_batch !!!! end ')
        print('evaluation time cost :{} m'.format((t2_-t2)/60))

# tf.random.categorical()
# tf.distributions.Categorical()



