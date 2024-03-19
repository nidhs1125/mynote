学习连接https://zhuanlan.zhihu.com/reinforce

[Welcome to Spinning Up in Deep RL! — Spinning Up documentation](https://spinningup.openai.com/en/latest/)

按照spinning up的介绍安装环境https://spinningup.openai.com/en/latest/user/installation.html

# 强化学习(Reinforcement learning)

强化学习（Reinforcement learning，RL）讨论的问题是一个**智能体(agent)** 怎么在一个复杂不确定的 **环境(environment)** 里面去极大化它能获得的奖励。通过感知所处环境的 **状态(state)** 对 **动作(action)** 的 **反应(reward)**， 来指导更好的动作，从而获得最大的 **收益(return)**，这被称为在交互中学习，这样的学习方法就被称作强化学习。



## 1.1核心概念

+ 状态$State$/观察值$Observation$：对环境的完整描述/部分描述，记为$S_t/O_t$
+ 动作空间$Action Space$：智能体采取的行动集合被称为动作空间，记为$A_t$​
+ 策略$Policy$：智能体根据当前状态， 决定下一步执行什么行动的规则
  + 确定性策略
    $a_t=\mu(s_t)$
  + 随机性策略
    $a_t\sim \pi(·|s_t)$​
  + 随机性策略中最常见的策略是分类策略(**categorical policies**)和对角高斯策略(**diagonal Gaussian policies**)。
    + **categorical policies**：指动作空间是离散的，因此，计算出每个动作的概率并进行采样即可。$log~\pi_{\theta}(a|s)=log~[P_{\theta}(s)]_a$
    + **diagonal Gaussian policies**：针对动作空间连续的情形，将动作分解成为不相关的若干子动作，并对每个动作$a_i$，输出一个高斯分布$N(\mu_i,\sigma_i^2)$，并在该分布上采样作为输出的结果。一般$\mu_i$通过神经网络确定，而$\sigma_i$作为独立的预设参数，或者也从神经网络中确定。$log~\pi_{\theta}(a|s)=-\frac{1}{2}(\sum_{i=1}^k(\frac{(a_i-\mu_i)^2}{\sigma_i^2}+2log\sigma_i)+klog2\pi)$



+ 轨迹$Trajectories$：定义为一条状态动作序列$\tau=(s_0,a_0,s_1,a_1,...)$
  + 初始状态由采样而来，定义为：$s_0\sim \rho_0(·)$
  + 状态之间的转移依赖于动作：
    可能是确定的$s_{t+1}=f(s_t,a_t)$
    也可能是离散的$s_{t+1}\sim P(·|s_t,a_t)$

+ 奖励$Reward$​和回报$Return$：由环境给出的标量反馈值。记为$R_t$​。

  $r_t=R(s_t,a_t,s_{t+1})$

  大部分情况下，$r_t$只和当前状态$s_t$有关。

  + 智能体的目标就是最大化一个轨迹的奖励$R(\tau)$​：

    有限长度的未折扣回报$R(\tau)=\sum_{t=0}^T r_t$

    无限长度的折扣回报$R(\tau)=\sum_{t=0}^{\infty}\gamma^t r_t,\gamma\in(0,1)$



![img](D:\云盘\WPS云盘\学习笔记\assets\202202061348504.png)

+ RL Problem

考虑环境转移和策略都是随机的，则$T$步的轨迹概率为
$$
P(\tau|\pi)=\rho_0s_0\prod_{t=0}^{T-1}P(s_{t+1}|s_t.a_t)\pi(a_t|s_t)
\label{1.1.1}
$$
对应的期望$return$为
$$
J(\pi)=\int_{\tau}P(\tau|\pi)R(\tau)=E_{\tau\sim\pi}[R(\tau)]
\label{1.1.2}
$$
则一个最优的策略$\pi^*$就是在所有策略$\pi$中使得$return$最大的那一个
$$
\pi^*=arg~\mathop{max}\limits_{\pi}~J(\pi)
\label{1.1.3}
$$

+ 价值函数$Value~Function$：评估在某个状态下，根据策略进行某个动作的期望$return$

  主要有4种期望函数：

  + 在线策略值函数 **On-Policy Value Function**
    $$
    V^{\pi}(s)=E_{\tau\sim\pi}[R(\tau)|s_0=s]
    \label{1.1.4}
    $$
  + 在线策略动作值函数**On-Policy Action-Value Function**
    $$
    Q^{\pi}(s,a)=E_{\tau\sim\pi}[R(\tau)|s_0=s,a_0=a]
    \label{1.1.5}
    $$
  + 最优值函数**Optimal Value Function**
    $$
    V^*(s)=max_{\pi}(E_{\tau\sim\pi}[R(\tau)|s_0=s])
    \label{1.1.6}
    $$
  + 最优动作值函数**Optimal Action-Value Function**
    $$
    Q^*(s,a)=max_{\pi}(E_{\tau\sim\pi}[R(\tau)|s_0=s,a_0=a])
    \label{1.1.7}
    $$

+ $Bellman~Equations$​：起点的价值是你从起点获得的期望价值，加上下一个点开始你能获得的期望价值。

$$
V^{\pi}(s)=E_{a\sim\pi,s'\sim P}[r(s,a,s')+\gamma V^{\pi}(s')] 
\label{1.1.9}
$$

$$
Q^{\pi}(s,a)=E_{a'\sim \pi,s'\sim P}[r(s,a,s')+\gamma Q^{\pi}(s',a')]
\label{1.1.10}
$$

$$
V^*(s)=arg~\mathop{max}\limits_a~E_{s'\sim P}[r(s,a,s')+\gamma V^*(s')] 
\label{1.1.11}
$$

$$
Q^*(s,a)=arg~\mathop{max}\limits_{a'}~E_{s'\sim P}[r(s,a,s')+\gamma Q^*(s',a')]
\label{1.1.12}
$$



+ $Advanlabele~Functions$：描述某个行为相对平均行为好多少(价值高多少)

$$
A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)
\label{1.1.13}
$$

## 1.2强化学习算法分类

![image-20240301122334993](D:\云盘\WPS云盘\学习笔记\assets\image-20240301122334993.png)

### 基于环境是否已知

主要根据是否对环境进行建模，分为以下两类

+ **Model-Free RL**：不基于模型
+ **Model-Based RL**：基于模型

不基于模型主要有两类方法：

+ **Policy Optimization**：策略被表示为$\pi_{\theta}(a|s)$，优化参数$\theta$以优化策略。可以学习一个$V_{\phi}(s)$作为$V^{\pi}(s)$的估计，用其梯度上升去调整$\theta$，或者最大化$V^{\pi}(s)$的局部近似。这些过程通常是在线的。如**A2C/A3C，PPO**
+ **Q-Learning**：学习一个$Q_{\theta}(s,a)$作为$Q^*(s,a)$的估计。这些优化通常是离线的。之后，利用$Q_{\theta}(s,a)$，即可对当前的行动做出指导。如**DQN,C51**

>Q-learning间接的优化策略，因此可能导致失败。但是某些情况下，Q-learning能够比Policy Optimization更有效地重用历史次数，从而获得更好的采样效率

+ **Interpolating Between Policy Optimization and Q-Learning**：平衡两种算法的优缺点。如**DDPG,SAC**

基于模型的方法非常多：

+ **Pure Planning**：单纯的规划，智能体每次观察环境的时候，都会计算得到一个对于当前模型最优的规划，这里的规划指的是未来一个固定时间段内，智能体会采取的所有行动。智能体先执行规划的第一个行动，然后立即舍弃规划的剩余部分。如**MPC,MBMF**
+ **Expert Iteration**：如**Exlt,AlphaZero**
+ **Data Augmentation for Model-Free Methods**：如**MBVE,World Models**
+ **Embedding Planning Loops into Policies**：如**I2A**

### 基于学习方式

+ **On-Policy**（在线策略）：the target and the behavior polices are the same。也就是Agent以拟合最优策略为主要目标，可能兼具收集数据的目标，而核心特征就是这两个目标会集成到同一个策略中。如**SARSA**
+ **Off-Policy**（离线策略）：the learning is from the data off the target policy。一般来说该类算法维护两个策略，行为策略(behavior policy)与目标策略(target policy)。行为策略控制Agent的行动(负责学习数据的获取)，而目标策略借助数据提升自身性能，拟合最优策略。如**Q-learning**

另外，还有一种**offline reinforcement learning**，其数据提前获取，而Agent在学习时不负责制定行动，而只从已经获得的数据中进行学习。

### 基于学习目标

+ **Policy-Based**基于策略：直接去学习最优的策略，学习特定状态下输出每个动作的概率。如**VPG**
+ **Value-Based**基于价值：对价值函数$(1.1.10)$​进行学习，基于价值函数决定执行的动作。如**Q-Learning,DQN**
+ 也有两者的结合，如**Actor-Critic**

## 1.3Policy Optimization

基于最大化回报$J(\pi_{\theta})=E_{\tau\sim\pi_{\theta}}[R(\tau)]$，利用其梯度对参数进行优化，即：
$$
\theta_{k+1}=\theta_k+\alpha\nabla_{\theta}J(\pi_{\theta})|_{\theta_k}
\label{1.3.1}
$$
为此，需要将回报写成具体的表达式并带入。

+ 轨迹概率公式：详见$(1.1.1)$
+ 期望回报：详见$(1.1.2)$

+ 对数求导技巧：

$$
\nabla_{\theta}P(\tau|\theta)=P(\tau|\theta)\nabla_{\theta}log~P(\tau|\theta)
\label{1.3.2}
$$

+ 轨迹的对数：

$$
\nabla_{\theta}log~P(\tau|\theta)=\sum_{t=0}^T\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)
\label{1.3.3}
$$

+ 回报关于$\theta$的导数：

$$
\begin{aligned}
\nabla_{\theta}J(\pi_{\theta})=E_{\tau\sim\pi_{\theta}}[\sum_{t=0}^T\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)R(\tau)]
\end{aligned}
\label{1.3.4}
$$

那么，在此将期望展开，我们先采集轨迹的集合$D=\{\tau_i\},i\in[1,N]$，那么回报的梯度即为：
$$
\hat{g}=\frac{1}{|D|}\sum_{\tau\in D}\sum_{t=0}^T\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)R(\tau)
\label{1.3.5}
$$

+ **EGLP(Expected Grad-Log-Prob)**引理

设$P_{\theta}(x)$根据参数$\theta$来给定随机变量$x$的分布，即满足$\int_x P_{\theta}(x)=1$，那么有：
$$
E_{x\sim P_{\theta}}[\nabla_{\theta}log~P_{\theta}(x)]=0
\label{1.3.6}
$$

>proof:https://spinningup.readthedocs.io/zh-cn/latest/spinningup/rl_intro3.html

+ **reward-to-go policy gradient**

当前时刻为$t$，考虑式$(1.3.4)$，当前时刻之前的奖励已经固定了，并显然对后面的奖励没有影响，因此，定义从当前时刻开始的回报$\hat{R}_t$
$$
\hat{R}_t=\sum_{t'=t}^T~R(s_{t'},a_{t'},s_{t'+1})
\label{1.3.7}
$$
将该式子替换掉$(1.3.4)$中的$R(\tau)$，得到
$$
\nabla_{\theta}J(\pi_{\theta})=E_{\tau\sim\pi_{\theta}}[\sum_{t=0}^T\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)\sum_{t'=t}^T~R(s_{t'},a_{t'},s_{t'+1})]
\label{1.3.8}
$$
这就是reward-to-go policy gradient。

+ **Baselines in Policy Gradients**

根据式$(1.3.5)$，由于$\int_{a_t}~\pi_{\theta}(a_t|s_t)=1$，因此有
$$
E_{a_t\sim \pi_{\theta}}[\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)]=0
\label{1.3.9}
$$
对于任意的函数$b(s_t)$，其值仅跟$s_t$有关，因此对$\theta$求导时该函数可以认为是常数。在式$(1.3.8)$两边同乘$b(s_t)$，得到
$$
E_{a_t\sim \pi_{\theta}}[\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)b(s_t)]=0
\label{1.3.10}
$$
也就是说，在求回报的梯度时，可以为了计算方便，对奖励函数$R$作偏置。根据式$(1.3.7),(1.3.9)$，得
$$
\nabla_{\theta}J(\pi_{\theta})=E_{\tau\sim\pi_{\theta}}[\sum_{t=0}^T\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)(\sum_{t'=t}^T~R(s_{t'},a_{t'},s_{t'+1})-b(s_t))]
\label{1.3.11}
$$
我们称$b(s_t)$为**Baselines in Policy Gradients**，通常令$b(s_t)=V^{\pi}(s_t)$，即从当前状态出发，根据策略$\pi_{\theta}$行动能获得的期望收益。

不过$V^{\pi}(s_t)$通常难以得到，需要进行估计，记估计值为$V_{\phi}(s_t)$

+ **acquire reward-to-go**

因为实际计算中我们无法预测未来的状态(甚至是下一个状态，因为它由环境随机决定)，因此只能以未来回报的期望$Q^{\pi_{\theta}}(s_t,a_t)$来近似估计$\hat{R}_t$。这是十分自然的。代入式$(1.3.7)$：
$$
\nabla_{\theta}J(\pi_{\theta})=E_{\tau\sim\pi_{\theta}}[\sum_{t=0}^T\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)Q^{\pi_{\theta}}(s_t,a_t)]
\label{1.3.12}
$$
当然，未来回报的期望也是非常难获得的，通常有两类方法：

从当前状态开始，依照策略$\pi_{\theta}$，不断进行模拟，直到游戏结束，获得确切的${\hat{R}}_t$；

另外搭建一个神经网络，以预测$Q^{\pi_{\theta}}(s_t,a_t)$

## 1.4经典算法/论文

### Vanilla Policy Gradient

https://spinningup.openai.com/en/latest/algorithms/vpg.html

结合$\ref{1.3.10},\ref{1.3.12},\ref{1.1.13}$，易得：
$$
\nabla_{\theta}J(\pi_{\theta})=E_{\tau\sim\pi_{\theta}}[\sum_{t=0}^T\nabla_{\theta}log~\pi_{\theta}(a_t|s_t)A^{\pi_{\theta}}(s_t,a_t)]
\label{1.4.1}
$$

### Trust Region Policy Optimization

控制每次梯度更新时，新的策略和原来策略之间的差异不会过大，从而保持策略相对的稳定(利用KL散度测量概率分布之间的距离)。同时可以快速对策略进行更新，从而防止环境快速恶化。



+ KL散度

$$
D_{KL}=\sum_i P(i)log\frac{P(i)}{Q(i)}
\label{1.4.2}
$$



































# Pytorch入门

https://pytorch.org/get-started/locally/#linux-pip

### Tensors

Pytorch中的核心数据类型，有着和ndarray相似的用法。主要API如下。

+ tensor可以在GPU中进行计算。

```python
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

+ tensor的大小用元组进行描述

```
shape = (2,3,)
rand_tensor = torch.rand(shape)
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

+ tensor的代数计算方式多样，加'_'表示对参数的tensor进行操作。

```
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
z1.add_(1)
```

+ tensor和numpy可以相互转化。并且转化前后的tensor和ndarray共享相同的存储空间

```
t = torch.ones(5)
n = t.numpy()
t.add_(1)
print(f"n: {n}")
```

### Datasets & DataLoaders

`torch.utils.data.DataLoader`：对Dataset生产一个可迭代的数据集

`torch.utils.data.Dataset`：存储样本和标签

一个可用于计算的数据集必须有3个基础的API，具体如下。

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

Dataset和DataLoader提供了相应的API，快速实现上述3个函数的功能。以torchvision.FashionMNIST为例，下载数据集(已经是Dataset格式)并加载到DataLoader中，并按照batch_size分批进行读取。

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
#train_features是大小为[batch_size,data_scale]的tensor对象
```

### Transforms

下载TorchVision的数据集才有的参数。

`transform`定义数据的格式

`target_transform`定义标签的格式

### Build the Neural Network

`torch.nn`中包含所有神经网络相关的模块。

搭建神经网络时，定义继承`nn.Module`的类，并调用`nn.Sequential`指定网络结构

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(device)
print(model)
```

使用时，将tensor输入，即可得到输出。

```py
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

并且，可以方便的查看中间步骤的结果

```py
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```



### Automatic Differentiation

利用 `torch.autograd`进行微分。

对于在网络中需要进行调整的参数(tensor)，可以在创建时加`requires_grad=True`，也可以随时执行`x.requires_grad_(True)`。

这样，在计算完损失`loss`之后，执行`loss.backward()`，就可以得到对应参数的梯度$\frac{\partial loss}{\partial x}$

### Optimizing Model Parameters

在进行epoch前，完成载入数据，建立神经网络结构等工作。另外，

指定一个损失函数

```python
loss_fn = nn.CrossEntropyLoss()
```

指定一个优化器

```py
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

每个`epoch`中，对于每一个`batch`，先前向计算得到结果，再计算loss并根据梯度更新参数。

```python
model.train()
for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Save and Load the Model

保存：对于一个已经训练好的模型`model`，操作如下

```python
torch.save(model.state_dict(), 'model_weights.pth')
```

加载：需要先新建一个和保存的模型结构相同的模型，然后载入参数。

```
model = NeuralNetwork() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

# 代码

### RL实例

https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

+ 定义环境：

```
env = gym.make(env_name)
```

环境规定了每次观察值和动作值的张量形式。

+ 定义网络:

```
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
```

+ 定义损失函数

```
def compute_loss(obs, act, weights):
	logp = Categorical(logits=logits_net(obs)).log_prob(act)
    return -(logp * weights).mean()
```

其中`weights`可以是这次动作获得的奖励。

+ 定义优化器，即定义如何根据损失函数的梯度进行参数的更新

```
optimizer = Adam(logits_net.parameters(), lr=lr)
```

+ 获取单次行动策略

对于当前环境有一观察值`obs`，将其输入网络`logits_net`中，并获得返回的确定性动作，或者从离散或连续的随机性策略中进行采样。该例中在离散的策略(categorical policies)中采样。

```py
def get_action(obs):
        return Categorical(logits=logits_net(obs)).sample().item()
```

```py
# save obs
batch_obs.append(obs.copy())

# act in the environment
act = get_action(torch.as_tensor(obs, dtype=torch.float32))
obs, rew, done, _ = env.step(act)

# save action, reward
batch_acts.append(act)
ep_rews.append(rew)
```



+ 进行训练

对于每个epoch，首先根据当前策略，获取至少`batch_size`轮行动(`act`)，获取对应的环境观察值`obs`，回报值`ret,weight`。然后计算损失函数并对参数进行梯度更新。

```
optimizer.zero_grad()
batch_loss = compute_loss(
obs=torch.as_tensor(batch_obs, dtype=torch.float32),
act=torch.as_tensor(batch_acts, dtype=torch.int32),
weights=torch.as_tensor(batch_weights, dtype=torch.float32)
)
batch_loss.backward()
optimizer.step()
```





# Functions

记录一些较为复杂的库函数

+ `scatter_`

`Tensor.scatter_(dim, index, src, reduce=None) → Tensor`

功能：根据$dim$的值，将$src$中的值写入到$self$中由$index$指定的位置。

若$dim==0$，则$self[index[i][j][k]][j][k] = src[i][j][k]$

注意$index$在各个维度上均应小于$src$，且小于$self$。

$reduce$可以指定操作的方式，默认为替换，也可以是加，乘等操作。

示例：

```python
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
tensor([[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]])
>>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
tensor([[1, 2, 3, 0, 0],
        [6, 7, 0, 0, 8],
        [0, 0, 0, 0, 0]])
```



### 

