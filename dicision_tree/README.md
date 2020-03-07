# 机器学习之决策树模型（1）

# 简介和原理

决策树学习本质上是从训练数据中归纳出一组分类规则。假设给定一个训练数据集：

$$
D = \{(x_1, y_x), (x_2, y_2), ..., (x_N, y_N)\}
$$

其中，$x_i = (x_i^{(1)}, x_i^{(2)}, ..., x_i^{(n)})$，n为特征的个数，
$y_i \in \{1, 2, ..., k\}$是类标记，$i = 1, 2, ..., N$,N是样本容量，
学习的目的是根据给定的训练数据集构建一颗树，非叶子结点代表分类的标准，叶子结点
表示分类后的类别。举个栗子，比如有这么一个数据集：
<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gcjdlwg57fj318w0r6jyr.jpg" width=75% height=75%>  
我们构建这么一棵决策树：  
<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gcje5ezpj7j30ru0rcn07.jpg" width=75% height=75%>  
输入一个实例$x' = (x'^{(1)}, x'^{(2)}), ..., x'^{(n)}$，便可以根据这
个决策树来决定输出的类别。
显而易见，构建一课分类效果好的决策树关键在于非叶子结点的确定，即我们用什么特征
来构造决策树。然后对于每一个叶子结点用多数表决的方式来决定它的类别。

## 如何选择结点特征：信息增益

信息增益是用来衡量某一个特征对于分类效果的好坏的度量。如果一个特征对于数据集的分类
效果比较好，那么这个特征的信息增益便会比较大。如何衡量特征对于数据集的分类效果，
可以用熵和条件熵来度量。  

### 熵

如果$X$是一个取有限个值的离散型变量，其概率分布为：

$$P(X=x_i) = p_i, i=1, 2, ..., n$$

那么随机变量$X$的熵定义为：

$$H(X) = -\sum_{i=1}^np_ilogp_i$$

（由于熵只依赖于X的分布而不依赖于X的取值，所以也可以记为$H(p)$  
从定义可以证明：$0≤H(p)≤logn$，熵是衡量一个随机变量的不确定性，举个栗子，一个
二分变量取0和1的概率为$\frac{1}{10}$和$\frac{9}{10}$的熵要比概率为$\frac{1}{2}$
和$\frac{1}{2}$的熵要小，代表着前者的不确定性更小，后者的不确定性更大。

### 条件熵

设随机变量$(X, Y)$，其联合概率分布为  

$$P(X=x_i, Y=y_j)=p_{ij}, i=1, 2, ..., n; j = 1, 2, ..., m$$

则在随机变量X给定的条件下随机变量Y的条件熵$H(Y|X)$定义为X给定条件下Y的条件分布
概率的熵对X的数学期望（有点绕）：

$$H(Y|X) = \sum_{i=1}^np_iH(Y|X=x_i)$$

其实是根据X把变量分成n个集合，然后对每一个集合计算Y的熵然后加和，就是X对Y的条件熵，
条件熵$H(Y|X)$表示在已知随机变量X的条件下随机变量Y的不确定性。

### 信息增益

有了熵和条件熵的，我们便可以定义信息增益了：特征A对训练数据集D的信息增益$g(D, A)$
，定义为集合D的经验熵$H(D)$与在特征A给定的条件下D的经验条件熵$H(D|A)$之差，即

$$g(D, A) = H(D) - H(D|A)$$

直观上，特征A对训练数据集D的信息增益$g(D, A)$代表着由于特征A而使得对数据集D的分类
的不确定性减少的程度。因此我们倾向于选择对数据集D信息增益大的特征去构建决策树，这样使得
训练出来的模型有着比较好的决策能力。

### 信息增益的算法

输入：训练数据集D和特征A  
输出：特征A对于训练数据集D的信息增益$g(D, A)$  
* 计算数据集D的经验熵$H(D)$:

$$H(D) = -\sum_{k=1}^K\frac{|C_K|}{|D|}log\frac{|C_K|}{|D|}$$

* 计算特征A对训练数据集D的条件熵$H(D|A)$:

$$H(D|A) = \sum_{i=1}^n\frac{|D_i|}{|D|}H(D_i)|$$

* 计算信息增益

$$g(D, A) = H(D|A) - H(D)$$

下面是信息增益的python代码

```
class information_gain:
    def get_entropy(self, X):
        num = X.shape[0]
        dic = {}
        for d in X:
            if d in dic:
                dic[d] += 1
            else:
                dic[d] = 1
        # print(dic)
        H_X = 0
        for key, val in dic.items():
            H_X -= val/num * np.log(val/num)
        return H_X  

    def get_condition_entropy(self, Y_X, X):#Y_X是一个2xn的矩阵
        num = X.shape[0]                    #Y_X中第一行存的是特征Y的特征值，第二行存的是特征X的特征值
        dic = {}
        for d in X:
            if d in dic:
                dic[d] += 1
            else:
                dic[d] = 1
        condition_entrop = 0
        for key, val in dic.items():
            Y_Xi = [Y_X[0][i] for i in range(num) if Y_X[1][i] == key]
            condition_entrop += val/num * self.get_entropy(np.array(Y_Xi))
        return condition_entrop
    def get_information_gain(self, D, A):
        D_A = np.vstack((D, A))
        H_D = self.get_entropy(D) #特征D的经验熵
        H_D_A =  self.get_condition_entropy(D_A, A) #特征A对特征D的条件熵
        ig = H_D - H_D_A #信息增益
        ig_rate = ig / H_D #信息增益比
        return ig, ig_rate
```


### 信息增益比

还有个衡量指标叫信息增益比，是特征选择的另一准则，这里不详细解释，只给出相关定义:  
（信息增益比）特征A对训练数据集D的信息增益比g_R(D, A)定义为其信息增益g(D, A)
与训练数据集D关于特征A的值的熵H_A(D)之比，即：

$$
    g_R(D, A) = \frac{g(D, A)}{H_A(D)}
$$

其中：

$$H_A(D) = -\sum_{i=1}^n\frac{|D_i|}{|D|}log\frac{D_i}{D}$$

n是特征A取值的个数。

## 构建决策树（ID3算法）

这一节主要介绍如何用ID3算法来构建决策树。ID3算法从直观上是递归地用对训练数据集
信息增益最大的特征来构建分类结点。  

### ID3算法

输入：训练数据集D，特征集A，阀值$\epsilon$  
输出：决策树T  

* 若D中所有实例属于同一类$C_k$，则T为单节点树，并将类$C_k$作为该结点的类标记
，返回T
* 若A为空集，则T为单节点树，并将D中实例最大的类$C_k$来作为该结点的类标记，返回T 
* 否则按照信息增益算法计算特征集A中个特征对D的信息增益，选择信息增益最大的特征$A_g$
* 如果特征$A_g$的信息增益小于阀值$\epsilon$，则T为单节点树，并将D中实例数目最大
  的类$C_k$作为该点的类标记，返回T
* 否则对$A_g$的每一可能的取值$a_i$，按照$A_g=a_i$将D分割为若干个非空子集$D_i$,
  将$D_i$中实例树最大的类作为标记，构建子结点，由结点及其子结点构成树T，返回T
* 对第i个子结点，以$D_i$为训练集，以$A - \{A_g\}$为特征集，递归地调用步（1）～步（5）
  ，得到子树$T_i$，返回$T_i$

ID3构建决策树的python代码：

```
class Dicision_tree:
    def ID3(self, Data_set, e, label, list_A):
        D = Data_set.loc[:, ['class']] #类集
        A = Data_set.loc[:, list_A] #特征集
        num = D.shape[0]
        class_D = self.hash(D.values.reshape((1, num))[0])
        if len(class_D) <= 1: #如果类别只有一种，返回单节点树，标签为这个类别
            return node("class", 2, list(class_D.keys())[0])

        if len(list_A) <= 0: #如果特征都分完了，返回单节点树，标签为类集中数量最多的那个类别
            c_k = max(class_D.items(), key=operator.itemgetter(1))[0]
            return node("class", 2, c_k)
        engein_information_gain = {}
        ig = information_gain()

        for c in A.columns:#得到每个特征对类别的信息增益
            engein_information_gain[c] = ig.get_information_gain(D.values.reshape((1, num))[0], A[c].values)[0]
        max_ig = max(engein_information_gain.items(), key=operator.itemgetter(1))#获得信息增益最大的那个
        if max_ig[1] < e:#如果这个信息增益小于阀值e，返回单节点树，标签为类集中数量最大的类别
            # print("less than e")
            c_k = max(class_D.items(), key=operator.itemgetter(1))[0]
            return node("class", 2, c_k)
        dic_max_ig = self.hash(A[max_ig[0]].values)
        T = node(max_ig[0], len(dic_max_ig), label)
        list_A.remove(max_ig[0])
        for key, value in dic_max_ig.items():#根据信息增益最大的特征对类集进行划分
            tmp = pd.DataFrame(columns=(list_A)+["class"])
            for i in range(num):
                if Data_set.loc[i, [max_ig[0]]].values[0] == key:
                    tmp = tmp.append(Data_set.loc[i, list_A + ["class"]], ignore_index=True)    
            t_dic = self.hash(tmp["class"].values)
            c_k = max(t_dic.items(), key=operator.itemgetter(1))[0]
            T.children[key] = self.ID3(tmp, e, c_k, list_A)
        return T

```

### C4.5算法

C4.5算法和ID3算法相似，只不过前者用的特征选择的标准是信息增益比，后者是信息增益，这里不过多阐述。

## 决策树的剪枝

为了防止得到的决策树模型过拟合，提高模型的泛化能力，我们需要对得到的决策树进行剪枝。  
设树T的叶结点个数为$|T|$，t是树T的叶结点，该结点上有$N_t$个样本点，其中k类的样本点有
$N_{tk}$个，$k = 1, 2, ..., K$，$H_t(T)$为叶结点t上的经验熵，$\alpha ≥ 0$为平衡
预测误差和模型复杂度的一个参数，决策树学习的损失函数可以定义为：

$$C_{\alpha} = \sum_{t=1}^{|T|}N_tH_t(T) + \alpha |T|$$

其中经验熵为：

$$H_t(T) = -\sum\frac{N_{tk}}{N_t}log\frac{N_{tk}}{N_t}$$

将定义式的第一项计为：

$$C(T) = \sum_{t=1}^{|T|}N_tH_t(T) = -\sum_{t=1}^{|T|}\sum_{k=1}^KN_{tk}log\frac{N_{tk}}{N_t}$$

这时有：

$$C_{\alpha}(T) = C(T) + \alpha |T|$$

第一项$C(T)$代表着模型的预测误差，$|T|$时叶结点的数量，也代表着模型的复杂度，$\alpha$
是平衡这两者的一个参数。

### 剪枝算法

输入：生成算法得到的整个树T，参数$\alpha$  
输出：剪枝之后的子树$T_{\alpha}$。

* 计算每个结点的经验熵
* 递归地从子树的叶结点向上回缩，设一组叶结点回缩到其父结点之前于之后的整体树分别为$T_B$和$T_A$，其对应的损失函数分别是$C_{\alpha}(T_B)$和$C_{\alpha}(T_A)$，如果$C_{\alpha}(T_A) ≤ C_{\alpha}(T_B)$，则进行剪枝（剪掉这个父结点所有的子树），即将父结点变为新的叶结点。
* 返回（2），直至回溯到根节点，得到损失函数最小的子树$T_{\alpha}$

剪枝算法的python代码：

```
def forecast_err(self, T, Data_set):#得到预测误差
        if T == None:
            return 0
        if T.engein == "class":
            dic = self.hash(Data_set["class"].values)
            # print(dic)
            s = sum(dic.values())
            H = 0
            for key, val in dic.items():
                H -= val/s * np.log(val/s)
            return H
        err = 0
        for key, val in T.children.items():
            tmp = pd.DataFrame(columns=Data_set.columns)
            for i in range(Data_set.shape[0]):
                if Data_set.loc[i, T.engein] == key:
                    tmp = tmp.append(Data_set.loc[i, :], ignore_index=True)
            err += self.forecast_err(val, tmp)
        return err

    def get_loss(self, T, alpha, Data_set):#得到损失函数
        leaves = self.get_leave(T)
        return self.forecast_err(T, Data_set) + alpha * len(leaves)

    def pruning(self, T, root, Data_set):#剪枝函数
        if T == None or T.engein == "class":
            return
        for key, val in T.children.items():#对当前节点剪枝前先递归地对子节点剪枝
            self.pruning(val, root, Data_set)
        before_loss = self.get_loss(root, 0.1, Data_set)
        print("before loss", before_loss)
        before_engein = T.engein
        T.engein = "class" #将标签设置为class做懒惰处理，
        after_loss = self.get_loss(root, 0.1, Data_set)
        print("after loss", after_loss)
        if after_loss < before_loss:
            for key in T.children:
                tmp = T.children[key]
                T.children[key] = None
                del tmp
        else:
            T.engein = before_engein

```


## 总结

关于CART（回归和分类树）在下一节






