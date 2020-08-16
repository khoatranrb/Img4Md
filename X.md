* [Entropy](#####**1.-Entropy**)

In this blog, we will discuss about **Information theory**, about its concept and its mathematical nature. 

<p>
    <p>
    <img src='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/B/MLPR8/it.png'>
    <center><b>Fig1</b> <i>(Source: Wikipedia)</i></center>
</p>
</p>






##### **1. Entropy**

* Let consider a discrete random variable $x$. We wonder how much information is received when we observe a specific value for this variable. The amount of information can be viewed as the **degree of surprise** of $x$ or the **uncertainty** of variables.

* The intuition for entropy is that it is the average number of bits  required to represent or transmit an event drawn from the probability  distribution for the random variable.

* We use a quantity $h(x)$ as a measure for information content and the $h(x)$ must be a monotonic function of the probability distribution $p(x)$.

* If we have two events $x$ and $y$ that are unrelated, so $h(x,y)=h(x)+h(y)$. Cause $x$ and $y$ are independent, then $p(x,y)=p(x)p(y)$. From two relationships, we can show that:
  $$
  h(x)=-\log p(x)\tag1
  $$
  where the negative sign ensures that the value of $h(x)$ is positive or zero.

* The average amount of information of a random variable $X$ with distribution $p$:
  $$
  \H[X]=-\sum_{x\in X}p(x)\log p(x)\tag2
  $$
  called the **entropy** of the random variable $X$.

* **(A)** We can see that, $\H[X]$ achieves the maximum when all of $p(x)$ are equal.

* In general, we have entropy of continuous variable:
  $$
  \H[X]=-\int p(x)\log p(x)\text{d}x\tag3.
  $$

##### **2. Joint Entropy**

* **Joint entropy** is a measure of the uncertainty associated with a set of variables. It is the entropy of a joint probability distribution. With two distributions, it is defined by:

$$
\H[(X,Y)]=-\int\int p(x,y)\log p(x,y)\text{d}x\text{d}y\tag4.
$$



##### **3. Condition entropy**

* Consider a joint distribution $p(x,y)$. If a value of $x$ is already known, the average additional information needed to specify $y$ can be written as:
  $$
  \begin{align}
  \H[Y|X]&=-\int p(y|X)\log p(y|X)\text{d}y\tag5\\
  &=-\int\int p(y,x)\log p(y|x)\text{d}y\text{d}x\tag6.
  \end{align}
  $$
  In other hand, we can write:
  $$
  \begin{align}
  \H[\mathbf{Y|X}]&=-\int\int p(y,x)\log \frac{ p(y,x)}{p(x)}\text{d}y\text{d}x\tag6\\
  &=-\int\int p(y,x)\log p(y,x)\text{d}y\text{d}x+\int\int p(y,x)\log p(x)\text{d}y\text{d}x\tag7\\
  &=-\int\int p(y,x)\log p(y,x)\text{d}y\text{d}x+\int p(x)\log p(x)\text{d}x\tag8\\
  &=\H[(X,Y)]-\H[X]\tag{9}.
  \end{align}
  $$
  Thus, we have:
  $$
  \H[(X,Y)]=\H[Y|X]+\H[X]=\H[X|Y]+\H[Y]\tag{10}.
  $$



##### **4. Mutual information**

* Mutual  information  is  a  quantity  that  measures  a  relationship  between  two random variables that are sampled simultaneously. It is  defined as:
  $$
  \mathbb{I}[X;Y]=\int\int p(x,y)\log\frac{p(x,y)}{p(x)p(y)}\text{d}y\text{d}x\tag{11}
  $$

* We also have:
  $$
  \begin{align}
  \mathbb{I}[X;Y]&=\int\int p(x,y)\log p(x,y)\text{d}x\text{d}y-\int\int p(x,y)\log p(x)\text{d}x\text{d}y-\int\int p(x,y)\log p(y)\text{d}x\text{d}y\tag{12}\\
  &=\H[X]+\H[Y]-\H[(X,Y)]\tag{13}\\
  &=\H[X]-\H[X|Y]\tag{14}\\
  &=\H[Y]-\H[Y|X]\tag{15}\\
  &=\H[(X,Y)]-\H[X|Y]-\H[Y|X]\tag{16}.
  \end{align}
  $$

* **Mutual information** is a symmetry function, so it can be used as a metric, that measures the same of two distribution.

##### **5. Properties**

* $\H[X]\ge0$. It is equal when $p(x)=1$ where $x\in X$. $(17)$

  Similarity, **joint entropy**, **condition entropy** and **mutual information** are also positive or zero.

* From $(10)$, we have $\H[(X,Y)]\ge\max\{\H[X],\H[Y]\}$. $(18)$

* From $(13)$, we have $\H[(X,Y)]\le\H[X]+\H[Y]$. $(19)$

* From $(14)$ and $(15)$, we have $\H[X|Y]\le\H[X]$ and $\H[Y|X]\le\H[Y]$. $(20)$

* From $(12)$ and $(18)$, we have $\mathbb{I}[X;Y]\le\min\{\H[X],\H[Y]\}$. $(21)$

##### **Reference:**

* [1] <a href='https://en.wikipedia.org/wiki/Entropy' style='color:green'>Entropy | Wikipedia.</a>

* [2] <a href='https://en.wikipedia.org/wiki/Joint_entropy' style='color:green'>Joint Entropy | Wikipedia.</a>

* [3] <a href='https://en.wikipedia.org/wiki/Conditional_entropy' style='color:green'>Condition Entropy | Wikipedia.</a>

* [4] <a href='https://en.wikipedia.org/wiki/Mutual_information' style='color:green'>Mutual information | Wikipedia.</a>

* [5] <a href=' https://people.cs.umass.edu/~elm/Teaching/Docs/mutInf.pdf' style='color:green'>Entropy and Mutual Information | University of Massachusetts Amherst.</a>

* [6] <a href='https://machinelearningmastery.com/cross-entropy-for-machine-learning/' style="color:green">Cross-entropy for machine learning | Machine Learning Mastery</a>.

* [7] 1.6| Pattern Recognition and Machine Learning | C.M. Bishop.

* [8] 2.8.1 | Machine Learning A Probabilistic Perspective | K.P. Murphy

  

