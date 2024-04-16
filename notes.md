# Notes

## Apr 13, 2024
A Neural Conditional Probability model (NCP) of rank $d$ is a probabilistic model describing a pair of random variables $(X, Y)$, where $X \sim \mu$ and $Y \sim \nu$. By letting $\mathsf{E}: L^{2}_{\nu} \to L^{2}_{\mu}$ be the conditional expectation operator of $Y$ given $X$
$$f \mapsto \mathbb{E}\left[f(Y) \mid X = x\right] =: (\mathsf{E}f)(x),$$
we aim to approximate the _deflated_ conditional expectation $\mathsf{D} := \mathsf{E} - 1_{\mu} \otimes 1_{\nu}$. By letting the class of rank-$d$ NCP models be 
$$\text{NCP}_{d} := \left\{(u_{i}, v_{i}, \sigma_{i}) \in L^{2}_{\mu} \times L^{2}_{\nu} \times [0, 1] : i \in [d]\right\},$$
every element of $\text{NCP}_{d}$ is naturally associated to a rank-$d$ linear operator
$$\mathsf{M} := \sum_{i=1}^{d} \sigma_{i} \left( u_{i} \otimes v_{i}\right).$$

After a leap of faith in which you trust me that the loss function we proposed to train the NCP gives us a good approximation of $\mathsf{D}$, let me denote $\mathsf{M}$ the linear operator resulting from an ERM procedure and for which $\mathsf{M} \simeq \mathsf{D}$.

## Computation of conditional statistics via NCPs
> [PIE] This is not going to be rigorous: I'll make use of Dirac's deltas to make the derivation direct, although they are not functions in any $L^{2}$. We'll fix everything in the draft.

### Conditional probability function $p(y | x)$

By letting $\delta_{y}$ be Dirac's delta centered at $y$. Then
$$
\begin{split}
p(y | x) = (\mathsf{E}\delta_{y})(x) &\simeq \left((\mathsf{M} + 1_{\mu} \otimes 1_{\nu})\delta_{y}\right)(x) \\
&=\left[\sum_{i = 1}^{d} \sigma_{i} u_{i}(x) \int_{\mathcal{Y}} v_{i}(y^{\prime})\delta(y - y^{\prime})\nu(dy^{\prime})\right] + \int 1_{\nu}(y^{\prime})\delta(y - y^{\prime})\nu(dy^{\prime}) \\
&=\left(1 + \sum_{i = 1}^{d} \sigma_{i} u_{i}(x) v_{i}(y)\right)p_{\nu}(y) \\
\end{split}
$$
Where $p_{\nu}(y)$ is the probability density function of $Y$, that is $\frac{d\nu}{dy}$, the Radon-Nikodym of $\nu$ with respect to the Lebesgue measure on $\mathcal{Y}$ (which we assume to exist). 

If we happen _not_ to know $p_{\nu}(y)$, we can estimate it from data e.g. by
1. Kernel Density Estimation
2. Histograms
3. ???

### Joint probability density function $p(x, y)$

We notice that $p(x, y) = p(y | x)p_{\mu}(x)$, which from the previous section gives
$$
\begin{split}
p(x, y) \simeq \left(1 + \sum_{i = 1}^{d} \sigma_{i} u_{i}(x) v_{i}(y) \right)p_{\nu}(y)p_{\mu}(x) \\
\end{split}
$$

### Conditional CDF 
Let $\mathcal{Y} = \mathbb{R}$. The conditional Cumulative Density Function of $Y$ given $X = x$ is by definition
$$
\begin{split}
F(t; x) = (\mathsf{E}1_{y \leq t})(x) &\simeq \left((\mathsf{M} + 1_{\mu} \otimes 1_{\nu})1_{y \leq t}\right)(x) \\
&=\left[\sum_{i = 1}^{d} \sigma_{i} u_{i}(x) \int_{-\infty}^{t} v_{i}(y)\nu(dy)\right] + \int_{-\infty}^{t}1_{\nu}(y)\nu(dy) \\
&=F_{\nu}(t) + \sum_{i = 1}^{d} \sigma_{i} u_{i}(x) \int_{-\infty}^{t} v_{i}(y)p_{\nu}(y)dy \\
\end{split}
$$
Where $F_{\nu}(t)$ is the _unconditional_ CDF of $Y \sim \nu$.
To evaluate the CDF, we need to compute an integral with respect to $\nu$ which we might not know. We can estimate it from data via e.g Montecarlo integration.

### Conditional moments
You got the drill
$$
\begin{split}
(\mathsf{E}|y|^{t})(x) &\simeq \left((\mathsf{M} + 1_{\mu} \otimes 1_{\nu})|y|^{t}\right)(x) \\
&=\left[\sum_{i = 1}^{d} \sigma_{i} u_{i}(x) \int_{-\infty}^{t} v_{i}(y)|y|^{t}\nu(dy)\right] + \int_{-\infty}^{t}|y|^{t}\nu(dy) \\
&=\mathbb{E}_{\nu}[|y|^{t}] + \sum_{i = 1}^{d} \sigma_{i} u_{i}(x) \int_{-\infty}^{t} v_{i}(y)|y|^{t}p_{\nu}(y)dy \\
\end{split}
$$

## Random Thoughts
To avoid any sorts of weird results we should make sure that for all $(x, y) \in \mathcal{X} \times \mathcal{Y}$.
- $1 + \sum_{i = 1}^{d} \sigma_{i} u_{i}(x)v_{i}(y) \geq 0$ 
- $\sum_{i = 1}^{d}\int_{\mathcal{X} \times \mathcal{Y}}  \sigma_{i} u_{i}(x)v_{i}(y)p_{\mu}(x)p_{\nu}(y) dxdy = 0$

## Mar 26, 2024
An NCP model of rank $d$ consists of $2d$ functions $(u_{i}, v_{i})_ {i = 1}^{d}$ and $d$ positive numbers $(\sigma_{i})_{i = 1}^{d}$, and it is written as

$$
\mathsf{G} = \sum_{i= 1}^{d} \sigma_{i} u_{i}\otimes v_{i},
$$

NCPs are a class of linear operator models constructed as a sum of rank-one learned components. The notation we use is a nudge to interpret $u := \\{ u_{i} : 1 \leq i \leq d\\}$, $\sigma$, and $v$ as the truncated SVD of our target operator, which $\mathsf{G}$ aims to approximate. And indeed, the loss function $\mathcal{L}_{\gamma}$ is minimized by the target operator's truncated SVD.

_However,_ in terms of the applicability of this model, we need to ensure
1. That the covariances satisfy $\int \mu(dx) u_{i}(x) \otimes u_{j}(x) = \delta_{ij}$  (same for $v$), meaning that the functions are correctly orthonormalized.
2. That $\text{span}(u) \perp 1$ (and the same for $v$), meaning that we are learning the _deflated_ conditional expectation operator. 
When 1. holds, we have a bona fide SVD. It may not approximate tightly the target operator, but it is an SVD nonetheless. Point 2. is specific to the problem of learning deflated operators, but this requirement can be elegantly included in 1. by simply ensuring

$$
\int \mu(dx) (u_{i}(x) - 1 \langle u_{i}, 1\rangle) \otimes (u_{j}(x) - 1 \langle u_{i}, 1\rangle) = \delta_{ij}.
$$

Notice that the above equation is simply the centered cross-covariance, as $1 \langle u_{i}, 1\rangle = \mathbb{E}_ {\mu}[u_{i}]$. 
**The orthogonalization of $u$ and $v$ is a pre-requisite.** Every downstream relation between NCPs and the predicted conditional probabilities/expectations assumes orthonormal $u$ and $v$.
### How to do that?
Two options:
1. **Optimization**: add aggressive metric regularizations or batch whitening schemes so that at the end of the training loop $u$ and $v$ are orthonormal. 
2. **Post-Processing**:  minimize $\mathcal{L}_{\gamma}$ up to convergence and then post-process $u$ and $v$ by centering + scaling + deflating. 

Our current understanding is that 1. does not work. But why? The optimization is going bananas, and we don't understand why. Here are some things I would like to get checked:
1. Are we converging to a spurious stationary point? If yes, what does it look like? Meaning, 
	1. What is the value of $u$ and $v$ on the training set (e.g., are they constant?)
	2. How are the covariances _and_ centered cross covariances of $u$ and $v$ compared to the identity?
2. If we are not converging to a spurious stationary point, it means we are not converging. What about
	1. Decreasing the learning rate?
	2. What are the values of the activations and the gradients at each intermediate layer of the MLP? What are their means and covariances? Are their values exploding or shrinking as we move from layer to layer? This should be checked at initialization (maybe we have a poor init) and along the training loop.
4. If we need to decrease the learning rate a lot, the network will not be well conditioned. Adding residual connections might help.
5. How does the minimal value attained by the loss scales by increasing the model size? 
6. On a well-trained model, I expect training and validation losses to be quite similar. Suppose the training and validation sets are both well representative of the statistics of the problem. In that case, the training and validation losses are empirical estimators of the total correlation captured by the NCP, and they should be pretty close.

Likely enough, I don't expect optimization alone to give us orthogonal features, and a (hopefully tiny) amount of post-processing will be needed. Because of this, in the draft, we also need to devise a fast post-processing scheme (it is unfeasible to recompute the `svd` at each evaluation of the model)

We should work hard to make our model fail so that we learn its soft spots and improve them. 
