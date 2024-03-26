# Notes

## Mar 26, 2024
An NCP model of rank $d$ consists of $2d$ functions $(u_{i}, v_{i})_{i=1}^{d}$ and $d$ positive numbers $(\sigma_{i})_{i = 1}^{d}$, and it is written as
$$
\mathsf{G} = \sum_{i= 1}^{d} \sigma_{i} u_{i}\otimes v_{i},
$$
NCPs are a class of linear operator models constructed as a sum of rank-one learned components. The notation we use is a nudge to interpret $u := \{ u_{i} : 1 \leq i \leq d\}$, $\sigma$, and $v$ as the truncated SVD of our target operator, which $\mathsf{G}$ aims to approximate. And indeed, the loss function $\mathcal{L}_{\gamma}$ is minimized by the target operator's truncated SVD.

_However,_ in terms of the applicability of this model, we need to ensure
1. That the cross covariances $\int \mu(dx) u_{i}(x) \otimes u_{j}(x) = \delta_{ij}$ (same for $v$), meaning that the functions are correctly orthonormalized.
2. That $\text{span}(u) \perp 1$ (and the same for $v$), meaning that we are learning the _deflated_ conditional expectation operator. 
When 1. holds, we have a bona fide SVD. It may not approximate tightly the target operator, but it is an SVD nonetheless. Point 2. is specific to the problem of learning deflated operators, but this requirement can be elegantly included in 1. by simply ensuring
$$
\int \mu(dx) (u_{i}(x) - 1 \langle u_{i}, 1\rangle) \otimes (u_{j}(x) - 1 \langle u_{i}, 1\rangle) = \delta_{ij}.
$$
Notice that the above equation is simply the centered cross-covariance, as $1 \langle u_{i}, 1\rangle = \mathbb{E}_{\mu}[u_{i}]$. 
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
