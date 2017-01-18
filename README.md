# State Space Model

We work with the following model:
  - **x[t+1] = f(x[t]) + v[t]**
  - **y[t+1] = g(x[t]) + w[t]**
  - The states **x[.]** are vectors of dimention **p** and are **unobserved**.
  - The outputs **y[.]** are vectors of dimention **q** and are **observed**.
  - The functions **f** and **g** are non linear functions.
  - The sequences **v[.]** and **w[.]** are sequence of independent identically distributed [multivariate Gaussian noises](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).
  
that can be represented with the following [graphical model](https://en.wikipedia.org/wiki/Graphical_model) :
![model](https://github.com/louity/state-space-model/raw/master/rapport/images/graph.png)

The purpose is to do:
  - **inference** : compute the probabilty of the states **x[.]** given the outputs **y[.]**
  - **learning** : given the outputs **y[.]** learn the functions **f** and **g** with an [EM algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm).

For a good mathematical presentation of the state space model, have a look at the [report]()

## Inference

The inference techniques implemented in the state space model are all based on the [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter):
  - **Kalman Filter**
  - **Kalman Smoother**
  - **Extended Kalman Filter**
  - **Extended Kalman Smoother**
  - **Unscented Kalman Filter** Not implemented yet

## Learning
