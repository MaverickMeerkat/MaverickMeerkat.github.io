---
layout: post
title: Cross Entropy
subtitle: 
tags: [Math, Loss Functions, ML, KL-Divergence]
comments: true
categories: Math, Machine Learning
published: true
usemathjax: true
mathjax: true
---

Inline math $f(x)=x^2$

test 12 3

$$
\begin{aligned}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{aligned}
$$

Cross-Entropy can be thought of as a measure of how much one distribution differs from a 2nd distribution, which is the ground truth. It is one of the two terms in the KL-Divergence, which is also a sort-of measure of (non-symmetric) distance between two distribution, the other term being the ground-truth entropy. But in machine learning, 


\\( 1/x^{2} \\)