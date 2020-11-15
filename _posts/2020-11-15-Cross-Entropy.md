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


test 12 3

Cross-Entropy can be thought of as a measure of how much one distribution differs from a 2nd distribution, which is the ground truth. It is one of the two terms in the KL-Divergence, which is also a sort-of measure of (non-symmetric) distance between two distribution, the other term being the ground-truth entropy. 

$$D_{KL}(P||Q) = - \sum_x P(x)\log{Q(x)} + \sum_x P(x)\log{P(x)} = H(P,Q) - H(P)
$$

$H(P)$ being the entropy, and $H(P,Q)$ being the cross-entropy.

But in machine learning, the data distribution is usually given, so you cannot change the data entropy. Instead you focus on the cross entropy term, trying to minimize it as much as possible. 

