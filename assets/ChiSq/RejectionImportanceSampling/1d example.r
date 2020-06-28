# Video - 1D - Numerical, Rejection Sampl., Importance Sampl.

f = function(x) {
  return (exp(-x^2))
}


# Numerical
dx = 0.01
x = seq(-5, 5, dx)
f.x = f(x)
sum(f.x*dx)
sqrt(pi)


# Rejection 
# Uniform(-5,5)
g.x = runif(1000, -5, 5)
c = 10 # how much (the highest point of) f.x is bigger than g.x
t = f(g.x)/(c*dunif(g.x, -5, 5))
u = runif(1000)
indices = (u < t)
# hist(g.x[indices])
ratio = sum(indices)/length(indices)
integral.f = ratio * c
integral.f 

# Normal
g.x = rnorm(1000)
c = 2.55
t = f(g.x)/(c*dnorm(g.x))
u = runif(1000)
indices = (u < t)
ratio = sum(indices)/length(indices)
integral.f = ratio * c
integral.f 


# Importance
g.x = rnorm(1000000)
w = f(g.x)/dnorm(g.x)
mean(w)

