
#%%
from sklearn.datasets import load_boston


#%%
data = load_boston()


#%%
x, y= data['data'],data['target']


#%%
x[0]


#%%
y[0]


#%%
data['feature_names']


#%%
data['DESCR']


#%%
x[:,0]


#%%
import matplotlib.pyplot as plt
plt.scatter(x[:, 5], y)


#%%
room_num = x[:,5]


#%%
price = y


#%%
import random
import numpy as np


#%%
def func (age, k ,b):
    return age*k+b


#%%
##def loss(y, yhat):
    ##return np.mean(np.abs(y-yhat))
def loss(y, yhat):
    return sum((y_i - yhat_i)**2 for y_i,yhat_i in zip(list(y),list(yhat)))/len(list(y))

#%%
min_erroe_rate=float('inf')
losses = []
loop_times=10000
lossed =[]


#%%
change_diection=[
    (+1, -1),
    (+1, +1),
    (-1, +1),
    (-1, -1),
]


#%%
k_hat = random.random()*20 -10
b_hat = random.random()*20 -10
best_k, best_b = k_hat, b_hat
best_direction = None


#%%
def derivate_k(y,yhat,x):
    abs_values=[1 if (y_i-yhat_i)>0 else -1 for y_i,yhat_i in zip (y,yhat) ]
    return np.mean([a * -x_i for a, x_i in zip(abs_values, x)])
def derivate_b(y, yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])


#%%
learing_rate = 1e-1 #learn rate 0.1


#%%
while loop_times > 0:
    k_delta = -1*learing_rate*derivate_k(price,func(room_num, k_hat, b_hat), room_num)
    b_delta = -1 * learing_rate * derivate_b(price, func(room_num, k_hat, b_hat))
    # k_delta_direction, b_delta_direction = direction
    #
    # k_delta = k_delta_direction * step()
    # b_delta = b_delta_direction * step()
    #
    # new_k = best_k + k_delta
    # new_b = best_b + b_delta
    k_hat +=k_delta
    b_hat +=b_hat
    estimated_price = func(room_num, k_hat, b_hat)
    error_rate = loss(y=price, yhat=estimated_price)
    print('loop == {}'.format(loop_times))
    print('f(age) = {} * age + {}, with error rate: {}'.format(best_k, best_b, error_rate))
    losses .append(error_rate)
    loop_times -= 1
plt.plot(range(len(losses)), losses)
plt.show()


#%%



#%%



#%%



#%%



#%%



