#%%
import torch as t

temp = t.tensor(27.0)
temp.requires_grad_()

init = t.zeros((1, 8, 8))
init[0, 4, 4] = 1

grid = init.detach().clone()

# %%
padded_batch = t.nn.ReflectionPad2d(1)(grid).float()
prob = t.exp(-t.sum(padded_batch) / temp)
print(prob)
logits = t.log(t.cat([prob.unsqueeze(0), (1 - prob).unsqueeze(0)]))

one_hot = t.nn.functional.gumbel_softmax(logits, hard=True)

print(one_hot)

grid[0, 5, 5] += one_hot[0]

print(grid)

dist = (t.sum(grid) - 1) ** 2

# %%
print(t.autograd.grad(dist, temp))

# %%
