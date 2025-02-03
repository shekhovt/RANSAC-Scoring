import math

# %%
for g in [0.01, 0.99]:
    for s in [0.1]: # /math.sqrt(2*math.pi)]:
        #t = -0.5*math.log(2*math.pi) + math.log(g/(1-g)) - math.log(sa)
        # t = math.log(g/(1-g))
        la = math.log(1/math.sqrt(2*math.pi)) + math.log(g/(1-g)) + math.log(1/s) -1/s
        print(math.exp(la))
# %%
