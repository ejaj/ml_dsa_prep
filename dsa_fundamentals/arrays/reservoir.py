import random
random.seed(42) 

stream = list(range(1, 10001))

reservior = []
k = 100
for i, val in enumerate(stream):
   if i < k:
      reservior.append(val)
   else:
      rn = random.randint(0, i)
      if rn<k:
         reservior[rn] = val

avg = sum(reservior) / len(reservior)
print("Final Reservoir:", reservior)
print("Estimated Average:", avg)
