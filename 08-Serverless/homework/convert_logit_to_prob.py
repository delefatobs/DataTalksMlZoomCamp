import math

# Replace this with your model output
logit = -0.20132342

# Convert raw logit to probability
prob = 1 / (1 + math.exp(-logit))

print("Logit:", logit)
print("Probability:", prob)
