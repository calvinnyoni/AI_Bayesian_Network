import math
from pdb import line_prefix
from random import randint, random

filename = "data\heart.csv"

with open(filename) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

training_data = [lines[0]] 
test_data = [lines[0]]

test_size = 10
random_line_indices = []

i = 0
while (i < test_size):
    random_line_indices.append(randint(1, len(lines) - 1))
    i = i + 1

i = 0
while (i < (len(lines) - 1)):
    if i in random_line_indices:
        test_data.append(lines[i]+"\n")
    else:
        training_data.append(lines[i]+"\n")
    i = i + 1

with open("data/test_data.csv", "w+") as file:
    for line in test_data:
        file.write(line)

with open("data/training_data.csv", "w+") as file:
    for line in training_data:
        file.write(line)