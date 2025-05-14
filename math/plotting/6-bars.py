#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))  # rows: fruits, columns: people

people = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']

x = np.arange(len(people))
bottom = np.zeros(len(people))

for i in range(len(fruit)):
    plt.bar(x, fruit[i], bottom=bottom, color=colors[i], label=fruit_labels[i], width=0.5)
    bottom += fruit[i]  # stack the bars

plt.xticks(x, people)
plt.yticks(np.arange(0, 81, 10))
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.legend()
plt.ylim(0, 80)
plt.show()

