import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

# style.use("dark_background") # search for mpl style sheets, lots of customization

# Scatter Plots
""" X_data = np.random.random(1000) * 100
Y_data = np.random.random(1000) * 100

plt.scatter(X_data, Y_data, c="#00f", s=150, marker="*", alpha=0.3)
plt.show() """

# Line and Bar Charts
""" x = ["C++", "C#", "Python", "Java", "Go"]
y = [20, 50, 140, 1, 45]
plt.bar(x, y, color="r", align="edge", width=.5, edgecolor="green")
plt.show()

years = [2006 + x for x in range(16)] # auto fills a bunch of years
weights = [80, 83, 84, 85, 86, 82, 81, 79, 83, 80, 
           82, 82, 83, 81, 80, 79]

plt.plot(years, weights, c="g", lw=3, linestyle="--")
plt.show()  """

# Histograms
""" ages = np.random.normal(20, 1.5, 1000) # normal dist., mean of 20, std. dev. of 1.5
# plt.hist(ages, bins=[ages.min(), 18, 21, ages.max()])
plt.hist(ages, bins=20, cumulative=True) # how many people are at a certain age or below
plt.show()
 """

# Pie Charts
""" langs = ["Python", "C++", "Java", "C#", "Go"]
votes = [50, 24, 14, 6, 17]
explodes = [0, 0, 0, 0.2, 0]

plt.pie(votes, labels=langs, explode=explodes, autopct="%.2f%%", pctdistance=1.4, startangle=90)
plt.show() """

# Box Plots
# heights = np.random.normal(172, 8, 300)
# plt.boxplot(heights)
""" 
first = np.linspace(0, 10, 25)
second = np.linspace(10, 200, 25)
third = np.linspace(200, 210, 25)
fourth = np.linspace(210, 230, 25)

data = np.concatenate([first, second, third, fourth])
plt.boxplot(data)
plt.show() """

# Customization of plots 
""" years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
income = [55, 56, 62, 61, 72, 72, 73, 75]

income_ticks = list(range(50, 81, 2))

plt.plot(years, income)
plt.title("Income of Case (in USD)", fontsize=20)
plt.xlabel("Year")
plt.ylabel("Yearly Income in USD")
plt.yticks(income_ticks, [f"{x}k USD" for x in income_ticks])

plt.show() """

# Multiple Plots - line graph
""" stock_a = [100, 102, 99, 101, 101, 100, 102]
stock_b = [90, 95, 102, 104, 105, 103, 109]
stock_c = [110, 115, 100, 105, 100, 98, 95]

plt.plot(stock_a, label="Company1")
plt.plot(stock_b, label="Company2")
plt.plot(stock_c, label="Company3")

plt.legend(loc="lower center")

plt.show() """

# Multiple Pie Charts
""" votes = [10,2,5,16,22]
people = ["A","B","C","D","E"]

plt.pie(votes, labels=None)
plt.legend(labels=people)
plt.show()  """

# Multiple Figures 
""" x1, y1 = np.random.random(100), np.random.random(100)
x2, y2 = np.arange(100), np.random.random(100)

plt.figure(1)
plt.scatter(x1, y1)
plt.figure(2)
plt.plot(x2, y2)
plt.show() """

# Subplots
""" x = np.arange(100)

fig, axs = plt.subplots(2, 2) # 4 subplots in one figure

axs[0,0].plot(x, np.sin(x))
axs[0,0].set_title("Sine Wave")

axs[0,1].plot(x, np.cos(x))
axs[0,1].set_title("Cosine Wave")

axs[1,0].plot(x, np.random.random(100))
axs[1,0].set_title("Random Function")

axs[1,1].plot(x, np.log(x))
axs[1,1].set_title("Log Function")
axs[1,1].set_xlabel("TEST")

fig.suptitle("Four Plots")

plt.tight_layout() # structures graphs so there is no overlap

# Exporting Graphs
plt.savefig("fourplots.png", dpi=300, transparent=False, bbox_inches="tight") """

# 3D Plots
""" ax = plt.axes(projection="3d")

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

X, Y = np.meshgrid(x, y)

Z = np.sin(X) * np.cos(Y)

ax.plot_surface(X, Y, Z, cmap="Spectral")
ax.set_title("3D Plot")
plt.show() """


# Animated Graph
""" heads_tails = [0,0]

for _ in range(100000):
    heads_tails[random.randint(0, 1)] += 1
    plt.bar(["Heads", "Tails"], heads_tails, color=["red", "blue"])
    plt.pause(0.001)
plt.show() """