import numpy as np
import matplotlib.pyplot as plt

para_dct = {}
para_best = {}
score = [0, 1, 2, 3, 4]

score_best = [0,0,0,0,0]
score_dct = [0,0,0,0,0]
score_ground_truth = [0,0,0,0,0]

# from dct 
with open('predictions/sst2and6/stsonly-sst-dev-output.csv', 'r') as file:
   count = 0
   for line in file:
        if count == 0:
            count += 1
            continue
        line = line.strip()
        line = line.split(',')

        # para_best[line[0].rstrip()] = float(line[1])
        score_dct[int(float(line[1]))]+= 1

file.close()
print(score_dct)

# from ground truth 
with open('data/ids-sst-dev.csv', 'r') as file:
   count = 0
   for line in file:
        if count == 0:
            count += 1
            continue
        line = line.strip()
        line = line.split('\t')
        # print(line)
        # para_best[line[0].rstrip()] = float(line[1])
        score_ground_truth[int(float(line[3]))]+= 1

file.close()
print(score_ground_truth)


# from best performance model 
with open('predictions/sst2and6/2sst-dev-output.csv', 'r') as file:
   count = 0
   for line in file:
        if count == 0:
            count += 1
            continue
        line = line.strip()
        line = line.split(',')

        # para_best[line[0].rstrip()] = float(line[1])
        score_best[int(float(line[1]))]+= 1

file.close()
print(score_best)


# Data sets
data1 = score_ground_truth
data2 = score_best
data3 = score_dct

# X locations for the groups
ind = np.arange(len(data1))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(ind - width, data1, width, label='Ground Truth')
bars2 = ax.bar(ind, data2, width, label='Model (2)')
bars3 = ax.bar(ind + width, data3, width, label='Model (6)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of predictions')
ax.set_xlabel('Scores')
ax.set_title('Scores by predictions and model')
ax.set_xticks(ind)
ax.set_xticklabels(['0', '1', '2', '3', '4'])
ax.legend()

# Function to add labels on top of each bar
def autolabel(bars):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call the function for each bar chart
autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

# plt.show()
plt.savefig('predictions/sst2and6/eval26.png')