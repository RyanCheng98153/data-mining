import matplotlib.pyplot as plt

#   - epochs = 10000, default setting
# | train / valid | Train RMSE | Validation RMSE | Epochs |
# | ------------- | ---------- | --------------- | ------ |
# | 8 : 4         | 4.0805     | 3.7969          | 10000  |
# | 9 : 3         | 4.0583     | 3.7901          | 10000  |
# | 10 : 2        | 4.0854     | 3.4406          | 10000  |
# | 11 : 1        | 4.0202     | 3.5780          | 10000  |
#   - epochs = 10000, patience = 1000, standardize
# | train / valid | Train RMSE | Validation RMSE | Epochs (Early-Exit) |
# | ------------- | ---------- | --------------- | ------------------- |
# | 8 : 4         | 3.8426     | 3.7469          | 2907                |
# | 9 : 3         | 3.8316     | 3.7281          | 1901                |
# | 10 : 2        | 3.8712     | 3.3289          | 5507                |
# | 11 : 1        | 3.8140     | 3.4201          | 6728                |

train_valid_settings = [
    (8, 4, 'train 8 : valid 4'),
    (9, 3, '9 : 3'),
    (10, 2, '10 : 2'),
    (11, 1, '11 : 1'),
]

train_rmses = [3.8426, 3.8316, 3.8712, 3.8140]
valid_rmses = [3.7469, 3.7281, 3.3289, 3.4201]
epochs_run = [2907, 1901, 5507, 6728]

x = range(len(train_valid_settings))
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
bars1 = ax.bar([i - width/2 for i in x], train_rmses, width, label='Train RMSE')
bars2 = ax.bar([i + width/2 for i in x], valid_rmses, width, label='Validation RMSE')
ax.set_xlabel('Train/Valid Split')
ax.set_ylabel("RMSE")
ax.set_title('Train and Valid RMSE')
ax.set_xticks(x)
ax.set_xticklabels([s[2] for s in train_valid_settings])
ax.legend()
ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)
# set the y-axis limit
ax.set_ylim(0, max(max(train_rmses), max(valid_rmses)) + 1)
plt.savefig('train_valid_rmses.png')

fig, ax = plt.subplots()
bars = ax.bar(x, epochs_run, width, label='Epochs Run')
ax.set_xlabel('Train/Valid Split')
ax.set_ylabel('Epochs Run')
ax.set_title("Train / Valid Split's Epochs Run")
ax.set_xticks(x)
ax.set_xticklabels([s[2] for s in train_valid_settings])
ax.bar_label(bars, padding=3)
# set the y-axis limit
ax.set_ylim(0, max(epochs_run) + 1000)
plt.savefig('train_valid_epochs_run.png')
