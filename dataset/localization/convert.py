import numpy as np
import pandas as pd

datafile = "data-night5.npy"
labelfile = "label-night5.npy"

data = np.load(datafile)
label = np.load(labelfile)

#--------------------- Spec--------------------
# reshaped_data = data.reshape(data.shape[0], -1)

# combined_data = np.hstack((reshaped_data, label.reshape(-1, 1)))

# sensor_columns = [f's{i}_w{j}' for i in range(1, 28) for j in range(1, 19)]
# columns = sensor_columns + ['loc']

# df = pd.DataFrame(combined_data, columns=columns)

# df = pd.DataFrame(combined_data,columns=columns)

# csv_file = "output.csv"
# df.to_csv(csv_file, index=False)

# print(f"Data and labels have been saved to {csv_file}")




#--------------------- Intensity--------------------
summed_data = data.sum(axis=2)


combined_data = np.hstack((summed_data, label.reshape(-1, 1)))

sensor_columns = [f's{i}' for i in range(1, 28)]
columns = sensor_columns + ['loc']

df = pd.DataFrame(combined_data, columns=columns)

csv_file = "output_summed.csv"
df.to_csv(csv_file, index=False)

print(f"Data and labels have been saved to {csv_file}")

