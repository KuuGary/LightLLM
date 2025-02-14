import pandas as pd
import numpy as np

blue = [0.9, 0.7, 0.62, 0.57, 0.6, 0.7, 0.8, 0.9, 0.98, 0.95, 0.27, 0, 0, 0, 0.15]
green = [1.05, 0.93, 0.97, 0.99, 0.96, 0.97, 1, 0.97, 0.91, 0.82, 0.32, 0.05, 0, 0, 0.1]


csv_path = 'solar3.csv'

data = pd.read_csv(csv_path)

blue = np.array(blue)
green = np.array(green)

solar_blue_values = []
solar_green_values = []

for index, row in data.iterrows():
    wavelengths = row[[f'wavelength{i+1}' for i in range(15)]].values
    solar_blue = np.sum(wavelengths * blue)
    solar_green = np.sum(wavelengths * green)
    solar_blue_values.append(solar_blue)
    solar_green_values.append(solar_green)

solar_blue_values = np.array(solar_blue_values)
solar_green_values = np.array(solar_green_values)

actual_solar2 = data['solar2'].values
actual_solar1 = data['solar1'].values

nonzero_indices_solar2 = actual_solar2 != 0
nonzero_indices_solar1 = actual_solar1 != 0

# Calculate MAPE for solar_blue and solar2
mape_solar_blue = np.mean(np.abs((actual_solar2[nonzero_indices_solar2] - solar_blue_values[nonzero_indices_solar2]) / actual_solar2[nonzero_indices_solar2])) * 100

# Calculate MAPE for solar_green and solar1
mape_solar_green = np.mean(np.abs((actual_solar1[nonzero_indices_solar1] - solar_green_values[nonzero_indices_solar1]) / actual_solar1[nonzero_indices_solar1])) * 100


print(f"Average Green MAPE: {mape_solar_green}%")
print(f"Average Blue MAPE: {mape_solar_blue}%")