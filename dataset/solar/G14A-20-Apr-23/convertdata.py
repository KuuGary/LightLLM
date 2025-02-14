import pandas as pd

# Path to your input text file and desired output CSV file
input_file_path = 'data-1'
output_file_path = 'solar3.csv'

# Initialize a list to store the combined records
combined_records = []

# Temporary storage for each group of four lines
temp_record = []

# Open and read the input text file
with open(input_file_path, 'r') as file:
    for line in file:
        # Split each line by commas
        line_parts = line.strip().split(',')
        # If this line is a data line (not a solar line)
        if len(line_parts) > 2:
            # Add the date and data parts to the temp record
            temp_record = [line_parts[0]] + line_parts[1:]
        else:
            # If it's a solar line, extract the solar value
            solar_value = line_parts[1].split(':')[1]
            # Append the solar value to the temp record
            temp_record.append(solar_value)
            # If we have collected all parts of the record
            if 'solar 3' in line_parts[1]:
                # Add the complete record to the combined records list
                combined_records.append(temp_record)
                # Reset the temp record for the next group of lines
                temp_record = []

# Convert the combined records into a DataFrame
df = pd.DataFrame(combined_records, columns=['date'] + [f'wavelength{i+1}' for i in range(18)] + ['solar1', 'solar2', 'solar3'])

# Convert the 'date' column to a datetime format (optional, for consistency with your example)
df['date'] = pd.to_datetime(df['date'])

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False)

print(" Data saved to:", output_file_path)
