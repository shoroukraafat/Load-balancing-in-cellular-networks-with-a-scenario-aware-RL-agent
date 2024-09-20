import csv
import matplotlib.pyplot as plt
import numpy as np

# csv file name
filename = "Rewards_mimot_1718899414.csv"

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields))

# Choose a specific row to plot
# Example: Plotting the fifth row (index 4)
selected_row = rows[0]

# Convert row data to float values
numeric_data = np.array(selected_row, dtype=float)

# Reshape the array to have size (40, 250) - because 10,000 / 250 = 40
reshaped_data = numeric_data.reshape(-1, 249)

# Compute the mean of each 250-value segment
averages = np.mean(reshaped_data, axis=1)

# Plotting averages
plt.figure()
plt.plot(averages, marker='o')  # Using circle markers for each point
plt.title('Average Throughput per 250 Steps')
plt.xlabel('Episode (Each of 250 Steps)')
plt.ylabel('Average Value')
plt.grid(True)
plt.show()

