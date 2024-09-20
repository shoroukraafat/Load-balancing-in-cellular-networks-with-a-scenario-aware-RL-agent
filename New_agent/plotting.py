import csv
import matplotlib.pyplot as plt
import numpy as np

# csv file name
filename1 = "Rewards_PwCIO_6_Antennas.csv"
filename2 = "Rewards_t_1716135829.csv"

# initializing the titles and rows list
fields1 = []
rows1 = []
fields2 = []
rows2 = []

# reading csv file
with open(filename1, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields1 = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows1.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields1))

# Choose a specific row to plot
# Example: Plotting the fifth row (index 4)
selected_row1 = rows1[0]

# Convert row data to float values
numeric_data1 = np.array(selected_row1, dtype=float)

# Reshape the array to have size (40, 250) - because 10,000 / 250 = 40
reshaped_data1 = numeric_data1.reshape(-1, 250)

# Compute the mean of each 250-value segment
averages1 = np.mean(reshaped_data1, axis=1)

# reading csv file
with open(filename2, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields2 = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows2.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields2))

# Choose a specific row to plot
# Example: Plotting the fifth row (index 4)
selected_row2 = rows2[0]

# Convert row data to float values
numeric_data2 = np.array(selected_row2, dtype=float)

# Reshape the array to have size (40, 250) - because 10,000 / 250 = 40
reshaped_data2 = numeric_data2.reshape(-1, 249)

# Compute the mean of each 250-value segment
averages2 = np.mean(reshaped_data2, axis=1)


fig1, ax = plt.subplots()
ln1, = plt.plot(averages1, label='MIMO')
ln1, = plt.plot(averages2, label='orginal')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-small')
plt.xlabel("Episode")
plt.ylabel("Average overall throuput")
plt.title('Average Throughput per 250 Steps')
plt.show()
plt.savefig('TD3_0CIO_{}.png'.format(int(time.time())))


