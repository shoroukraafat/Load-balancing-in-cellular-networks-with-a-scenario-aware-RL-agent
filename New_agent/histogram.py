import csv
import matplotlib.pyplot as plt
import numpy as np
import re

# csv file name

#filename1 = "R1_testTD3_normal_40users.csv"
filename1 = "R1_test_lampda0TD3_sec_test_20episodes.csv"
#filename1 = "R1_test_lampda5_10episodes.csv"

# initializing the titles and rows list


fields2 = []
rows2 = []
rows = {}  # Dictionary to store rows  
      
with open(filename1, mode='r', encoding='utf-8') as file:
    csvreader = csv.reader(file, delimiter=';')
    for i, row in enumerate(csvreader):
        rows[f'row{i}'] = row

# Accessing rows from the dictionary      
needed_row = rows['row4']      
#print(type(rows['row27']))

mimo_array = [int(item.strip('[]')) for item in needed_row]
#data = [float(s.strip('[]')) for s in rows['row2']]   

#print(data)
'''
#string_data = needed_row[0]
#print(string_data)

# Remove the brackets and split the string into individual number strings
number_strings = string_data.replace('[', '').replace(']', '').split(',')

# Convert the list of number strings to integers
cleaned_list = [int(num) for num in number_strings]

# Convert the cleaned list to a NumPy array with integer data type
mimo_array = np.array(cleaned_list, dtype=int)

#mimo_array = np.array(data, dtype=int)

print(len(mimo_array))
'''
# Function to plot histogram
def plot_histogram(mimo_array, bins=10, title="Histogram", xlabel="Value", ylabel="Frequency"):
    plt.hist(mimo_array, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Main function
if __name__ == "__main__":
    # Path to the CSV file
    #file_path = 'data.csv'  # Replace with your CSV file path

    # Name of the column to plot
    #column_name = 'column_name'  # Replace with your column name

    # Read data from the specified column
    #data = read_column_data(file_path, column_name)

    # Plot the histogram
    plot_histogram(mimo_array, bins=10, title=f"Histogram of MIMO frequency", ylabel="Frequency")



