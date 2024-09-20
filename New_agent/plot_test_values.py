import csv
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.signal import savgol_filter

# csv file name
#filename1 = "R1_testTD3_1500step.csv"
#filename1 = "R1_testTD3_250.csv"
#filename1 = "R1_testTD3_20episodes.csv"
#filename1 = "R1_testTD3_3Episodes.csv"
#filename1 = "R1_test_lampda0TD3_step1723417027.csv"
filename1 = "R1_test_lampda0TD3_sec_test_20episodes.csv"


# initializing the titles and rows list
fields1 = []
rows1 = []
fields2 = []
rows2 = []
rows = {}  # Dictionary to store rows  
      
with open(filename1, mode='r', encoding='utf-8') as file:
    csvreader = csv.reader(file, delimiter=';')
    for i, row in enumerate(csvreader):
        rows[f'row{i}'] = row

# Accessing rows from the dictionary      
        
double_model_avg = [float(s.strip('[]')) for s in rows['row2']]   
single_model_avg = [float(s.strip('[]')) for s in rows['row0']]   
double_model = [float(s.strip('[]')) for s in rows['row3']]   
single_model = [float(s.strip('[]')) for s in rows['row1']]   


double_array_avg = np.array(double_model_avg)
single_array_avg = np.array(single_model_avg)

double_array = np.array(double_model)
single_array = np.array(single_model)

#print(double_model)  # Prints the first row
#print(single_model)  # Prints the second row  

reshaped_data1 = double_array.reshape(20, 250)
reshaped_data2 = single_array.reshape(20, 250)
reshaped_data1_avg = double_array_avg.reshape(20, 250)
reshaped_data2_avg = single_array_avg.reshape(20, 250)

double_avg = np.mean(reshaped_data1, axis=0)
single_avg = np.mean(reshaped_data2, axis=0)

double_avg_epi = np.mean(reshaped_data1_avg, axis=0)
single_avg_epi = np.mean(reshaped_data2_avg, axis=0)


'''
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
print('Field names are:' + ', '.join(field for field in fields1))      #fields one is the first row on the csv file refered to as column title row
selected_row1 = rows1[0]
selected_row2 = rows1[2]
# printing the field names

# Remove square brackets and split the string into parts
old_list = str(fields1).replace('[', '').replace(']', '')
number_strings = old_list.split(',')

# Strip extra whitespace and single quotes, then convert to float
double_model = [float(num.strip().strip("'")) for num in number_strings]

# Convert each part to a float
double_model = [float(item.strip('[]')) for item in selected_row1]
single_model = [float(item.strip('[]')) for item in selected_row2]
print('third line:',single_model)
'''
single_avg = savgol_filter(single_avg, window_length=11, polyorder=2)
double_avg = savgol_filter(double_avg, window_length=11, polyorder=2)



fig1, ax = plt.subplots()

ln1, = plt.plot(single_avg, label='Conventional Single TD3 agent')
ln1, = plt.plot(double_avg, label='Alternating between 2 TD3 agents')
ln1, = plt.plot(single_avg_epi, label='Long term average reward single model')
ln1, = plt.plot(double_avg_epi, label='Long term average rewad alternating mode')


legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
plt.xlabel("Steps")
plt.ylabel("Sum throuput per step (Mbps)")
plt.title('Average Throughput over 20 episodes')
plt.show()
#plt.savefig('TD3_0CIO_{}.png'.format(int(time.time())))


'''
fig1, ax = plt.subplots()
#ln1, = plt.plot(np.repeat(step_rewards,episode_steps,axis=0), label='All traied model')
#ln1, = plt.plot(np.repeat(step_rewards0,episode_steps,axis=0), label='MIMO trained model')
ln1, = plt.plot(single_model, label='All trained model')
ln1, = plt.plot(double_model, label='MIMO trained model')
plt.ylim(22, 40)
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
plt.xlabel("Step")
plt.ylabel("Average Overall throughput")
plt.title('Comparing step reward: All cases model vs. MIMO only model')
plt.show()
#plt.savefig('TD3_0CIO_{}.png'.format(int(time.time())))
'''




