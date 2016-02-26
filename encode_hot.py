import csv
import sys
import pandas as pd

# data front reading format - basically a type of dictionary
df = pd.read_csv('train.csv')


# categorical variables identified by hand stored in a list
Categorical_vars_new = ['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V11','T1_V12','T1_V15','T1_V16','T1_V17','T2_V3','T2_V5','T2_V11','T2_V12','T2_V13']

# temp file new.csv
with open('new.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile)
	# iterating through all columns of data
	for x in df:	#Categorical_vars_old:
		# if column is categorical
		if x in Categorical_vars_new:
			# use one-hot encoding scheme to create new dichotomy variables
			y=pd.get_dummies(df[x])
			for j in y:
				writer.writerow(y[j].transpose().tolist())
		else:
			writer.writerow(df[x].transpose().tolist())
# new.csv contains transpose of all req data
# routine to transpose given file
with open('new.csv') as f:
    reader = csv.reader(f)
    cols = []
    for row in reader:
        cols.append(row)

with open('train_one_hot.csv', 'wb') as f:
    writer = csv.writer(f)
    for i in range(len(max(cols, key=len))):
        writer.writerow([(c[i] if i<len(c) else '') for c in cols])
# part of pre-processing done


			
		

		


