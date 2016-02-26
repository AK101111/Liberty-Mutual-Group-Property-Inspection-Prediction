import numpy as np

def gini(list_of_values):
  sorted_list = sorted(list(list_of_values))
  height, area = 0, 0
  for value in sorted_list:
    height += value
    area += height - value / 2.
  fair_area = height * len(list_of_values) / 2
  return (fair_area - area) / fair_area
  
def normalized_gini(y_pred, y):
    normalized_gini = gini(y_pred)/gini(y)
    return normalized_gini
    

#predicted_y = np.random.randint(100, size = 1000)
#desired_y = np.random.randint(100, size = 1000)

#ngc = normalized_gini(predicted_y, desired_y)
#print ngc
