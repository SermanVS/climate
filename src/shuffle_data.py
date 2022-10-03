'''
cyclone_occurances = []
no_cyclone = []
for i in range(113960):
    if len(cyclone_events_2[:, :, i][cyclone_events_2[:, :, i] != False]) > 0:
        cyclone_occurances.append(i)
    else:
        no_cyclone.append(i)
        
import random
random.shuffle(cyclone_occurances)
random.shuffle(no_cyclone)    

f1 = open("shuffle_cyclone.csv", "w")
f2 = open("shuffle_no_cyclone.csv", "w")
[f1.write(str(item) + ',') for item in cyclone_occurances]
[f2.write(str(item) + ',') for item in no_cyclone]
f1.close()
f2.close()
'''