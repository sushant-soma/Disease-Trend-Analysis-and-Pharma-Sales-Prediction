import pandas as pd
import random
import matplotlib.pyplot as plt


df = pd.read_csv('all_year.csv')

c = ['Prscrbr_Geo_Desc','Gnrc_Name']
for i in range(14,22):
    for j in range(1,13):
        c.append(f'20{i}-{j}-1')


e=[None] * 96
#print(e)
v = pd.DataFrame(columns=c)
for r in df.values:
    a = list(r[0:2])
    for i in range(2,len(r)-1):
        prev=float(str(r[i]).replace(",",""))
        
        next=float(str(r[i+1]).replace(",",""))
        med=(next-prev)/132
        #print(prev,next,med)
        prev/=12
        
        for j in range(12):
            #print(prev)
            a.append(int(prev))
            ran=random.random()-0.5
            prev+=med+ran*med
    #print(len(a))
    v.loc[len(v.index)] = a
    

#print(v)
# plt.plot(c[2::], v.values[0][2::])
# plt.show()

v.to_csv('augment.csv')

# for i in df.values:
#     print(i)

