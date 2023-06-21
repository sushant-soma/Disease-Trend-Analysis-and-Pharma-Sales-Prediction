def sales(state):
    import csv

    temp = {}

    # opening the CSV file
    with open('usa.csv', mode='r') as file:
        # reading the CSV file
        csvFile = csv.DictReader(file)

        # displaying the contents of the CSV file
        for lines in csvFile:
            if (lines['Prscrbr_Geo_Desc'] == state):
                words = lines['Gnrc_Name']
                count = int(lines['Tot_Clms'])
                words = words.split('/')
                for word in words:
                    word = word.lower()
                    if word in temp:
                        temp[word] += count
                    else:
                        temp[word] = count
                        
    return temp

#print(dict)



def analyse(state):
    import csv
    place_array = []

    # opening the CSV file
    with open('usa.csv', mode='r') as file:
        # reading the CSV file
        csvFile = csv.DictReader(file)

        # displaying the contents of the CSV file
        for lines in csvFile:
            place = lines['Prscrbr_Geo_Desc']
            if (place not in place_array):
                place_array.append(place)

    print(place_array)
    # state = input("Enter the state name : ")

    if (state not in  place_array):
        print("State not found")
    else:
        dict1 = sales(state)


    #print(dict1)

    import csv

    dict2 = {}

    # opening the CSV file
    with open('who.csv', mode='r') as file:
        # reading the CSV file
        csvFile2 = csv.DictReader(file)

        # displaying the contents of the CSV file
        for lines in csvFile2:
            disease = lines['Indication']
            quantity = lines['Formulations']
            quantity = quantity.replace(" ", "")
            contents = lines['Medicine name']
            contents = contents.replace(" + ","/")
            contents = contents.replace("+ ","/")
            contents = contents.replace(" +","/")
            contents = contents.replace("+","/")
            #print(content_array)
            
            for i in range(len(quantity)):
                if (i<=len(quantity)-2):
                    if (quantity[i]=='m' and quantity[i+1]=='g'):
                        i = i-1
                        qua = 0
                        factor = 1
                        while(quantity[i].isdigit()):
                            qua = qua+factor*int(quantity[i])
                            i-=1

                if (quantity[i]=='%'):
                    i = i-1
                    qua = 0
                    factor = 1
                    while(quantity[i].isdigit()):
                        qua = qua+factor*int(quantity[i])
                        factor = factor*10
                        i-=1
            
            
            
            arr = []
            arr.append(contents)
            arr.append(qua)
                
            if disease not in dict2:
                dict2[disease] = []
                dict2[disease].append(arr)

            else:
                dict2[disease].append(arr)
            
    #print(dict2)
    sevierty = {}
    check = []

    for disease in dict2:
            if disease not in sevierty:
                severe_num = 0
                #print(disease)
                for content in dict2[disease]:
                    #print(content)
                    ############# content[1] is coefficient
                    if content[0] in dict1:
                        check.append(content[0])
                        #print("YESS")
                        #print(content[0])
                        severe_num = severe_num + dict1[content[0]]/content[1]
                        #print(content[1]*dict1[content[0]])
                sevierty[disease] = severe_num

            else:
                #print(disease)
                for content in dict2[disease]:
                    #print(content)
                    ############# content[1] is coefficient
                    if content[0] in dict1:
                        #print("YESS")
                        #print(content[0])
                        severe_num = severe_num + dict1[content[0]]/content[1]
                        #print(content[1]*dict1[content[0]])
                sevierty[disease] = sevierty[disease] + severe_num

    #print(check)
    #print(sevierty)

    import numpy as np
    import matplotlib.pyplot as plt
    from operator import itemgetter
    plt.switch_backend('Agg') 

    sev_data = (sorted(sevierty.items(), key=itemgetter(1), reverse=True)[5:10])
    print(sev_data)

    courses = [x[0] for x in sev_data]
    values = [x[1] for x in sev_data]

    fig = plt.figure(figsize=(30, 10))

    # creating the bar plot
    plt.bar(courses, values, color='maroon', width=0.4)

    plt.xlabel("Disease")
    plt.ylabel("Occurrence")
    plt.title("Top 5 diseases of the state: " + state)
    plt.savefig("static/graph.png")




