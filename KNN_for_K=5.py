import numpy as np
import math


def most_of_labels(knn, labels):###Which label appears in KNN the most
    most_of = 0
    for label in labels:
        count = list(knn).count(label)
        if count > most_of:
            most_of = label
    return most_of


def euclid_distance(x1, x2):
    sum = 0

    for i in range(len(x1) ):
        sum += (x1[i] - x2[i]) * (x1[i] - x2[i])

    return float(math.sqrt(sum))


def manhattan_distance(x1, x2):
    sum = 0

    for i in range(len(x1)):
        sum += abs(x1[i] - x2[i])

    return sum


def compare_distances(knn, distance):
    for i in range(len(knn)):
        if (knn[i] > distance):
            return i
    return None

def fix_confusion_matrix(confusion):####Matrix it takes shows the amount of time that particular label has been chosen
                                    ###This function changes that to odds.Meaning biggest number this array hold
                                    ###can be 1.
    size=len(confusion[0])
    sums = [0 for x in range(size)]

    for i in range(len(confusion)):
        for k in confusion[i]:
            sums[i] += k

    for i in range(size):
        for k in range(size):
            if(sums[i] is not 0 ):
                confusion[i][k] = confusion[i][k]/sums[i]
    return confusion



###train_data=data with just features,no label
###test_data=data with just features,no label
###train_labels=data with just label,index with tain_data matches
###train_data=data with just label,index with tain_data matches
###labels=types of posibles labels.For iris.dat example:[1,2,3]

def guess_by_euclid(train_data, test_data,train_labels,test_labels,labels):##uses euclid distance helper function
    knn = np.array([0.0, 0.0, 0.0, 0.0, 0.0])###Closest distances to hold
    knn_index = np.array([0, 0, 0, 0, 0])###To hold the indexes of the datas that closest distances belongs to.

    confussion=[[0 for x in range(len(labels))] for y in range(len(labels))]##empty 2d array for confusion matrix
                                                                            ##initiated to 0

    match = 0
    not_match = 0
    for test in range(len(test_data)):
        n = 0
        for train in range(len(train_data)):
            if n < 5:
                knn[n] = euclid_distance(train_data[train], test_data[test])
                knn_index[n] = train_labels[train]
                n += 1
            else:
                distance = euclid_distance(train_data[train], test_data[test])
                index = compare_distances(knn, distance)
                if index is not None:
                    knn = np.concatenate((knn[:index], [distance], knn[index:-1]), axis=0)

                    knn_index[index] = train_labels[train]

        ####WHİCH LABEL APPEARS THE MOST İN KNN FOR K=5
        guess=most_of_labels(knn_index, labels)
        if  guess == test_labels[test]:
            confussion[guess-1][guess-1]+=1
            match += 1
        else:
            not_match += 1

            confussion[guess-1][test_labels[test]-1] += 1


        knn = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    print("accuracy")
    if (match+not_match) is 0:
        print("%100")
    else:
        print(match / (match + not_match))

    print("confussion matrix")
    print(np.matrix(fix_confusion_matrix(confussion)))



###train_data=data with just features,no label
###test_data=data with just features,no label
###train_labels=data with just label,index with tain_data matches
###train_data=data with just label,index with tain_data matches
###labels=types of posibles labels.For iris.dat example:[1,2,3]
def guess_by_manhattan(train_data, test_data,train_labels,test_labels,labels):##uses manhattan distance helper function
    knn = np.array([0.0, 0.0, 0.0, 0.0, 0.0])###Closest distances to hold
    knn_index = np.array([0, 0, 0, 0, 0])###To hold the indexes of the datas that closest distances belongs to.
    confussion = [[0 for x in range(len(labels) )] for y in range(len(labels) )]##empty 2d array for confusion matrix
                                                                            ##initiated to 0
    match = 0
    not_match = 0
    for test in range(len(test_data)):
        n = 0
        for train in range(len(train_data)):
            if n < 5:
                knn[n] = manhattan_distance(train_data[train],test_data[test])
                knn_index[n] = train_labels[train]
                n += 1
            else:
                distance = manhattan_distance(train_data[train], test_data[test])
                index = compare_distances(knn, distance)
                if index is not None:
                    knn = np.concatenate((knn[:index], [distance], knn[index:-1]), axis=0)

                    knn_index[index] =  train_labels[train]

        ####WHİCH LABEL APPEARS THE MOST İN KNN FOR K=5
        guess = most_of_labels(knn_index, labels)
        if guess == test_labels[test]:
            confussion[guess-1][guess-1] += 1
            match += 1
        else:
            not_match += 1


            confussion[guess-1][test_labels[test]-1] += 1


        knn = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    print("accuracy")
    if (match+not_match) is 0:
        print("%100")
    else:
        print(match / (match + not_match))
    print("confussion matrix")
    print(np.matrix(fix_confusion_matrix(confussion)))



myarray = np.loadtxt('iris.dat', dtype=float)

test = np.concatenate((myarray[0:10], myarray[50:60], myarray[100:110]), axis=0)

train = np.concatenate((myarray[10:50], myarray[60:100], myarray[110:150]), axis=0)


test_data=[]
test_labels=[]
train_data=[]
train_labels=[]
labels = []
for i in test:
    test_data.append(i[:-1])
    test_labels.append(int(i[-1]))


for i in train:
    train_data.append(i[:-1])
    train_labels.append(int(i[-1]))

labels=[1,2,3]

print("\nManhattan distance iris.dat KNN for K=5")
guess_by_manhattan(train_data,test_data,train_labels,test_labels,labels)
print("\nEuclid distance iris.dat KNN for K=5")
guess_by_euclid(train_data,test_data,train_labels,test_labels,labels)

##########################################################################
##########################################################################
##########################################################################


myarray = np.loadtxt('leaf.dat', dtype=float)

test =[]
train=[]
prev=None
i=0
while(i<len(myarray)):
    if prev != myarray[i][0]:
        test.append(myarray[i])
        test.append(myarray[i+1])
        test.append(myarray[i+2])

        prev= myarray[i][0]
        i += 2
    else:
        train.append(myarray[i])
    i += 1

test_data=[]
test_labels=[]
train_data=[]
train_labels=[]
labels = []
for i in test:
    test_data.append(i[1:])
    test_labels.append(int(i[0]))


for i in train:
    train_data.append(i[1:])
    train_labels.append(int(i[0]))




for i in range(36):
    labels.append(i+1)
print("\nManhattan distance leaf.dat KNN for K=5")
guess_by_manhattan(train_data,test_data,train_labels,test_labels,labels)
print("\nEuclid distance leaf.dat KNN for K=5")
guess_by_euclid(train_data,test_data,train_labels,test_labels,labels)



