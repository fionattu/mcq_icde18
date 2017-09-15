#		The task is to identify the adult level of a website G(0), P(1), R(2), X(3), B(4)
import csv

file_data=open('labels.txt')
X=[]
for line in file_data:
    line=line.split()
    a=-1
    if line[2]=='G':
        a=0
    if line[2]=='P':
        a=1
    if line[2]=='R':
        a=2
    if line[2]=='X':
        a=3
    if line[2]=='B':
        a=4
    if a<0:
        print "Answer Error!!!!"
    else:
        X.append([line[1],line[0],a])


file_gt=open('gold.txt')
Y=[]
for line in file_gt:
    line=line.split()
    a=-1
    if line[1]=='G':
        a=0
    if line[1]=='P':
        a=1
    if line[1]=='R':
        a=2
    if line[1]=='X':
        a=3
    if line[1]=='B':
        a=4
    if a<0:
        print "Truth Error!!!!"
    else:
        Y.append([line[0],a])

csvfile_w = file('answer.csv', 'wb+')
writer = csv.writer(csvfile_w)
temp=[['question', 'worker', 'answer']]
writer.writerows(temp)
writer.writerows(X)
csvfile_w.close()

csvfile_w = file('truth.csv', 'wb+')
writer = csv.writer(csvfile_w)
temp=[['question', 'truth']]
writer.writerows(temp)
writer.writerows(Y)
csvfile_w.close()