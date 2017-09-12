import csv
import regex as re

def read_worker_label(dataset):
    file_path = './datasets/' + dataset + '/answer.csv'
    print file_path
    with open(file_path) as worker_labels:
        reader = csv.reader(worker_labels, delimiter=' ', quotechar='|')
        next(reader, None) # neglect the header
        labels = []
        for line in reader:
            currentline = line[0].split(',')
            labels.append({'worker': currentline[1], 'task':currentline[0], 'answer':currentline[2]})
        return labels


def read_truths(dataset):
    file_path = './datasets/' + dataset + '/truth.csv'
    print file_path
    with open(file_path) as truths:
        reader = csv.reader(truths, delimiter=' ', quotechar='|')
        next(reader, None)
        truths = []
        for line in reader:
            currentline = line[0].split(',')
            truths.append({'task': currentline[0], 'truth': currentline[1]})
        return truths


def add_to_arrivals(arrivals, worker, time):
    for index in range(len(arrivals)):
        for key in arrivals[index]:
            if key == worker:
                arrivals[index][key].append(time)
    arrivals.append({worker:[time]})


def read_arrival_times(dataset):
    file_path = './datasets/arrival_times/' + dataset + '.csv'
    print file_path
    with open(file_path, 'rU') as truths:
        reader = csv.reader(truths, delimiter='\t',quotechar='|', dialect=csv.excel_tab)
        header = reader.next()[0].split(',')
        index_of_arrivals = header.index('_last_judgment_at')
        index_of_user_ids = header.index('id')
        index_of_golden = header.index('_golden')
        arrivals = []
        rx = re.compile(r'"[^"]*"(*SKIP)(*FAIL)|,')
        for line in reader:
            currentline = rx.split(line[0])
            if (currentline[index_of_golden] == 'FALSE'):
                print currentline[index_of_user_ids]
                print currentline[index_of_arrivals]
                print " "
                add_to_arrivals(arrivals, currentline[index_of_user_ids], currentline[index_of_arrivals])


        return arrivals


times = read_arrival_times('Relevance_of_terms_to_disaster_relief_topics')
# labels = read_worker_label('d_Duck Identification_40w217q')
# read_worker_label('d_sentiment_1kQ20w')
# read_worker_label('f201_Emotion_FULL_700q10w')
# truths = read_truths('d_Duck Identification_40w217q')