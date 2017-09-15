import csv
import regex as re
import time


def read_worker_label(dataset):
    file_path = './datasets/' + dataset + '/answer.csv'
    print file_path
    with open(file_path) as worker_labels:
        reader = csv.reader(worker_labels, delimiter=' ', quotechar='|')
        next(reader, None) # neglect the header
        labels = []
        workers = []
        for line in reader:
            currentline = line[0].split(',')
            labels.append({'worker': currentline[1], 'task':currentline[0], 'answer':currentline[2]})
            if currentline[1] not in workers:
                workers.append(currentline[1])
        return workers, labels


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


def convert_to_epoch(datetime_str):
    patterns = ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M')
    for pattern in patterns:
        try:
            return int(time.mktime(time.strptime(str(datetime_str), pattern)))
        except ValueError:
            pass


def add_to_arrivals(arrivals, worker, time):
    epoch_time = convert_to_epoch(time)
    for index in range(len(arrivals)):
        for key in arrivals[index]:
            if key == worker:
                arrivals[index][key].append(epoch_time)
                return
    arrivals.append({worker:[epoch_time]})


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
                add_to_arrivals(arrivals, currentline[index_of_user_ids], currentline[index_of_arrivals])
        print arrivals
        return arrivals

# arrival_test = [{'4': [1453978938, 1454122879]}, {'5': [1453978938, 1454122879]}, {'6': [1453978938, 1454122879]}]
# read_arrival_times('Relevance_of_terms_to_disaster_relief_topics')