from real import *
from qasca import *
from approx import *


def main(ans_dataset='d_Duck Identification_40w217q',arrival_dataset='Relevance_of_terms_to_disaster_relief_topics', matching_mode='random', num_of_batches=40):
    start_and_end_timestamps, workers, truths, ans = batch_assignment(ans_dataset, arrival_dataset, matching_mode,
                                                                      num_of_batches)
    accuracy, batch, num_of_tasks = run_qasca(start_and_end_timestamps, workers, truths, ans, num_of_batches + 1, 2, 'accuracy', 2,1,0.5,0.5,0.5)
    # accuracy, batch, num_of_tasks = run_ff(start_and_end_timestamps, workers, truths, ans, num_of_batches, 2, 0.5, 0.5, 0.5)
    print 100 * float(accuracy) / num_of_tasks, '%'
    print "total_batch: ", batch
    # baseline(even_timestamps, start_and_end_timestamps, num_of_batches, )
    # ff(even_timestamps, start_and_end_timestamps)
    # bf(even_timestamps, start_and_end_timestamps)
    # random(even_timestamps, start_and_end_timestamps)
    # baseline(even_timestamps, num_of_batches, workers, truths,ans)

num_of_batches = 40
main()