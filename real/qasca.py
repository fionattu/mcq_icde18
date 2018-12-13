from qasca_helper import *
from qasca_synthetic import *



def run_qasca(timestamps, worker_set, truths, ans, num_of_batches, num_of_choices, eval, repetition, k, confidence_init, difficulty_init, expertise_init):
    num_of_workers = len(worker_set)
    print "workers len:", num_of_workers
    num_of_tasks = 80
    print "tasks len: ", num_of_tasks
    assign_tbw = np.zeros((num_of_tasks, num_of_workers))
    assign_scheme_tbw = [np.zeros((num_of_tasks, num_of_workers)) for _ in range(num_of_choices)]  # assignmnet scheme
    quality = [uniform_random_generator(0.5, 0.999)] * num_of_workers
    tasks = []
    processing = []
    repeats = [repetition] * num_of_tasks
    resList = []
    for t in range(num_of_tasks):
        task = [1.0 / num_of_choices] * num_of_choices
        tasks.append(task)

    completed = False
    current_batch = 0
    workers = []
    while (completed is False):
        print "______________________batch______________________", current_batch
        if current_batch <= num_of_batches:
            batch_workers = find_batch_workers(current_batch, timestamps) # add avaliable worker set, worker set can be empty
            add_to_workers(workers, batch_workers) #assume workers wil remain in task pool
        available_workers = obtain_available_workers(workers, processing)
        assign(eval, num_of_tasks, available_workers, assign_tbw, quality, tasks, k, processing, repeats)
        worker_submit_answers(processing, tasks, quality, assign_scheme_tbw, resList, ans)  # update prococessing, update qc
        if check_completed(num_of_tasks, repeats, processing) is True:
            completed = True
        current_batch += 1
    print "finish"
    # using em as inference
    # em = qasca_EM.infer(resList, quality)
    #
    # infer_truths = get_result(num_of_tasks, em)
    #using MCQ inference
    [infer_expertise, infer_expertise_score, infer_confidence, infer_confidence_score, infer_difficulty,
     infer_difficulty_score] = start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw,
                                               expertise_init, difficulty_init)
    infer_truths = get_infer_truths(num_of_tasks, infer_confidence)

    return [print_accuracy(num_of_tasks, truths, infer_truths), current_batch, num_of_tasks]

