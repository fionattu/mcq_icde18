import numpy as np
import math


def gaussian_random_generator(mea, dev):
    return np.random.normal(mea, dev)


def uniform_random_generator(low, high):
    return np.random.uniform(low, high)


# generate the truth of each task
def answer_generator(num_of_choices):
    fraction = 1.0/num_of_choices
    random_float = uniform_random_generator(0,1)
    return int(random_float/fraction) + 1


# generate all tasks with truths
def tasks_generator(num_of_tasks, num_of_choices):
    ans = np.zeros(num_of_tasks)
    for i in range(num_of_tasks):
        ans[i] = answer_generator(num_of_choices)
    return ans


# generate each worker answer
def choices_generator(expertise, num_of_choices, answer):
    random_float = uniform_random_generator(0,1)
    if random_float <= expertise:
        return answer
    else:
        wrong_ans = np.zeros(num_of_choices - 1)
        index = 0
        for choice in range(num_of_choices):
            if (choice + 1) != answer:
                wrong_ans[index] = choice + 1
                index += 1
        fraction = (1.0 - expertise)/(num_of_choices - 1)
        return wrong_ans[int((random_float-expertise)/fraction)]


# simulate the choices for all the workers (assume every worker answers all tasks)
def workers_generator(num_of_workers, num_of_tasks, num_of_choices, ans):
    workers = [{} for _ in range(num_of_workers)]
    expertise_truth = [{} for _ in range(num_of_workers)]
    difficulty_truth = [{} for _ in range(num_of_tasks)]
    tasks = [{} for _ in range(num_of_tasks)]
    for i in range(num_of_tasks):
        difficulty_truth[i] = uniform_random_generator(0.8,1);
        # difficulty_truth[i] = gaussian_random_generator(0.9, 0.03)
    for i in range(number_of_workers):
        expertise_truth[i] = uniform_random_generator(0.5,0.999);
        # expertise_truth[i] = gaussian_random_generator(0.7, 0.1);
    for worker in range(num_of_workers):
        # expertise = uniform_random_generator(0.8, 1) #matters
        # expertise_truth[worker] = expertise
        for task in range(num_of_tasks):
            answer = ans[task]
            # probability of correct answers
            choice = choices_generator(1/(1 + np.power(math.e, -3*expertise_truth[worker] * difficulty_truth[task])), num_of_choices, answer)
            # choice = choices_generator(difficulty_truth[task] / (1 + np.power(math.e, -1 * expertise_truth[worker])), num_of_choices,answer)
            workers[worker][task] = choice
            tasks[task][worker] = choice

    return [workers, tasks, expertise_truth, difficulty_truth]


# initialize inference
def initialization(num_of_workers, expertise_init, num_of_tasks, difficulty_init):
    expertise = [expertise_init] * num_of_workers
    difficulty = [difficulty_init] * num_of_tasks
    return [expertise, difficulty]

# truth inference algorithm
def truth_inference(num_of_workers, expetise_init, num_of_tasks, difficulty_init, num_of_choices):
    ans = tasks_generator(num_of_tasks, num_of_choices)
    a = workers_generator(num_of_workers, num_of_tasks, num_of_choices, ans)
    [workers, tasks, expertise_truth, difficulty_truth] = [a[0], a[1], a[2], a[3]]
    init = initialization(num_of_workers, expetise_init, num_of_tasks, difficulty_init)
    [expertise, difficulty] = [init[0], init[1]]

    # create tasks_x_workers matrix for each choice, 1 for voted
    tasks_by_workers = [np.zeros((num_of_tasks, num_of_workers)) for _ in range(num_of_choices)]
    for idx, task in enumerate(tasks):
        for key, value in task.iteritems(): # key: worker index; value: worker choice
            tasks_by_workers[int(value - 1)][idx][key] = 1 # three dimentions: choice, task, worker

    # learning
    for ite in range(5):
        total_confidence1 = 0
        total_confidence2 = 0
        for iter_exp in range(5): #left cycle
            # expertise can be equal to 1, problematic when calculating the expertise_score
            for i, x in enumerate(expertise):
                if  x >= 1:
                    expertise[i] = 0.999
            expertise_score =[-np.log(1-x) for x in expertise]

            # calculate the confidence score of each answer
            confidence_score = [np.dot(x, expertise_score) for x in tasks_by_workers]
            confidence_score_damping = np.zeros((num_of_choices, num_of_tasks))
            confidence = np.zeros((num_of_choices, num_of_tasks))
            # dampening effect
            # choice 0 confidence score damping confidence_score_damping[0][:]
            # confidence_score_damping[0][:] = confidence_score[0][:];
            # confidence_score_damping[1][:] = confidence_score[1][:];

            confidence_score_damping[0][:] = confidence_score[0][:] - 1*confidence_score[1][:];
            confidence_score_damping[1][:] = confidence_score[1][:] - 1*confidence_score[0][:];


            # calculate the confidence of each answer
            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    # if task is 1:
                    #     print ""
                    #     print "choice: ", choice
                    #     print "task: ", task
                    #     print "value: ", -confidence_score_damping[choice][task]
                    confidence[choice][task] = difficulty[task] * (1/(1 + np.power(math.e, -confidence_score_damping[choice][task])))
                    # confidence[choice][task] = 1 / (1 + np.power(math.e, -confidence_score_damping[choice][task]))
                    total_confidence1 += confidence[choice][task]
            # print "1st con: ", total_confidence1
            # update expertise
            for x in range(num_of_workers):
                expertise[x] = 0
                for task in range(num_of_tasks):
                    for choice in range(num_of_choices):
                        expertise[x] += tasks_by_workers[choice][task][x] * confidence[choice][task]
            expertise = [x / num_of_tasks for x in expertise]

        # print ""
        # print "dampen value: ", -confidence_score_damping[0][1]
        # print "dampen value: ", -confidence_score_damping[1][1]
        # print "confidence value: ", confidence[0][1]
        # print "confidence value: ", confidence[1][1]
        # calculate the difficulty score & difficulty of each question
        difficulty_score = np.zeros(num_of_tasks)
        for iter_dif in range(5): # right cycle
            for task in range(num_of_tasks):
                difficulty_score[task] = np.abs(confidence[0][task] - confidence[1][task])
                difficulty[task] = 1/(1 + 0.4 * np.power(math.e, -difficulty_score[task]))
            # update confidence
            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    confidence[choice][task] = difficulty[task] * (1 / (1 + np.power(math.e, -confidence_score_damping[choice][task])))
                    total_confidence2 += confidence[choice][task]
            # print "2rd con: ", total_confidence2

        # update expertise
        for x in range(num_of_workers):
            expertise[x] = 0
            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    expertise[x] += tasks_by_workers[choice][task][x] * confidence[choice][task]
        expertise = [x / num_of_tasks for x in expertise]

    # average error
    # exp_difference = 0
    # for i in range(num_of_workers):
    #     exp_difference += np.abs(expertise_truth[i] - expertise[i])
    #
    # dif_difference = 0
    # for i in range(num_of_tasks):
    #     dif_difference += np.abs(difficulty_truth[i] - difficulty[i])
    #
    # return [exp_difference/num_of_workers, dif_difference/num_of_tasks]

    #rmse
    exp_difference = 0
    for i in range(num_of_workers):
        exp_difference += np.power(np.abs(expertise_truth[i] - expertise[i]), 2)

    dif_difference = 0
    for i in range(num_of_tasks):
        dif_difference += np.power(np.abs(difficulty_truth[i] - difficulty[i]), 2)

    #accuracy
    number_correct_answers = 0
    for i in range(number_of_tasks):
        infer_answer = 1
        if (confidence[0][i] < confidence[1][i]):
            infer_answer = 2
        if (infer_answer == ans[i]):
            number_correct_answers += 1
    # return [np.power(exp_difference/num_of_workers, 0.5), np.power(dif_difference/num_of_tasks, 0.5), (float)(number_correct_answers)/number_of_tasks]
    return [np.power(exp_difference/num_of_workers, 0.5), np.power(dif_difference/num_of_tasks, 0.5), number_correct_answers]

number_of_tasks = 50
number_of_choices = 2
number_of_workers = 15
first_iteration = 10
second_iteration = 10
# truth_inference(number_of_workers, 0.5, number_of_tasks, 0.5, number_of_choices)

# for j in range(first_iteration):
exp_difference = dif_difference = accuracy = 0
for i in range(second_iteration):
    res = truth_inference(number_of_workers, 0.5, number_of_tasks, 0.5, number_of_choices)
    exp_difference += res[0]
    dif_difference += res[1]
    accuracy += res[2]
    print "accuracy:", res[2]
print exp_difference/second_iteration, "  ", dif_difference/second_iteration, " ", accuracy/second_iteration