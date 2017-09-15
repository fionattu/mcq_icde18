from truthfinder import *


def init(num_of_workers, expertise_init, num_of_tasks, difficulty_init):
    return initialization(num_of_workers, expertise_init, num_of_tasks, difficulty_init)


def start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw, expertise_init, difficulty_init):
    # start learning
    [expertise, difficulty] = init(num_of_workers, expertise_init, num_of_tasks, difficulty_init)
    for iter_all in range(3): # whole process
        for iter_left in range(3): # left cycle
            # expertise should be value of last iteration
            for i, x in enumerate(expertise):
                expertise[i] = 0.999 if x >= 1 else expertise[i]

            expertise_score = [-np.log(1 - x) for x in expertise]
            confidence_score = [np.dot(x, expertise_score) for x in assign_scheme_tbw]
            confidence_score_damping = np.zeros((num_of_choices, num_of_tasks))
            confidence = np.zeros((num_of_choices, num_of_tasks))

            confidence_score_damping[0][:] = confidence_score[0][:] - 0.6* confidence_score[1][:]
            confidence_score_damping[1][:] = confidence_score[1][:] - 0.6* confidence_score[0][:]

            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    confidence[choice][task] = difficulty[task] * (1 / (1 + np.power(math.e, -confidence_score_damping[choice][task])))
                    # confidence[choice][task] = 1 / (1 + np.power(math.e, -confidence_score_damping[choice][task]))

            confidence = confidence / confidence.sum(axis=0) # normalize confidence
            for x in range(num_of_workers):
                expertise[x] = 0
                for task in range(num_of_tasks):
                    for choice in range(num_of_choices):
                        expertise[x] += (assign_scheme_tbw[choice][task][x] * confidence[choice][task])
                num_of_ans = (np.count_nonzero(assign_scheme_tbw[0][:,x]) + np.count_nonzero(assign_scheme_tbw[1][:,x]))
                if num_of_ans == 0:
                    expertise[x] = expertise_init
                else:
                    expertise[x] /= (np.count_nonzero(assign_scheme_tbw[0][:,x]) + np.count_nonzero(assign_scheme_tbw[1][:,x]))

        difficulty_score = np.zeros(num_of_tasks)
        for iter_right in range(3):  # right cycle
            for task in range(num_of_tasks):
                difficulty_score[task] = np.abs(confidence[0][task] - confidence[1][task])
                difficulty[task] = 1 / (1 + 0.4 * np.power(math.e, -difficulty_score[task])) #todo: 0.4
            # update confidence
            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    confidence[choice][task] = difficulty[task] * (1 / (1 + np.power(math.e, -confidence_score_damping[choice][task])))
            confidence = confidence / confidence.sum(axis=0)  # normalize confidence
        # update expertise
        for x in range(num_of_workers):
            expertise[x] = 0
            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    expertise[x] += (assign_scheme_tbw[choice][task][x] * confidence[choice][task])
            num_of_ans = (np.count_nonzero(assign_scheme_tbw[0][:, x]) + np.count_nonzero(assign_scheme_tbw[1][:, x]))
            if num_of_ans == 0:
                expertise[x] = expertise_init
            else:
                expertise[x] /= (np.count_nonzero(assign_scheme_tbw[0][:, x]) + np.count_nonzero(assign_scheme_tbw[1][:, x]))

    return [expertise, expertise_score, confidence, confidence_score, difficulty, difficulty_score] #check whether local can return