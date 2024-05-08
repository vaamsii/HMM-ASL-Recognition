import numpy as np
import operator

def part_1_a():
    """Provide probabilities for the word HMMs outlined below.
    Word ALLIGATOR, NUTS, and SLEEP.
    Review Udacity Lesson 8 - Video #29. HMM Training
    Returns:
        tuple() of
        (prior probabilities for all states for word ALLIGATOR,
         transition probabilities between states for word ALLIGATOR,
         emission parameters tuple(mean, std) for all states for word ALLIGATOR,
         prior probabilities for all states for word NUTS,
         transition probabilities between states for word NUTS,
         emission parameters tuple(mean, std) for all states for word NUTS,
         prior probabilities for all states for word SLEEP,
         transition probabilities between states for word SLEEP,
         emission parameters tuple(mean, std) for all states for word SLEEP)
        Sample Format (not complete):
        (
            {'A1': prob_of_starting_in_A1, 'A2': prob_of_starting_in_A2, ...},
            {'A1': {'A1': prob_of_transition_from_A1_to_A1,
                    'A2': prob_of_transition_from_A1_to_A2,
                    'A3': prob_of_transition_from_A1_to_A3,
                    'Aend': prob_of_transition_from_A1_to_Aend},
             'A2': {...}, ...},
            {'A1': tuple(mean_of_A1, standard_deviation_of_A1),
             'A2': tuple(mean_of_A2, standard_deviation_of_A2), ...},
            {'N1': prob_of_starting_in_N1, 'N2': prob_of_starting_in_N2, ...},
            {'N1': {'N1': prob_of_transition_from_N1_to_N1,
                    'N2': prob_of_transition_from_N1_to_N2,
                    'N3': prob_of_transition_from_N1_to_N3,
                    'Nend': prob_of_transition_from_N1_to_Nend},
             'N2': {...}, ...}
            {'N1': tuple(mean_of_N1, standard_deviation_of_N1),
             'N2': tuple(mean_of_N2, standard_deviation_of_N2), ...},
            {'S1': prob_of_starting_in_S1, 'S2': prob_of_starting_in_S2, ...},
            {'S1': {'S1': prob_of_transition_from_S1_to_S1,
                    'S2': prob_of_transition_from_S1_to_S2,
                    'S3': prob_of_transition_from_S1_to_S3,
                    'Send': prob_of_transition_from_S1_to_Send},
             'S2': {...}, ...}
            {'S1': tuple(mean_of_S1, standard_deviation_of_S1),
             'S2': tuple(mean_of_S2, standard_deviation_of_S2), ...}
        )
    """

    # first I got the MU and st.deviation of the initial set given to me, I was also given the boundaries from above
    # Then check the left probability of the boundary with |46−μ1| / σ1 vs |46−μ2| / σ2 equation from above
    # do recursively until there is a convergence and then I found transition probs, by getting count of the state
    # divide that count by the number of samples for that word in that state, which is 3 for all of the words.
    # take that number, divide 1 over that number. That's the S1 -> S2 value, the S1 -> S1 value is 1- that value.

    """Word ALLIGATOR"""
    a_prior_probs = {
        'A1': 0.333,
        'A2': 0.0,
        'A3': 0.0,
        'Aend': 0.0
    }
    a_transition_probs = {
        'A1': {'A2': 0.167, 'A3': 0.0, 'A1': 0.833, 'Aend': 0.0},
        'A2': {'A1': 0.0, 'A2': 0.786, 'A3': 0.214, 'Aend': 0.0},
        'A3': {'A2': 0.0, 'A1': 0.0, 'A3': 0.727, 'Aend': 0.273},
        'Aend': {'A2': 0.0, 'A3': 0.0, 'A1': 0.0, 'Aend': 1.0}
    }
    # Parameters for end state is not required͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    a_emission_paras = {
        'A1': (51.056, 21.986),
        'A2': (28.357, 14.936),
        'A3': (53.727, 16.707),
        'Aend': (None, None)
    }

    """Word NUTS"""
    n_prior_probs = {
        'N1': 0.333,
        'N2': 0.0,
        'N3': 0.0,
        'Nend': 0.0
    }
    # Probability of a state changing to another state.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    n_transition_probs = {
        'N1': {'N3': 0.0, 'N1': 0.919, 'N2': 0.081, 'Nend': 0.0},
        'N2': {'N3': 1.0, 'N1': 0.0, 'N2': 0.0, 'Nend': 0.0},
        'N3': {'N3': 0.625, 'N2': 0.0, 'N1': 0.0, 'Nend': 0.375},
        'Nend': {'N1': 0.0, 'N2': 0.0, 'N3': 0.0, 'Nend': 1.0}
    }
    # Parameters for end state is not required͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    n_emission_paras = {
        'N1': (38.081,11.175),
        'N2': (42,2.828),
        'N3': (60,13.491),
        'Nend': (None, None)
    }

    """Word SLEEP"""
    s_prior_probs = {
        'S1': 0.333,
        'S2': 0.0,
        'S3': 0.0,
        'Send': 0.0
    }
    s_transition_probs = {
        'S1': {'S1': 0.625, 'S3': 0.0, 'S2': 0.375, 'Send': 0.0},
        'S2': {'S2': 0.864, 'S1': 0.0, 'S3': 0.136, 'Send': 0.0},
        'S3': {'S3': 0.0, 'S2': 0.0, 'S1': 0.0, 'Send': 1.0},
        'Send': {'S3': 0.0, 'S2': 0.0, 'S1': 0.0, 'Send': 1.0}
    }
    # Parameters for end state is not required͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    s_emission_paras = {
        'S1': (29.5,8.411),
        'S2': (36.182,5.99),
        'S3': (36.667,1.886),
        'Send': (None, None)
    }

    return (a_prior_probs, a_transition_probs, a_emission_paras,
            n_prior_probs, n_transition_probs, n_emission_paras,
            s_prior_probs, s_transition_probs, s_emission_paras)

def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous
        distribution should be zero, instead of the number outputted here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x.
        The probability of getting value "x" is zero because a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile



def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.
    Args:
        evidence_vector (list): List of right hand Y-axis positions (integer).
        states (list): List of all states in a word. No transition between words.
                       example: ['A1', 'A2', 'A3', 'Aend', 'N1', 'N2', 'N3', 'Nend']
        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}
        transition_probs (dict): dictionary representing transitions from each
                                 state to every other valid state such as for the above
                                 states, there won't be a transition from 'A1' to 'N1'
        emission_paras (dict): parameters of Gaussian distribution
                                from each state.
    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )
    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.
    """
    sequence = []
    probability = 0.0

    # First let's start with an constraint, get the base case out of the way

    # we are told: If no sequence can be found, the algorithm should return one of the following tuples:
    # (None, 0) (null), ([], 0) (empty list) or (['A1', 'A1', ... 'A1'],0) (Or all being the first state of that letter)
    # I am not check if the evidence vector input is empty then just return [], 0.0, since that's default values
    # given for sequence and probability in this code template.

    if not evidence_vector:
        return [], 0.0

    # next let's initialize the variables that hold the size of states and evidence vector inputs
    # We are told that "Two 2-dimensional tables of size K x T are constructed:" K and T need to be initialized here
    # K is the size of states input and T is the size of the evidence_vector from what I understand
    # since S = {s1,s2,...,sK} is for states, the last element represents the size of the set which is K.
    # For T, it's Y = {y1, y2, ...,yT}, same thing as states, y represents the evidence_vector for us and it's size is T.

    K = len(states)
    T = len(evidence_vector)

    # now let's initialize the T1 and T2, 2-d table with size K x T
    # T1 holds the probability of the most likely path and T2 holds the back pointers to assemble the most likely path
    # in the paper it explicitly mentions this with "T2 stores xj-1 of the most likely path so far"
    # T2 will be of type integers since it will hold the back pointers not the probabilities, it will be whole numbers.
    # Float is resource intensive, but probability is float number so we can't do anything there but we can reduce need in T2

    T1 = np.zeros((K,T))
    T2 = np.zeros((K,T), dtype=int)


    # Going to now use the T1 and T2 to calculate the forumla given
    # T1[i,j] = max(T1[k,j-1]*Aki * Biyj) and T2[i,j] = argmax(T1[k,j-1] * Aki * Biyj)

    # But one thing I noticed is that in pdf that they have an input called PI, initial probabilities
    # although T is greater than 1 is what it says in the PDF. I am going to make sure that at T =0,
    # For when T1 is at T=0, the value of it will be initial probability which is PI
    # From my understanding PI is an array of initial probabilites, so let's create an variable first then append
    # the input prior probabilities to it.

    # EDIT: I got an error below, I after some trail and error I realized that PI is actually not just prior probability
    # it's supposed to be prior multiplied by the gaussian probability of Y, evidence vector and B, emission matrix.
    # AssertionError: 0.01576664562875057 != 0.333 within 7 places (0.31723335437124944 difference)
    # this helped me pass the local test case
    # it kind of makes sense since, T1[i,j] = max(T1[k,j-1]*Aki * Biyj) We won't have transition matrix for when T =0
    # we aren't doing max, because we have only one j value.

    Pi = []
    # s represents the state index in the list of states.
    for s in range(K):
        # getting the current state from the states input
        current_state = states[s]

        # calculating the input prior probability at current state
        prior_probability = prior_probs[current_state]

        # as I mentioned above my reasoning for this, taking gaussian_probability of Y, evidence vector and emission
        gaussian_probability = gaussian_prob(evidence_vector[0], emission_paras[current_state])

        # this is where we finally calculate the initial probability
        initial_probability = prior_probability * gaussian_probability

        # we then append to the PI array we initialized at the top
        Pi.append(initial_probability)

        # Once we have PI, we can initialize the T1 at T =0 and the K = current index of state, is equal to PI at current index s
        T1[s,0] = Pi[s]
        # T2 doesn't exist by the way, since we won't have a back pointer by then.

    # next we have to do the calculation of T1 and T2 for when T > 1.
    # in pseudocode we are told that both T1 and T2 are [i,j] by this you know since it's an 2d data point
    # we will need to loop twice over it. That's the first thing I am going to do.
    # the outer loop of the nested for loop will be us looping over T, inner will be K.
    # we have to start the outer loop from range of 1 to T, not just T. Since range is an 0 inclusive in python.
    # well after the nested loop what do we do? well we have to do another loop to get the
    # probability of the most likely path. Remember what's given to us
    # T1[i,j] = max(T1[k,j-1]*Aki * Biyj) -> If I am just unpacking the variables here
    # there are 3, first is i, then j and finally k. What is i? i is our K variable, j is the our T and k is what?
    # well the variable they have as k, will represent the iterator for the third loop or triple nested loop.
    # The reason being? Well they do T1[k,j-1], you can tell that is an nested loop. Again the reason being
    # in python range is first boundary inclusive or 0, if not specified. and outer boundary exclusive.
    # so in python, range(j) means 0 to j-1. Where are we going to get access to k and j-1 variables, only inside 2 loops.
    # also it's mentioned that the table entries T1, T2 are filled by increasing order of K*j + i

    for j in range(1,T):
        for i in range(K):
            # before we do the next loop for iterator k, I am going to initalize the variables for max_probability and
            # also the best_state, both of these will be used to set our T1 and T2, after this calculation of probability
            max_probability = 0
            best_state = 0
            # now do for loop for k iterator
            for k in range(K):
                # now calculate the probability using our equation:
                # T1[i,j] = max(T1[k,j-1]*Aki * Biyj) -> remember we don't need max right now, that will be done outside
                # I am just getting probability, don't max or argmax for T1 and T2 just yet
                # so then new equation is T1[k,j-1]*Aki * Biyj
                # before I implement that, will explain the variables I mean everything lines up we have i, j, k, T1,
                # the A represents the transition matrix from PDF, for us it's transition_probs and B represents emision
                # we already used this before. It's emission_paras in this code. also notice A at k,i.
                # what this means is get A at k,i states. remember A is K X K size, so at state k iterative variable
                # of the transition_probability, get the value for the key i in the transition probability dict.
                # similarly for B, it's K x N, K being the states and N being the count of observations.
                # this is very similar to what I did for T = 0, for T1. I am computing the guassian proability of
                # evidence vector at j and emission at i. remember I flipped the order around from BiYj to YjBi, because
                # that's how the helper method for gaussian proability is set up in terms of the method signature.

                probability = (T1[k, j-1] * transition_probs[states[k]].get(states[i], 0) * gaussian_prob(evidence_vector[j],emission_paras[states[i]]))

                # okay now we have the probability, let's make use of the max_probability and best_state
                # what I mean by that is let's set check if the probability we just found is greater than max_probability
                # if it is, then set the current probability we just found as the max_probability and the current index
                # of k as the best_state. it will represent the state at which max_probability occurs.
                # note essentially this represents the max and argmax parts for T1 and T2.
                # note, for the first iteration of this loop, this conditional statement will hit as max_probability is 0.

                if probability > max_probability:
                    max_probability = probability
                    best_state = k

            # now outside the loop for k iterative variable, let's set the T1 and T2 with the max_probability and best_state
            # with this we have successfully intialized the table entries of T1 and T2 by increasing order of K*j+i
            T1[i,j] = max_probability
            T2[i,j] = best_state

    # let's exist the loop now, we are done with getting probabilities
    # one thing that was said in the problem description was:
    # "In order to reconstruct your most-likely path after running Viterbi, you’ll need to keep track of a back-pointer
    # at each state, which directs you to that state’s most-likely predecessor."
    # I think for this, I inferred to the part from PDF where it says:
    # "Vtk is the probability of the most probable state sequence P(x1..,xt, y1...,yt) responsible for
    # the first t observations that have k as its final state"
    # Vtk = max(P(yt|k)* ax,k * Vt-1,x
    # also it said "The Viterbi path can be retrieved by saving back pointers that remember which
    # state was used in the second equation."
    # from what I understand we need the xT = argmax(Vt,x), well why>
    # from what I think, Vt is already found which is the T1, it has the best probability from above
    # however Vt,x or in general Vtk, is saying, we need the t observations that k as it's final state.
    # so essentially what the xT = argmax(Vt,x) is doing to get the final state at which we has the most probability
    # T-1 as given in the formula can be used to represent the argmax of the T1, where T-1 of evidence vectors.

    last_maxprob_state = np.argmax(T1[:,T-1])

    # first we can append this to the sequence variable as again this is the back pointer start of the sequence.
    # last_maxprob_state is the state at which the max_probability last occured in T1.

    sequence.append(states[last_maxprob_state])

    # next I noticed that the xt-1 = Ptr(xt,t). They said the Viterbi path can be found using this.
    # well backpointers are something that start at the end at move to the start.
    # We will use backpointers from T2, to see which state was most likley predcessor, given in the hint.
    # remember we have the last_maxprob_state, which represents the absoulte end.
    # our goal is to start from there, hence why I appended to the sequence list, work our way backwards using
    # back pointers to reconstruct the most-liekely path since we already did viterbi as state in hint
    # to do back tracking, we will need to do an reverse loop.
    # the loop will start at T-1, well why? our last_maxprob_state's bounds is that, that's where it ended potentially
    # The end of this will be at 1, but in the loop will specify 0, since the end of range isn't exclusive in python
    # it will decrement by 1 unit each time. this will be the range of the loop.
    # now what will be do in the loop? well I will set the last_maxprob_state at each iteration to be the value of T2
    # at last_maxprob_state and t, the iterative variable. this coreseponds to PDF again.
    # after we get the last_maxprob_state at each iteration, we then insert that state into the sequence list

    for t in range(T-1, 0, -1):
        last_maxprob_state = T2[last_maxprob_state, t]
        sequence.insert(0, states[last_maxprob_state])

    # next is the final part where we need to get the probability this state sequence fits the evidence as a float
    # to do that I am actually going to use the Vt,k formula here as I explained above, it's the probability
    # of the most probable state sequence. which is exactly what we need.
    # Vtk = max(P(yt|k)* ax,k * Vt-1,x)
    # it's exact same way we found the argmax of T1 above, except here we are finding the max
    # we will set this value to the probability variable which was initialized at the top and given to me.

    probability = np.max(T1[:, T-1])

    return sequence, probability

def part_2_a():
    """Provide probabilities for the word HMMs outlined below.
    Now, at each time frame you are given 2 observations (right hand Y
    position & right thumb Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for right thumb, create
    a tuple of (right-hand-y, right-thumb-y) to represent high-dimension transition &
    emission probabilities.
    """

    """Word ALLIGATOR"""
    a_prior_probs = {
        'A1': 0.333,
        'A2': 0.0,
        'A3': 0.0,
        'Aend': 0.0
    }
    # example: {'A1': {'A1' : (right-hand Y, right-thumb Y), ... }͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    a_transition_probs = {
        'A1': {'A2': (0.167, 0.176), 'A1': (0.833, 0.824), 'A3': (0.0, 0.0), 'Aend': (0.0, 0.0)},
        'A2': {'A3': (0.214, 0.231), 'A2': (0.786, 0.769), 'A1': (0.0, 0.0), 'Aend': (0.0, 0.0)},
        'A3': {'A2': (0.0, 0.0), 'A3': (0.727, 0.769), 'A1': (0.0, 0.0), 'Aend': (0.273, 0.231)},
        'Aend': {'A1': (0.0, 0.0), 'A2': (0.0, 0.0), 'A3': (0.0, 0.0), 'Aend': (1.0, 1.0)}
    }
    # example: {'A1': [(right-hand-mean, right-thumb-std), (right-hand--mean, right-thumb-std)] ...}͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    # typo fix: {'A1': [(right-hand-mean, right-hand-std), (right-thumb--mean, right-thumb-std)] ...}͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    a_emission_paras = {
        'A1': [(51.056, 21.986), (53.529, 17.493)],
        'A2': [(28.357, 14.936), (40.769, 6.104)],
        'A3': [(53.727, 16.707), (51.000, 12.316)],
        'Aend': [(None, None), (None, None)]
    }

    """Word NUTS"""
    n_prior_probs = {
        'N1': 0.333,
        'N2': 0.0,
        'N3': 0.0,
        'Nend': 0.0
    }
    n_transition_probs = {
        'N1': {'N2': (0.081, 0.136), 'N1': (0.919, 0.864), 'N3': (0.0, 0.0), 'Nend': (0.0, 0.0)},
        'N2': {'N2': (0.0, 0.727), 'N3': (1.0, 0.273), 'N1': (0.0, 0.0), 'Nend': (0.0, 0.0)},
        'N3': {'N3': (0.625, 0.800), 'N1': (0.0, 0.0), 'N2': (0.0, 0.0), 'Nend': (0.375, 0.200)},
        'Nend': {'N3': (0.0, 0.0), 'N1': (0.0, 0.0), 'N2': (0.0, 0.0), 'Nend': (1.0, 1.0)}
    }
    n_emission_paras = {
        'N1': [(38.081,11.175), (36.318, 7.376)],
        'N2': [(42,2.828), (60.000, 15.829)],
        'N3': [(60,13.491), (37.476, 8.245)],
        'Nend': [(None, None), (None, None)]
    }

    """Word SLEEP"""
    s_prior_probs = {
        'S1': 0.333,
        'S2': 0.0,
        'S3': 0.0,
        'Send': 0.0
    }
    s_transition_probs = {
        'S1': {'S3': (0.0, 0.0), 'S2': (0.375, 0.214), 'S1': (0.625, 0.786), 'Send': (0.0, 0.0)},
        'S2': {'S1': (0.0, 0.0), 'S2': (0.864, 0.769), 'S3': (0.136, 0.231), 'Send': (0.0, 0.0)},
        'S3': {'S1': (0.0, 0.0), 'S3': (0.0, 0.500), 'S2': (0.0, 0.0), 'Send': (1.0, 0.500)},
        'Send': {'S2': (0.0, 0.0), 'S1': (0.0, 0.0), 'S3': (0.0, 0.0), 'Send': (1.0, 1.0)}
    }
    s_emission_paras = {
        'S1': [(29.5,8.411), (35.357, 7.315)],
        'S2': [(36.182,5.99), (31.462, 5.048)],
        'S3': [(36.667,1.886), (38.333, 7.409)],
        'Send': [(None, None), (None, None)]
    }

    return (a_prior_probs, a_transition_probs, a_emission_paras,
            n_prior_probs, n_transition_probs, n_emission_paras,
            s_prior_probs, s_transition_probs, s_emission_paras)

def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.
    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    Evidence vector is a list of tuples where the first element of each tuple is the right
    hand coordinate and the second element is the right thumb coordinate.
    """

    sequence = []
    probability = 0.0

    # I am going to reuse 1b, going to remove the comments since it's redudant, i will explain any new steps I implemented only

    # just a high level understanding, I think we just have to add another dimension to evidence vector and emission
    # especially when getting the gaussian probability, since we need their exact dimensions.
    # Reason why I think only evidence vector and emission paras, need to be updated for multidimensional viterbi
    # Is because, first of all in 2a, the emission paras had a structure change.
    # I was also looking at the test case for this part 2b, if we look at case3 test case:
    # right_hand_y = [20, 65, 20, 30, 45, 60, 60, 42] right_thumb_y = [56, 74, 48, 41, 38, 55, 56, 44]
    # evidence = list(zip(right_hand_y, right_thumb_y))
    # You can see that now we have two vectors instead of one. it's two dimensional now. there is an X and Y cords.
    # so this is the evidence why I think only wherever we are calculating the gaussian probability, we need an extra
    # variable to account for the extra dimension for these two variables.
    # EDIT: this idea of extra variable is changed, see the explanation below and my reference to it.


    if not evidence_vector:
        return [], 0.0

    K = len(states)
    T = len(evidence_vector)

    T1 = np.zeros((K,T))
    T2 = np.zeros((K,T), dtype=int)

    Pi = []

    for s in range(K):
        current_state = states[s]

        prior_probability = prior_probs[current_state]

        # first part is below, I will comment out the existing code for gaussian probability
        # remember as I said we need another dimension iterative varible now for both evidence_vector and emiision_paras

        # How to combine probabilities from two states? Multiply them together"
        # The last part stuck with me, as well as the other pointer TA gave, which was in the index.
        # so multiply and index, and I realized I should just multiply the gaussian probabilities together
        # since from before as I said, from part 2A that evidence vector, emission and transition all got new dimension
        # so I multiplied the gaussian probability the same way but by itself from the first column of each evidence vector
        # and emission paras to the second column of evidence and emission. That's how you do matrix multiplicaton.

        gaussian_probability = gaussian_prob(evidence_vector[0][0], emission_paras[current_state][0]) * gaussian_prob(evidence_vector[0][1], emission_paras[current_state][1])

        initial_probability = prior_probability * gaussian_probability

        Pi.append(initial_probability)

        T1[s,0] = Pi[s]

    for j in range(1,T):
        for i in range(K):
            max_probability = 0
            best_state = 0
            for k in range(K):

                # it's the same exact logic as previously when I multiplied the gaussian prob together
                # we will be doing that also for transition probs, it was pretty much same, we are multiplying column 1
                # by column 2. But one thing different is the .get method from the dict, I had this before
                # get(states[i], 0), that means get states[i] or default to value 0, if that doesn't exist.
                # here idea is similar since we are working with an tuple now. it's get(states[i], (0,0)), since
                # we default to (0,0) an tuple if states[i] doesn't exist. So that made it work. passed local test cases.

                probability = (T1[k, j-1] * transition_probs[states[k]].get(states[i], (0,0))[0] * transition_probs[states[k]].get(states[i], (0,0))[1] * gaussian_prob(evidence_vector[j][0],emission_paras[states[i]][0]) * gaussian_prob(evidence_vector[j][1], emission_paras[states[i]][1]))

                if probability > max_probability:
                    max_probability = probability
                    best_state = k

            T1[i,j] = max_probability
            T2[i,j] = best_state

    last_maxprob_state = np.argmax(T1[:,T-1])

    sequence.append(states[last_maxprob_state])

    for t in range(T-1, 0, -1):
        last_maxprob_state = T2[last_maxprob_state, t]
        sequence.insert(0, states[last_maxprob_state])

    probability = np.max(T1[:, T-1])

    return sequence, probability