
# Project Overview: Hidden Markov Models for ASL Recognition

This project demonstrates the application of Hidden Markov Models (HMMs) to recognize words from video sequences of American Sign Language (ASL). HMMs are statistical models that assume an underlying process to be a Markov process with hidden states. They are widely used in areas such as natural language processing, bioinformatics, and speech recognition. The hidden states, which are not directly observable, are inferred from the observable events through the model's structure of states and transitions.

## Introduction to Hidden Markov Models

In HMMs, each hidden state is independent of all other states except for its immediate predecessor, which allows the model to simplify complex probabilistic models where states are interdependent. The model is defined by:

- **States**: In the context of ASL recognition, each state could represent a specific position or movement in sign language.
- **Observations**: These are the data points that are visible, such as the coordinates of the hands in a video frame.
- **Transitions**: The probabilities of moving from one state to another.
- **Emissions**: The probabilities of observing a specific visible state given a particular hidden state.

## Project Details

### Model Training

The training involves calculating the parameters of the HMM:
- **Transition Probabilities**: These are the probabilities of transitioning from one state to another. For instance, the probability of moving from the "beginning of a sign" state to the "middle of a sign" state.
- **Emission Probabilities**: Given a hidden state, these are the probabilities of observing each possible visible state. This is often modeled using Gaussian distributions characterized by a mean (the average value of observations in that state) and a standard deviation (a measure of the variability of observations in that state).

### Viterbi Algorithm

To decode the observed sequences of ASL signs into the most likely sequence of hidden states, the Viterbi algorithm is employed. This dynamic programming algorithm calculates the most likely sequence of hidden states based on the observed events. It involves:
- **Initialization**: Starting with the known probabilities of initial states.
- **Recursion**: Progressively calculating the probabilities of each subsequent state being the most likely, given the previous state and the current observation.
- **Termination**: Determining the most likely sequence of states at the end of the observation period.

## Implementation and Usage

### Part 1: One-Dimensional HMM
This part involves using only the y-coordinates of the right hand. The model recognizes words based solely on this one-dimensional data.

### Part 2: Multidimensional HMM
Expanding on Part 1, this model incorporates additional features—both the right hand and right thumb y-coordinates—enhancing its complexity and accuracy.

## Example Words and Recognition

The models have been trained to recognize specific words in ASL, such as “ALLIGATOR,” "NUTS," and "SLEEP," using data from isolated sign language video sequences. Each word is associated with a unique sequence of hand movements and positions, represented by hidden states in the HMM.

This project showcases the power of probabilistic models in interpreting sequences and provides a foundation for further exploration into more complex models and larger datasets in sign language recognition.
