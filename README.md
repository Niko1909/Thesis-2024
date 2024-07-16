# Thesis: Quantifying Game Difficulty for Humans vs. Computers.
This repository contains the primary scripts used for my thesis, all done in Python:
* ***fen_to_matrix.py:*** Contains functions that take as input FEN-formatted string which represents a chess position, and return as output a matrix of the equivalent chess position in the form of a 3-D list. fen_to_matrix() returns an 8x8x(6+1) matrix, where the first 6 "layers" of the matrix are one-hot encodings of each unique chess piece, and the last 1 "layer" of the matrix represents which colour the piece is, 0 being black and 1 being white. fen_to_matrix2() returns an 8x8x12 matrix, where the first 6 "layers" of the matrix are one-hot encodings of each unique black piece and the last 6 "layers" of the matrix are one-hot encodings of each unique white piece.
* ***pychess-script.py:*** Converts FEN-formatted string chess positions into a ground truth score of the position, analyzed by Leela Chess Zero which uses the LD2 weights also provided in the repository.  Utilizes the fen_to_matrix() function, the PyChess library to use lc0 within Python, and pickle to save the ground truth labels. The chess FEN positions came from a [dataset on Kaggle](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations).
* ***training.py:*** Trains several "naive" CNN models on the ground truths above: given a matrix chess position, predict the score for white. This is the primary experiment of the thesis, performing a distillation of Leela Chess Zero, attempting to transfer its knowledge into much smaller models to see if they can predict at a similar level.
* ***model_eval.py:*** This file loads the weights of the trained models from the training.py, and evaluates their performance on a test set. 

### Requirements
To run the above scripts, you will need to have your own version of Leela Chess Zero to generate the ground truth labels, in which you can use the weights provided in the repository or other weights that you can find on [lc0's website](https://lczero.org/dev/wiki/best-nets-for-lc0/). Additionally, the below libraries are required to run the scripts:
* [PyTorch](https://pytorch.org/)
* [NumPy](https://numpy.org/)
* [python-chess](https://github.com/niklasf/python-chess)
* [matplotlib](https://matplotlib.org/stable/install/index.html)
