import numpy as np

def fen_to_matrix(fen):
    piece_dict = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                  'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

    matrix = np.zeros((8, 8, 7), dtype=int)
    color = np.zeros((8,8))

    # Split FEN string into sections
    fen_parts = fen.split()
    board_rows = fen_parts[0].split('/')

    # Translate FEN into matrix
    for row_index, row in enumerate(board_rows):
        file_index = 0
        for char in row:
            if char.isdigit():
                file_index += int(char)
            else:
                piece_index = piece_dict[char]
                matrix[7 - row_index, file_index, piece_index] = 1
                matrix[7 - row_index, file_index, 6] = 0 if char.islower() else 1
                file_index += 1

    return matrix

def fen_to_matrix2(fen):
    piece_dict = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                  'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11}

    matrix = np.zeros((8, 8, 12), dtype=int)
    color = np.zeros((8,8))

    # Split FEN string into sections
    fen_parts = fen.split()
    board_rows = fen_parts[0].split('/')

    # Translate FEN into matrix
    for row_index, row in enumerate(board_rows):
        file_index = 0
        for char in row:
            if char.isdigit():
                file_index += int(char)
            else:
                piece_index = piece_dict[char]
                matrix[7 - row_index, file_index, piece_index] = 1
                #matrix[7 - row_index, file_index, 6] = 0 if char.islower() else 1
                file_index += 1

    return matrix   


# Example usage:
fen_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"
fen_position1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
fen2 = "r1b1k1nr/pppp1ppp/8/8/2B1P1nq/3P2P1/PPP2b1P/RNB1K2R w KQkq - 0 8"
chess_matrix = fen_to_matrix(fen_position)
chess_matrix1 = fen_to_matrix2(fen_position)
# Print the resulting matrix
#print(chess_matrix)

# Print the resulting matrix
# for row in chess_matrix:
#     for square in row:
#         print(square, end=" ")
#     print()

# # # Print the resulting matrix
# chess_matrix = np.array(chess_matrix)
# print(chess_matrix.shape)
# for i in range(chess_matrix.shape[2]):
#     print(chess_matrix[:,:,i])

# print(chess_matrix1.shape)
# for i in range(chess_matrix1.shape[2]):
#     print(chess_matrix1[:,:,i])