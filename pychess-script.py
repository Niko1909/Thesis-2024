import chess
import chess.engine
from fen_to_matrix import fen_to_matrix
import pickle

def analyse(board, engine):
    result = engine.analyse(board, chess.engine.Limit(time=10))
    print("Score:", result["score"])
    return result

def main():

    engine = chess.engine.SimpleEngine.popen_uci("/home/niko/lc0/lc0/build/release/lc0")
    fen_data = "FEN_10-15k.txt"
    #fen_data = "FEN_test.txt"
    with open(fen_data, "r") as file:
        fen_positions = file.readlines()
    
    pos_score = [] # (fen, score) where score is the cp of white's position
    counter = 0
    for fen_position in fen_positions:
        fen_position = fen_position.strip()
        board = chess.Board(fen_position)
        result = analyse(board, engine)
        matrix = fen_to_matrix(fen_position)
        pos_score.append([fen_position, matrix, result["score"].white().score()])
        counter += 1
        if counter % 10 == 0:
            print(f"Processed {counter} positions")

    with open("pos-score-data10-15k.pkl", "wb") as file:
        pickle.dump(pos_score, file)

    engine.quit()

main()

