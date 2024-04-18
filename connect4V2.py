import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict


ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY =0


####### Connect 4


def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def drop_piece(board, row, col, piece):
    board[row][col] = piece

# def is_valid_location(board, col):
#     return board[ROW_COUNT-1][col] == 0

# def get_next_open_row(board, col):
#     for r in range(ROW_COUNT):
#         if board[r][col] == 0:
#             return r


def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return None  # Retourne None si la colonne est pleine

def is_valid_location(board, col):
    return board[0][col] == EMPTY and get_next_open_row(board, col) is not None


# Add this function to check if the board is full
def is_board_full(board):
    return not any(0 in row for row in board)


def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
            
    return False

def evaluate_window(window, piece):
    score = 0
    opponent_piece = 1 if piece == 2 else 2

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opponent_piece) == 3 and window.count(0) == 1:
        score -= 4

    return score

def evaluate_board(board, piece):
    score = 0

    # Score horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + 4]
            score += evaluate_window(window, piece)

    # Score vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + 4]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score



#############################                         Question 2


def minimax(board, depth, maximizing_player, piece):
    if depth == 0:
        return None, evaluate_board(board, piece)

    if maximizing_player:
        value = -math.inf
        column = random.choice([col for col in range(COLUMN_COUNT) if is_valid_location(board, col)])
        for col in range(COLUMN_COUNT):
            if is_valid_location(board, col):
                temp_board = board.copy()
                row = get_next_open_row(temp_board, col)
                drop_piece(temp_board, row, col, piece)
                new_score = minimax(temp_board, depth - 1, False, piece)[1]
                if new_score > value:
                    value = new_score
                    column = col
        return column, value
    else:
        value = math.inf
        column = random.choice([col for col in range(COLUMN_COUNT) if is_valid_location(board, col)])
        for col in range(COLUMN_COUNT):
            if is_valid_location(board, col):
                temp_board = board.copy()
                row = get_next_open_row(temp_board, col)
                drop_piece(temp_board, row, col, 1 if piece == 2 else 2)
                new_score = minimax(temp_board, depth - 1, True, piece)[1]
                if new_score < value:
                    value = new_score
                    column = col

        return column, value
    

def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player, piece):
    if depth == 0:
        return None, evaluate_board(board, piece)

    if maximizing_player:
        value = -math.inf
        column = None
        for col in range(COLUMN_COUNT):
            if is_valid_location(board, col):
                temp_board = board.copy()
                row = get_next_open_row(temp_board, col)
                drop_piece(temp_board, row, col, piece)
                _, new_score = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, False, piece)
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return column, value
    else:
        value = math.inf
        column = None
        for col in range(COLUMN_COUNT):
            if is_valid_location(board, col):
                temp_board = board.copy()
                row = get_next_open_row(temp_board, col)
                drop_piece(temp_board, row, col, 1 if piece == 2 else 2)
                _, new_score = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, True, piece)
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return column, value







##############################                          Question 3 


## Default Opponent
def opponent_move(board, player_piece):
    # Check for winning moves
    for col in range(COLUMN_COUNT):
        temp_board = board.copy()
        if is_valid_location(temp_board, col):
            row = get_next_open_row(temp_board, col)
            drop_piece(temp_board, row, col, player_piece)
            if winning_move(temp_board, player_piece):
                return col
    
    # Check for blocking moves
    opponent_piece = 1 if player_piece == 2 else 2
    for col in range(COLUMN_COUNT):
        temp_board = board.copy()
        if is_valid_location(temp_board, col):
            row = get_next_open_row(temp_board, col)
            drop_piece(temp_board, row, col, opponent_piece)
            if winning_move(temp_board, opponent_piece):
                return col
    
    # Otherwise, make a random move
    valid_moves = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    return random.choice(valid_moves)




################################                      Question 4



def play_game():
    board = create_board()
    print_board(board)
    game_over = False
    turn = 0

    while not game_over:
        # Player's turn
        if turn == 0:
            #col = int(input("Player 1, choose a column (0-6): "))
            col, _ = minimax(board, 4, True, 1)
            #col = opponent_move(board, 1)


            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 1)
                if winning_move(board, 1):
                    print("Player 1 wins!")
                    game_over = True
                    return "Minimax win"

        # AI's turn
        else:
            #col, _ = minimax(board, 4, True, 2)
            col = opponent_move(board, 2)

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 2)
                if winning_move(board, 2):
                    print("Player 2 wins!")
                    game_over = True
                    return "Default opp win"

        print_board(board)
        turn += 1
        turn %= 2

        if is_board_full(board):
            print("The game is a tie!")
            game_over = True
            return "Draw"


#winner = play_game()


def results_Minimax_vs_random(number_games):

    wins = 0
    losses =0
    draws = 0


    Lwins = []
    Llosses = []
    Ldraws = []
    L_nbr_game = []

    L_time_game= []

    for game in range(1,number_games+1):
        start_time = time.time()
        winner = play_game()
        end_time = time.time()
        game_time = end_time - start_time

        L_time_game.append(game_time)


        if winner == "Minimax win" :
            wins = wins +1
        if winner == "Default opp win" :
            losses = losses +1
        if winner == "Draw":
            draws = draws +1

        Lwins.append(wins/game)
        Llosses.append(losses/game)
        Ldraws.append(draws/game)
        L_nbr_game.append(game)
        print("Game" + str(game))




    print("Aevarge Time Game :" + str(np.mean(L_time_game)))
    print("Aevarge Wins MiniMax :" + str(np.mean(Lwins)))
    print("Aevarge Wins Default Opponent :" + str(np.mean(Llosses)))
    print("Aevarge draw :" + str(np.mean(Ldraws)))


    
    plt.subplot(1,2,1)
    plt.plot(L_nbr_game,Lwins , label='wins minimax')
    plt.plot(L_nbr_game,Llosses , label='losses minimax')
    plt.plot(L_nbr_game,Ldraws , label='draws ')
    plt.xlabel('Number of games')
    plt.ylabel('rates results')
    plt.title('MiniMax vs Default Opponent')
    plt.legend()


    plt.subplot(1,2,2)
    plt.plot(L_nbr_game,L_time_game, label='Time ')
    plt.xlabel('Number of game')
    plt.ylabel('Time execution')
    plt.title('Execution Time of a game')

    plt.show()



########### Playing MiniMax versus random Opponent
######## To play with Minimax with prunning against default opponent, we just need to replace the name of the function
####### Minimax by minimax_alpha_beta in the play_game function or in the play_ql_minimax function if we want play against QL

# number_games = 2
# results_Minimax_vs_random(number_games)






################ Q Learning versus

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.2, rows=6, cols=7):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rows = rows
        self.cols = cols

    def create_board(self):
        return np.zeros((self.rows, self.cols), dtype=int)

    def is_valid_location(self, board, col):
        return board[0][col] == 0

    def get_next_open_row(self, board, col):
        for r in range(self.rows-1, -1, -1):
            if board[r][col] == 0:
                return r

    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def is_winning_move(self, board, piece):
        # Horizontal, vertical, and diagonal check here
        # Omitted for brevity, assume definition similar to previous examples
        return winning_move(board, piece)

    def get_legal_actions(self, board):
        return [c for c in range(self.cols) if self.is_valid_location(board, c)]

    def epsilon_greedy_policy(self, board):

        legal_actions = self.get_legal_actions(board)
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(self.get_legal_actions(board))
        else:
            state_key = board.tostring()
            legal_actions = self.get_legal_actions(board)
            qs = [self.Q[(state_key, a)] for a in legal_actions]
            max_q = max(qs)
            max_actions = [a for a in legal_actions if self.Q[(state_key, a)] == max_q]
            return random.choice(max_actions)

    def update_q_value(self, board, action, reward, next_board):
        current_state_key = board.tostring()
        next_state_key = next_board.tostring()
        next_actions = self.get_legal_actions(next_board)
        if next_actions:
            max_next_q = max(self.Q[(next_state_key, a)] for a in next_actions)
        else:
            max_next_q = 0
        self.Q[(current_state_key, action)] += self.alpha * (reward + self.gamma * max_next_q - self.Q[(current_state_key, action)])

    def train(self, episodes):
        for _ in range(episodes):
            board = self.create_board()
            done = False
            while not done:
                action = self.epsilon_greedy_policy(board)
                if action is None:
                    break
                row = self.get_next_open_row(board, action)
                self.drop_piece(board, row, action, 1)
                if self.is_winning_move(board, 1):
                    reward = 1
                    done = True
                else:
                    reward = 0.5
                    if np.all(board != 0):
                        done = True
                        reward = 0  # Reward for a draw could be considered
                    else:
                        # Assuming opponent plays randomly here
                        opp_action = random.choice(self.get_legal_actions(board))
                        #opp_action = opponent_move(board, 2)
                        opp_row = self.get_next_open_row(board, opp_action)
                        self.drop_piece(board, opp_row, opp_action, 2)
                        if self.is_winning_move(board, 2):
                            reward = -2
                            done = True

                self.update_q_value(board, action, reward, board)


def play_ql_random(agent, opponent, episodes):
    wins = 0
    losses = 0
    draws = 0

    Lwins = []
    Llosses= []
    Ldraws = []
    Lepi = []

    for epi in range(1,episodes+1):
        board = agent.create_board()
        done = False
        turn = random.choice([1, 2])  # Randomly decide who starts

        while not done:
            if turn == 1:  # Agent's turn
                action = agent.epsilon_greedy_policy(board)
                if action is None:
                    break
                row = agent.get_next_open_row(board, action)
                agent.drop_piece(board, row, action, 1)
                if agent.is_winning_move(board, 1):
                    wins += 1
                    done = True
                turn = 2
            else:  # Random opponent's turn
                actions = agent.get_legal_actions(board)
                if not actions:
                    break
                #action = opponent_move(board, 2)
                action = random.choice(actions)
                row = agent.get_next_open_row(board, action)
                agent.drop_piece(board, row, action, 2)
                if agent.is_winning_move(board, 2):
                    losses += 1
                    done = True
                turn = 1
            
            if not done and np.all(board != 0):  # Board is full, draw
                draws += 1
                done = True

        
        Lwins.append(wins/epi)
        Llosses.append(losses/epi)
        Ldraws.append(draws/epi)
        Lepi.append(epi)

    return wins, losses, draws, Lwins, Llosses, Ldraws, Lepi


def plot_results_QL_vs_random():


    # Ltrain = [10, 1000, 10000, 100000,200000, 300000,400000, 500000,600000, 700000]
    # Ltrain_epi = [10, 1000-10, 10000-1000, 100000-10000, 200000-100000, 300000-200000,400000-300000, 500000-400000, 600000-500000, 700000-600000]
    
    # We use this list just for the demo otherwise we use the one above
    Ltrain = [1,3]
    Ltrain_epi = [1,3]

    Lwins_r = []
    Llos_r =[]
    Ldraws_r = []

    agent = QLearningAgent()

    total_training_time = 0
    L_training_time = []

    number_game=2

    for epi in Ltrain_epi :
        start_time = time.time()
        agent.train(epi)  # Train for 1000 episodes
        end_time = time.time()
        q_train_execution_time = end_time - start_time
        total_training_time = total_training_time + q_train_execution_time
        L_training_time.append(total_training_time)


        # Test the trained agent against a random opponent
        wins, losses, draws, Lwins, Llosses, Ldraws, Lepi = play_ql_random(agent, None, number_game)
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")


        Lwins_r.append(wins/number_game)
        Llos_r.append(losses/number_game)
        Ldraws_r.append(draws/number_game)

    plt.subplot(1, 2, 1)  
    plt.plot(Ltrain, Lwins_r, label= 'wins')
    plt.plot(Ltrain, Llos_r, label= 'losses')
    plt.plot(Ltrain, Ldraws_r, label= 'draws')
    plt.xlabel('episodes')
    plt.ylabel('result rates')
    plt.title('Q Learning vs default opponent')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(Ltrain, L_training_time, label='QL Training Time vs number of epochs ')  
    plt.xlabel('episodes')
    plt.ylabel('Time training ( s) ')
    plt.title('Q Learning vs default opponent')

    plt.show()



### Plot results Q Learning vs Random
#plot_results_QL_vs_random()


def play_ql_minimax(agent, opponent, episodes):
    wins = 0
    losses = 0
    draws = 0

    Lwins = []
    Llosses= []
    Ldraws = []
    Lepi = []

    for epi in range(1,episodes+1):
        board = agent.create_board()
        done = False
        turn = random.choice([1, 2])  # Randomly decide who starts

        while not done:
            if turn == 1:  # Agent's turn
                action = agent.epsilon_greedy_policy(board)
                if action is None:
                    break
                row = agent.get_next_open_row(board, action)
                agent.drop_piece(board, row, action, 1)
                if agent.is_winning_move(board, 1):
                    wins += 1
                    done = True
                turn = 2
            else:  # Random opponent's turn
                actions = agent.get_legal_actions(board)
                if not actions:
                    break
                #action = opponent_move(board, 2)
                #action, _ = minimax(board, 4, True, 2)
                action, _ = minimax_alpha_beta(board, 4, - 100, 100, True, 2)
                row = agent.get_next_open_row(board, action)
                agent.drop_piece(board, row, action, 2)
                if agent.is_winning_move(board, 2):
                    losses += 1
                    done = True
                turn = 1
            
            if not done and np.all(board != 0):  # Board is full, draw
                draws += 1
                done = True

        print("Game Number" + str(epi))
        Lwins.append(wins/epi)
        Llosses.append(losses/epi)
        Ldraws.append(draws/epi)
        Lepi.append(epi)

    return wins, losses, draws, Lwins, Llosses, Ldraws, Lepi



def plot_results_QL_vs_Minimax():
        
    # Ltrain = [10,100000, 300000]
    # Ltrain_epi = [10, 100000-10, 300000-100000]

    Ltrain = [1,2]
    Ltrain_epi = [1,2]


    Lwins_r = []
    Llos_r =[]
    Ldraws_r = []

    agent = QLearningAgent()

    total_training_time = 0
    L_training_time = []

    number_game= 1

    for epi in Ltrain_epi :

        print("Episodes" + str(epi))

        start_time = time.time()
        agent.train(epi)  # Train for 1000 episodes
        end_time = time.time()
        q_train_execution_time = end_time - start_time
        total_training_time = total_training_time + q_train_execution_time
        L_training_time.append(total_training_time)


        # Test the trained agent against a random opponent
        wins, losses, draws, Lwins, Llosses, Ldraws, Lepi = play_ql_minimax(agent, None, number_game)
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")


        Lwins_r.append(wins/number_game)
        Llos_r.append(losses/number_game)
        Ldraws_r.append(draws/number_game)

    plt.subplot(1, 2, 1)  
    plt.plot(Ltrain, Lwins_r, label= 'wins')
    plt.plot(Ltrain, Llos_r, label= 'losses')
    plt.plot(Ltrain, Ldraws_r, label= 'draws')
    plt.xlabel('episodes')
    plt.ylabel('result rates')
    plt.title('Q Learning vs MiniMax Prunning')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(Ltrain, L_training_time, label='QL Training Time vs number of epochs ')  
    plt.xlabel('episodes')
    plt.ylabel('Time training ( s) ')
    plt.title('Q Learning vs MiniMax Prunning')

    plt.show()



#### Plot results Q Learning vs MiniMax
plot_results_QL_vs_Minimax()