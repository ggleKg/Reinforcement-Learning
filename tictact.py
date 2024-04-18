import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import time

# TikTacToe environement
class TicTacToe:
    def __init__(self):
        self.board = [' ']*9
        self.player = 'X'
        self.winner = None
        self.done = False

    def reset(self):
        self.board = [' ']*9
        self.player = 'X'
        self.winner = None
        self.done = False

    def print_board(self):
        for i in range(0, 9, 3):
            print('|'.join(self.board[i:i+3]))

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.player
            if self.check_winner():
                self.winner = self.player
                self.done = True
            elif ' ' not in self.board:
                self.done = True
            else:
                if self.player == 'X':
                    self.player = 'O'
                else:
                    self.player = 'X'
            return True
        else:
            return False

    def check_winner(self):
        win_conditions = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                          [0, 3, 6], [1, 4, 7], [2, 5, 8],
                          [0, 4, 8], [2, 4, 6]]

        for condition in win_conditions:
            if self.board[condition[0]] == self.board[condition[1]] == self.board[condition[2]] != ' ':
                return True
        return False


############ Q Learning agent implementation
class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.last_state = None
        self.last_action = None

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else :
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q_value = max(q_values)
            return available_actions[q_values.index(max_q_value)]

    def learn(self, env, state, action, reward, next_state):
        if self.last_state is not None:
            old_q_value = self.get_q_value(state, action)
            if env.done:
                next_max_q_value = 0
            else:
                next_max_q_value = max([self.get_q_value(next_state, next_action) for next_action in env.available_moves()])
            new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q_value - old_q_value)
            self.q_table[(state, action)] = new_q_value
        self.last_state = state
        self.last_action = action

    def reset(self):
        self.last_state = None
        self.last_action = None


#### Training QL
def train_q_learning(nbr_epoch):
    env = TicTacToe()
    agent = QLearningAgent()

    for i in range(nbr_epoch):
        env.reset()
        agent.reset()
        state = ''.join(env.board)
        while not env.done:
            available_actions = env.available_moves()
            action = agent.choose_action(state, available_actions)
            env.make_move(action)
            next_state = ''.join(env.board)
            reward = 0
            if env.done:
                if env.winner == 'X':
                    reward = 1
                elif env.winner == 'O':
                    reward = -1
            agent.learn(env, state, action, reward, next_state)
            state = next_state
        
    return agent

# Playing against QL
def play_vs_q_learning(agent):
    env = TicTacToe()
    state = ''.join(env.board)
    while not env.done:
        env.print_board()
        print("Enter your move (0-8):")
        player_move = int(input())
        env.make_move(player_move)
        if env.done:
            break
        state = ''.join(env.board)
        available_actions = env.available_moves()
        agent_move = agent.choose_action(state, available_actions)
        env.make_move(agent_move)
        state = ''.join(env.board)
    env.print_board()
    if env.winner:
        print("Winner is:", env.winner)
    else:
        print("It's a draw!")


######## Question 2 :  Minimax


def evaluate_state(env):
    if env.check_winner():
        if env.winner == 'X':
            return -1
        elif env.winner == 'O':
            return 1
    return 0

##Minimax function,returnsthe best columns
def minimax(env, maximizing_player):
    if env.done or len(env.available_moves()) == 0:
        return evaluate_state(env)

    if maximizing_player:
        max_eval = float('-inf')
        for move in env.available_moves():
            # Créer une copie de l'environnement pour évaluer le mouvement
            env_copy = copy.deepcopy(env)
            env_copy.make_move(move)
            eval = minimax(env_copy, False)  # Le joueur suivant est un joueur minimisant
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in env.available_moves():
            # Créer une copie de l'environnement pour évaluer le mouvement
            env_copy = copy.deepcopy(env)
            env_copy.make_move(move)
            eval = minimax(env_copy, True)  # Le joueur suivant est un joueur maximisant
            min_eval = min(min_eval, eval)
        return min_eval

def minimax_prunning(env, depth, alpha, beta, maximizing_player):
    if env.done or depth == 0:
        return evaluate_state(env)

    if maximizing_player:
        max_eval = float('-inf')
        for move in env.available_moves():
            # Effectuer une copie profonde de l'environnement
            env_copy = copy.deepcopy(env)
            # Effectuer le mouvement sur la copie
            env_copy.make_move(move)
            # Appeler récursivement Minimax pour le joueur suivant (minimisant)
            eval = minimax_prunning(env_copy, depth - 1, alpha, beta, False)
            # Mettre à jour l'alpha
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            # Effectuer l'élagage
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in env.available_moves():
            # Effectuer une copie profonde de l'environnement
            env_copy = copy.deepcopy(env)
            # Effectuer le mouvement sur la copie
            env_copy.make_move(move)
            # Appeler récursivement Minimax pour le joueur suivant (maximisant)
            eval = minimax_prunning(env_copy, depth - 1, alpha, beta, True)
            # Mettre à jour le beta
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            # Effectuer l'élagage
            if beta <= alpha:
                break
        return min_eval




def minimax_prunning_agent(env):
    best_move = None
    best_eval = -float('inf')
    for action in env.available_moves():
        env_copy = copy.deepcopy(env)
        env_copy.make_move(action)
        #eval = minimax(env, False)
        eval = minimax_prunning(env, depth=7, alpha=-100 , beta= 100, maximizing_player=False)

        env_copy.board[action] = ' '  # annuler le mouvement
        if eval > best_eval:
            best_eval = eval
            best_move = action
    return best_move


def minimax_agent(env):
    best_move = None
    best_eval = -float('inf')
    for action in env.available_moves():
        env_copy = copy.deepcopy(env)
        env_copy.make_move(action)
        #eval = minimax(env, False)
        eval = minimax(env, maximizing_player=False)

        env_copy.board[action] = ' '  # annuler le mouvement
        if eval > best_eval:
            best_eval = eval
            best_move = action
    return best_move





############# Question 3 : Default   Opponent
def find_winning_or_blocking_move(env):
    available_moves = env.available_moves()
    
    # Recherche de mouvements gagnants
    for move in available_moves:
        env.make_move(move)
        if env.check_winner():
            env.board[move] = ' '
            return move
        else:
            env.board[move] = ' '
    
    # Recherche de mouvements bloquants
    for move in available_moves:
        if env.player == 'O' :   
            env.board[move] = 'X'
        else : 
            env.board[move] = 'O'
            
        if env.check_winner():
            env.make_move(move)
            env.board[move] = ' '
            return move
        else:
            env.board[move] = ' '
    
    # Si aucun mouvement gagnant ou bloquant n'est trouvé, retourne un mouvement aléatoire
    return random.choice(available_moves)





####################         Question 4


##### Q Learning vs Random Game
def q_learning_vs_random(agent):
    env = TicTacToe()
    state = ''.join(env.board)
    while not env.done:
        env.print_board()
        if env.player == 'O':
            #print("Enter your move (0-8):")
            #player_move = int(input())
            #player_move = random.choice(env.available_moves())
            player_move = find_winning_or_blocking_move(env)
            env.make_move(player_move)
        else:
            agent_move = agent.choose_action(state, env.available_moves())
            print("AI plays:", agent_move)
            env.make_move(agent_move)
        state = ''.join(env.board)
    env.print_board()
    if env.winner:
        return env.winner
    else:
        return 'draw'


##### Q Learning vs minimax

def q_learning_vs_minimax(agent):
    env = TicTacToe()
    state = ''.join(env.board)
    while not env.done:
        env.print_board()
        if env.player == 'O':
            #print("Enter your move (0-8):")
            #player_move = int(input())
            #player_move = random.choice(env.available_moves())
            #player_move = find_winning_or_blocking_move(env)
            #env.make_move(player_move)
            player_move = minimax_agent(env)
            env.make_move(player_move)
        else:
            agent_move = agent.choose_action(state, env.available_moves())
            print("AI plays:", agent_move)
            env.make_move(agent_move)
        state = ''.join(env.board)
    env.print_board()
    if env.winner:
        return env.winner
    else:
        return 'draw'
    

##### Random vs minimax

def random_vs_minimax():
    env = TicTacToe()
    state = ''.join(env.board)
    while not env.done:
        env.print_board()
        if env.player == 'O':
            #print("Enter your move (0-8):")
            #player_move = int(input())
            #player_move = random.choice(env.available_moves())
            #player_move = find_winning_or_blocking_move(env)
            #env.make_move(player_move)
            player_move = minimax_agent(env)
            env.make_move(player_move)
        else:
            player_move = find_winning_or_blocking_move(env)
            print("AI plays:", player_move)
            env.make_move(player_move)
        state = ''.join(env.board)
    env.print_board()
    if env.winner:
        return env.winner
    else:
        return 'draw'
    

def random_vs_minimax_prunning():
    env = TicTacToe()
    state = ''.join(env.board)
    while not env.done:
        env.print_board()
        if env.player == 'O':
            #print("Enter your move (0-8):")
            #player_move = int(input())
            #player_move = random.choice(env.available_moves())
            #player_move = find_winning_or_blocking_move(env)
            #env.make_move(player_move)
            player_move = minimax_prunning_agent(env)
            env.make_move(player_move)
        else:
            player_move = find_winning_or_blocking_move(env)
            print("AI plays:", player_move)
            env.make_move(player_move)
        state = ''.join(env.board)
    env.print_board()
    if env.winner:
        return env.winner
    else:
        return 'draw'
    




##### Q Learninng  games
if __name__ == "__main__":

#     #L_nbr_epoch = [1,30, 50, 100, 500, 1000, 3000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000 ]
#     #L_nbr_epoch = [1,30, 50, 100, 500, 1000, 3000, 5000, 10000, 30000, 50000, 100000, 150000]
#     L_nbr_epoch = [10,100]


#     Lo =  []
#     Lx =  []
#     Ldraw =  []
#     Lgames =[]
#     L_time_train_Q = []

#     for epoch in L_nbr_epoch :

#         start_time = time.time()
#         trained_agent = train_q_learning(epoch)
#         end_time = time.time()
#         q_train_execution_time = end_time - start_time
#         L_time_train_Q.append(q_train_execution_time)

#         nbr_win_o = 0 
#         nbr_win_x = 0 
#         nbr_draw = 0 

#         L_time_game = []

#         nbr_game = 2  

#         for game in range(1,nbr_game +1) :
#             print("Partie " +  str(game))
#             start_time = time.time()

#             ### We just choose if we want to play against random or minimax
#             winner = q_learning_vs_random(trained_agent)
#             #winner = q_learning_vs_minimax(trained_agent)

#             end_time = time.time()
#             q_game_execution_time = end_time - start_time
#             L_time_game.append(q_game_execution_time)

#             if winner == 'X' :
#                 nbr_win_x = nbr_win_x +1

#             if winner == 'O' : 
#                 nbr_win_o = nbr_win_o + 1
#             if winner == 'draw':
#                 nbr_draw = nbr_draw + 1
            
#         Lx.append(nbr_win_x/nbr_game)
#         Lo.append(nbr_win_o/nbr_game)
#         Ldraw.append(nbr_draw/nbr_game)
#         #Lgames.append(game)

    
#     print("Aevarge Time Game :" + str(np.mean(L_time_game)))
#     print("Aevarge Wins X :" + str(np.mean(Lx)))
#     print("Aevarge Wins O :" + str(np.mean(Lo)))
#     print("Aevarge drw :" + str(np.mean(Ldraw)))

#     # Plots Wins rate
#     plt.subplot(1, 2, 1)  
#     plt.plot(L_nbr_epoch, Lx, label='X wins')
#     plt.plot(L_nbr_epoch, Lo, label='O wins')
#     plt.plot(L_nbr_epoch, Ldraw, label='draw')
#     plt.xlabel(' Number of epoch ')
#     plt.ylabel(' Wins rate ')

#     plt.title('Q Learning vs MiniMax - Win Rates')
#     plt.legend()

#     # Time Graphs
#     plt.subplot(1, 2, 2)  
#     plt.plot(L_nbr_epoch, L_time_train_Q, label='Time execution training')
#     plt.xlabel(' Number of epoch ')
#     plt.ylabel(' Time Training (en s) ')
#     plt.title(' Time Execution of games - Tictactoe')
#     plt.legend()

#     plt.tight_layout()  
#     plt.show()



# # MiniMax vs Random
if __name__ == "__main__":


    Lo =  []
    Lx =  []
    Ldraw =  []
    Lgames =[]
    L_time_train_Q = []

    L_time_game = []

    ### Number of games have to be high enough to take conclusion
    nbr_game = 3

    nbr_win_o = 0 
    nbr_win_x = 0 
    nbr_draw = 0 



    for game in range(1,nbr_game +1) :
        print("Partie " +  str(game))
        start_time = time.time()
        #winner = random_vs_minimax()
        winner = random_vs_minimax_prunning()

        end_time = time.time()
        game_execution_time = end_time - start_time
        L_time_game.append(game_execution_time)

        if winner == 'X' :
            nbr_win_x = nbr_win_x +1

        if winner == 'O' : 
            nbr_win_o = nbr_win_o + 1
        if winner == 'draw':
            nbr_draw = nbr_draw + 1

        
        Lx.append(nbr_win_x/game)
        Lo.append(nbr_win_o/game)
        Ldraw.append(nbr_draw/game)
        Lgames.append(game)

    
    print("Aevarge Time Game :" + str(np.mean(L_time_game)))
    print("Aevarge Wins X :" + str(np.mean(Lx)))
    print("Aevarge Wins O :" + str(np.mean(Lo)))
    print("Aevarge drw :" + str(np.mean(Ldraw)))


    # Plots Wins rate
    plt.subplot(1, 2, 1)  
    plt.plot(Lgames, Lx, label='X wins')
    plt.plot(Lgames, Lo, label='O wins')
    plt.plot(Lgames, Ldraw, label='draw')
    plt.xlabel(' Number of games ')
    plt.ylabel(' Wins rate ')

    plt.title('MiniMax vs Random - Win Rates')
    plt.legend()

    # Time Graphs
    plt.subplot(1, 2, 2)  
    plt.plot(Lgames, L_time_game, label='Time execution gaming')
    plt.xlabel(' Number of game ')
    plt.ylabel(' Time Gaming (en s) ')
    plt.title(' MiniMax Time Execution Game - Tictactoe')
    plt.legend()

    plt.tight_layout()  
    plt.show()
        





# # Entraîner l'agent Q-learning et jouer contre lui
# if __name__ == "__main__":
#     trained_agent = train_q_learning(10000)
#     #winner = q_learning_vs_random(trained_agent)
#     winner = q_learning_vs_minimax(trained_agent)
#     print(winner)

#     # Jouer contre l'agent Minimax
#     play_vs_minimax()

