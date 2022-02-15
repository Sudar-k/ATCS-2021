import random


class TicTacToe:
    def __init__(self):
        self.board = [['-' for x in range(3)] for x in range(3)]

    def print_instructions(self):
        print("Welcome to TicTacToe! \n"
              "Player 1 is X and Player 2 is 0 \n"
              "Take turns placing your pieces - the first to 3 in a row wins!")

    def print_board(self):
        print("   " + str(0) + "  " + str(1) + "  " + str(2))
        for i in range(3):
            print(i, " " + "  ".join(self.board[i]))

    def is_valid_move(self, row, col):
        if row > 2 or col > 2:
            return False
        if self.board[row][col] != '-':
            return False
        return True

    def place_player(self, player, row, col):
        self.board[row][col] = player

    def minimax(self, player):
        if self.check_win('O'):
            return 10, None, None
        if self.check_win('X'):
            return -10, None, None
        if self.check_tie():
            return 0, None, None

        if player == 'O':
            best = -1738
            opt_row = -1
            opt_col = -1
            for r in range(3):
                for c in range(3):
                    if self.is_valid_move(r, c):
                        self.place_player('O', r, c)
                        score = self.minimax('X')[0]
                        self.place_player('-', r, c)
                        if best < score:
                            best = score
                            opt_row = r
                            opt_col = c
            return best, opt_row, opt_col

        if player == 'X':
            worst = 1738
            opt_row = -1
            opt_col = -1
            for r in range(3):
                for c in range(3):
                    if self.is_valid_move(r, c):
                        self.place_player('X', r, c)
                        score = self.minimax('O')[0]
                        self.place_player('-', r, c)
                        if worst > score:
                            worst = score
                            opt_row = r
                            opt_col = c
            return worst, opt_row, opt_col

    def take_manual_turn(self, player):

        while True:
            try:
                row = int(input("Enter a row: "))
                col = int(input("Enter a column: "))

            except ValueError:
                print("Please enter a valid integer.")
                continue
            else:
                if self.is_valid_move(row, col):
                    break
                print("Please enter a valid move.")

        self.place_player(player, row, col)

    def take_turn(self, player):
        print(player + "'s Turn")
        if player == "X":
            self.take_manual_turn(player)
        else:
            self.take_minimax_turn(player)

    def take_random_turn(self, player):
        r = random.randint(0, 2)
        c = random.randint(0, 2)
        while not self.is_valid_move(r, c):
            r = random.randint(0, 2)
            c = random.randint(0, 2)

        self.place_player(player, r, c)

    def take_minimax_turn(self, player):
        score, row, col = self.minimax(player)
        self.place_player(player, row, col)

    def check_col_win(self, player):
        win = [player for x in range(3)]
        for i in range(3):
            col = [self.board[0][i], self.board[1][i], self.board[2][i]]
            if col == win:
                return True
        return False

    def check_row_win(self, player):
        win = [player for x in range(3)]
        for i in range(3):
            if self.board[i] == win:
                return True
        return False

    def check_diag_win(self, player):
        win = [player for x in range(3)]
        d1 = [self.board[0][0], self.board[1][1], self.board[2][2]]
        d2 = [self.board[0][2], self.board[1][1], self.board[2][0]]
        return win == d1 or win == d2

    def check_win(self, player):
        if self.check_diag_win(player) or self.check_col_win(player) or self.check_row_win(
                player):
            return True
        return False

    def check_tie(self):
        return sum(row.count('-') for row in self.board) == 0

    def play_game(self):
        self.print_instructions()
        self.print_board()
        player = 'O'
        while not self.check_win(player) and not self.check_tie():
            if player == 'O':
                player = 'X'
            else:
                player = 'O'

            self.take_turn(player)
            self.print_board()

        if self.check_win(player):
            print("Player", player, "has won!")
        else:
            print("The game has ended in a tie!")