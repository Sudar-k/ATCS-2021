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
        return

    def is_valid_move(self, row, col):
        if row > 2:
            return False
        if col > 2:
            return False

        if self.board[row][col] != '-':
            return False
        return True

    def place_player(self, player, row, col):
        self.board[row][col] = player

    def take_manual_turn(self, player):
        row = int(input("Enter a row: "))
        col = int(input("Enter a column: "))

        b = self.is_valid_move(row, col)

        while b is False:
            print("Please enter a valid move.")
            row = int(input("Enter a row: "))
            col = int(input("Enter a column: "))

            b = self.is_valid_move(row, col)

        self.place_player(player, row, col)

    def take_turn(self, player):
        print(player + "'s Turn")
        if player == "X":
            self.take_manual_turn(player)
        else:
            self.take_random_turn(player)

    def take_random_turn(self, player):
        r = random.randint(0,2)
        c = random.randint(0,2)
        while not self.is_valid_move(r, c):
            r = random.randint(0, 2)
            c = random.randint(0, 2)

        self.place_player(player, r, c)

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
        if self.check_diag_win(player) is True or self.check_col_win(player) is True or self.check_row_win(
                player) is True:
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

        if self.check_tie():
            print("The game has ended in a tie!")
        else:
            print("Player", player, "has won!")