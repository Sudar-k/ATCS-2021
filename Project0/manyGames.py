games = ["Minecraft", "Geometry Dash", "Pokemon GO", "Clash Royale"]

for game in games:
    print("I like playing" , game)

still_adding = True

while still_adding:

    new_game = input("What is a game you like to play? (type x to stop adding games) ")
    if new_game == 'x':
        break
    games.append(new_game)


for game in games:
    print("We like playing", game)