games = ["Minecraft", "Geometry Dash", "Pokemon GO", "Clash Royale"]

for game in games:
    print("I like playing" , game)

new_game = input("What is your favorite game? ")
games.append(new_game)

for game in games:
    print("We like playing", game)