s = "   fly me   to   the moon  "
def length_of_last_word(sentence):
    print(len(sentence.strip().split(" ")[-1]))
length_of_last_word(s)
