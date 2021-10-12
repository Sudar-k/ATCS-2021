alphabet = "abcdefghijklmnopqrstuvwxyz"
sentence = "A man, a plan, a canal: Panama"
sentence_list = sentence.split(" ")


def valid_palindrome(s):
    char = []
    for word in s:
        for letter in word.lower():
            if letter in alphabet:
                char.append(letter)
    return char == list(reversed(char))


print(valid_palindrome(sentence))
