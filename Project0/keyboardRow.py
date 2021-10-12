words = ["Hello", "Alaska", "Dad", "Peace"]
def find_words(w):
    first, second, third = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'], ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'], ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    ans = []
    for word in w:
        letters = set(list(word.lower()))
        if letters.issubset(first) or letters.issubset(second) or letters.issubset(third):
            ans.append(word)
    return ans
print(find_words(words))