import string

def token(string):
    start = 0
    i = 0
    token_list = []
    for x in range(0, len(string)):
        if " " == string[i:i+1]:
            token_list = token_list + string[start:i+1]
            print(string[start:i+1])
            start = i + 1
        i += 1
    return token_list 