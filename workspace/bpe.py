from typing import Union
from uu import decode

# table = {}
# bow = "aaabdaaabac"
#
with open("input.txt", "r", encoding="utf-8") as f:
    bow = f.read()

utf = list(bow.encode("utf-8"))

table: dict[Union[int, tuple[int, int]], int] = {
    ch: i for i, ch in enumerate(sorted(list(set(utf))))
}
tokens = list(map(lambda c: table[c], utf))

vocab = 256


def pairs(tokens: list[int]) -> dict[tuple[int, int], list[int]]:
    pairs: dict[tuple[int, int], list[int]] = {}
    i = 0
    while i < len(tokens) - 1:
        now = tokens[i]
        later = tokens[i + 1]
        pair = (now, later)
        if pair in pairs:
            pairs[pair].append(i)
        else:
            pairs[pair] = [i]
        i += 1
    return pairs


while len(table) < vocab:
    p = pairs(tokens)
    if not p:
        break
    common = max(p.items(), key=lambda e: len(e[1]))
    target, idxs = common
    idxs = set(idxs)
    print(target)
    new_token = len(table)
    table[target] = new_token
    new_tokens = []
    i = 0
    print("replace: ", target, " -> ", new_token)
    while i < len(tokens):
        if i in idxs:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    tokens = new_tokens

print(tokens)
print(table)
