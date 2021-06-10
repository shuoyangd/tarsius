import sys

wordlist = [
  "aunt",
  "boy",
  "bride",
  "brother",
  "daughter",
  "father",
  "girl",
  "granddaughter",
  "grandfather",
  "grandmother",
  "grandson",
  "groom",
  "husband",
  "lady",
  "lord",
  "man",
  "monk",
  "mother",
  "nun",
  "sister",
  "son",
  "uncle",
  "wife",
  "woman"
]

for line in sys.stdin:
  toks = line.strip().split()
  split_points = []
  for word in wordlist:
    if word in toks:
      idx = toks.index(word)
      toks[idx] = ','
  sys.stdout.write(" ".join(toks) + '\n')
  
