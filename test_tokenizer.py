from tokenizer.bpe import CharTokenizer

with open('data/input.txt', 'r') as f:
    text = f.read()

tokenizer = CharTokenizer(text)

# Test 1: basic encode/decode roundtrip
sample = "Hello, World!"
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)
assert decoded == sample, f"Roundtrip failed: got {decoded}"
print(f"Test 1 passed: roundtrip works")

# Test 2: vocab size is reasonable
assert 60 < tokenizer.vocab_size < 100, f"Unexpected vocab size: {tokenizer.vocab_size}"
print(f"Test 2 passed: vocab size is {tokenizer.vocab_size}")

# Test 3: every token is a valid integer
assert all(isinstance(t, int) for t in encoded)
print(f"Test 3 passed: tokens are ints")

print(f"\nSample encoding: '{sample}' -> {encoded}")