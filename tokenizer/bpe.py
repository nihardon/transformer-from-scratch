class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        # maps characters to unique integers
        self.string_to_index = {ch : i for i, ch, in enumerate(self.chars)}
        # maps integers to characters
        self.index_to_string = {i : ch for i, ch, in enumerate(self.chars)}

    def encode(self, text):
        return [self.string_to_index[ch] for ch in text]

    def decode(self, tokens):
        return "".join([self.index_to_string[i] for i in tokens])
