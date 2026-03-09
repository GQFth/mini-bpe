# Mini BPE Tokenizer

> 🎯 **Implementing BPE (Byte Pair Encoding) from Scratch to Understand LLM Tokenization**

## 📖 Project Overview

This is a **pure Python implementation** of a BPE tokenizer, designed to learn and understand the core mechanisms of text tokenization in Large Language Models (LLMs).

Through this project, you will see:
- How text is split into tokens
- What the training process actually "learns"
- Why `vocab_size` is the key parameter for a tokenizer

> 💡 **This is NOT a production-ready tool.** It is an **educational implementation**. For production scenarios, please use [Hugging Face tokenizers](https://github.com/huggingface/tokenizers).

---

## 🎯 Motivation

When learning about LLMs, I found that the Tokenizer was often a "black box":
- You call `tokenizer.encode("hello")` and get `[15496]`
- But **why** this number?
- What exactly does the **training process** do?
- What is the **saved model** actually storing?

To answer these questions, I decided to **hand-write a BPE tokenizer from scratch**, turning the black box into a white box.

---

## 🔍 Core Principles

### What is BPE?

The core idea of BPE (Byte Pair Encoding) is simple:

> **Count the most frequent pair of characters, merge them into a new token, and repeat this process until the vocabulary is full.**

### Training Process Example

Assume the initial vocabulary contains only characters: `['a', 'b', 'c', 'd', 'e']`

**Corpus:** `"aaabdaaabc"`

1.  **Count Frequencies:** `('a', 'a')` appears 3 times, `('a', 'b')` appears 2 times...
2.  **Merge Most Frequent:** `('a', 'a')` → Create new token `'aa'`
3.  **Update Corpus:** `"aaabdaaabc"` → `"aabdaaabc"` (the first two `aa` are merged)
4.  **Repeat:** Continue counting and merging...

**Training Result:**
- `vocab.json`: Mapping from characters/tokens to IDs
- `merges.txt`: Merge rules (the "character relationships" learned during training)

### Key Code Snippet

```python
# Training loop: Stop when the vocabulary is full
while len(self.merges) < self.vocab_size - len(self.vocab):
    # 1. Count frequencies of all adjacent character pairs
    pairs = get_stats(self.corpus)
    
    # 2. Find the pair with the highest frequency
    best_pair = max(pairs, key=pairs.get)
    
    # 3. Merge this pair
    self.corpus = merge_pairs(self.corpus, best_pair)
    
    # 4. Record the merge rule
    self.merges.append(best_pair)
```

**Key Understanding:**
- `vocab_size` is a **budget limit**, not a target.
- Each merge consumes 1 unit of budget.
- When the budget is spent, **stop immediately**. It does not merge infinitely.

---

## 📦 Project Structure

```
mini-bpe/
├── README.md           # This file
├── bpe.py              # Core BPE implementation
├── train.py            # Training script
├── encode.py           # Encoding/Decoding script
├── example.txt         # Example corpus
└── outputs/            # Training artifacts
    ├── vocab.json      # Vocabulary
    └── merges.txt      # Merge rules
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Pure Python implementation, no extra dependencies required
python --version  # Recommended Python 3.7+
```

### 2. Train the Tokenizer

```bash
python train.py \
    --input data/corpus.txt \
    --vocab_size 1000 \
    --output outputs/
```

**Arguments:**
- `--vocab_size`: Vocabulary size (budget limit), typically 1000-50000
- `--input`: Training corpus file
- `--output`: Model save path

### 3. Use the Tokenizer

```bash
# Encode (Text → Token IDs)
python encode.py \
    --model outputs/ \
    --text "hello world"

# Output: [3421, 1234]

# Decode (Token IDs → Text)
python encode.py \
    --model outputs/ \
    --ids "3421,1234" \
    --decode

# Output: "hello world"
```

### 4. Python API Usage

```python
from bpe import BPETokenizer

# Load trained model
tokenizer = BPETokenizer.from_pretrained("outputs/")

# Encode
tokens = tokenizer.encode("hello world")
print(tokens)  # [3421, 1234]

# Decode
text = tokenizer.decode([3421, 1234])
print(text)  # "hello world"

# Check vocabulary size
print(len(tokenizer.vocab))  # 1000

# Check number of merge rules
print(len(tokenizer.merges))  # 950 (assuming 50 initial characters)
```

---

## 💡 Key Concepts Explained

### 1. What does "Training" actually learn?

It does **NOT** learn neural network weights. It learns **merge rules**:

```
Before Training: ['t', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k']
After Training:  ['the', ' ', 'quick']  # Learned "th"→"the", "qu"→"qui"...
```

**Saved Model** = `vocab.json` + `merges.txt`

### 2. Why is `vocab_size` limited?

Without limitation:
- All words would merge into a single token.
- Cannot handle Out-Of-Vocabulary (OOV) words.
- Vocabulary explodes, running out of memory.

With limitation:
- Only the **most frequent** combinations are kept.
- Low-frequency words are still split.
- Balances vocabulary size and sequence length.

### 3. BPE vs. Neural Network Training

| Feature | BPE Tokenizer | Neural Network (GPT) |
|------|-----------|----------------|
| **Goal** | Learn character combination patterns | Learn language semantics |
| **Output** | Vocabulary + Merge Rules | Weight Matrices |
| **Method** | Statistical Frequency | Backpropagation |
| **Speed** | Fast (Hours) | Slow (Days/Weeks) |
| **Size** | MBs | GBs/TBs |

---

## 🔬 Experiments & Comparison

### Effect of Different `vocab_size`

```bash
# Too Small: Over-segmented
vocab_size=100
"transformer" → ['t', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'e', 'r']

# Moderate: Balanced
vocab_size=1000
"transformer" → ['transform', 'er']

# Very Large: Almost no segmentation
vocab_size=50000
"transformer" → ['transformer']
```

### Comparison with Hugging Face

| Metric | This Implementation | HF tokenizers |
|------|--------|---------------|
| **Language** | Pure Python | Rust + Python |
| **Speed** | Slow (Educational) | Fast (Production) |
| **Dependencies** | None | Requires Compilation |
| **Readability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Performance** | ⭐ | ⭐⭐⭐⭐⭐ |

**Conclusion:**
- **Learn Principles** → Use this implementation
- **Real Projects** → Use HF tokenizers

---

## 🤔 FAQ

### Q: Why not use Hugging Face's tokenizers?

A: The HF library is a **black box**. It is written in Rust underneath, powerful but hard to understand internally. This implementation is a **white box**, with all logic transparent and visible, suitable for learning.

### Q: Can this be used in production?

A: **No.** It is slow, lacks optimizations, and doesn't handle edge cases. Please use HF tokenizers for production environments.

### Q: What is a good `vocab_size`?

A: 
- Small experiments: 1,000 - 5,000
- Medium models: 10,000 - 30,000
- Large models: 30,000 - 100,000

### Q: Must training and inference use the same tokenizer?

A: **Yes!** Otherwise, token IDs won't match, and the model won't work. This is why we save `vocab.json` and `merges.txt`.

---

## 📝 TODO

- [ ] Support special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`)
- [ ] Support Unicode normalization
- [ ] Add unit tests
- [ ] Performance optimization (caching, reducing string copies)
- [ ] Comparative experiment: Merge rule differences across corpora
- [ ] Visualization: Show vocabulary growth process

---

## 📚 Resources

- **Original Paper**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- **Karpathy's minbpe**: https://github.com/karpathy/minbpe
- **Hugging Face tokenizers**: https://github.com/huggingface/tokenizers
- **BPE Visual Demo**: https://youtu.be/zduSFxRajkE?t=194

---

## 🙏 Acknowledgements

- Inspired by Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe) and [nanoGPT](https://github.com/karpathy/nanoGPT)
- Thanks to all authors of open-source BPE implementations

---

## 📄 License

MIT License - Feel free to use and modify, just credit the source 😊

---

**Final Words:**

> Understanding principles is more important than calling APIs.  
> Once you implement it yourself, HF's tokenizers will no longer seem mysterious.

Happy Coding! 🚀
