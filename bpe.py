from collections import defaultdict
from typing import List, Dict, Tuple
import json
import re

class BPE:
    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.merges = []  # 合并规则列表: [(a, b), ...]
        self.vocab = {}   # token -> id
        self.id_to_token = {}  # id -> token
        
    def _get_stats(self, corpus: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """统计所有相邻字符对的频率"""
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs
    
    def _merge_pair(self, corpus: Dict[Tuple[str, ...], int], pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
        """将指定 pair 合并为新 token"""
        new_corpus = {}
        bigram = pair
        new_token = bigram[0] + bigram[1]
        
        for word, freq in corpus.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i+1] == bigram[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus[tuple(new_word)] = freq
        return new_corpus
    
    def train(self, text: str):
        """训练 BPE 模型"""
        # 1. 预处理：分词 + 添加 </w>
        words = text.split()
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # 2. 将单词转为字符元组，并添加 </w>
        corpus = {}
        for word, freq in word_freq.items():
            chars = tuple(list(word[:-1]) + [word[-1] + '</w>'])
            corpus[chars] = freq
        
        # 3. 初始化词表（所有字符）
        all_chars = set()
        for word in corpus.keys():
            all_chars.update(word)
        for char in sorted(all_chars):
            self.vocab[char] = len(self.vocab)
        
        # 4. 迭代合并
        while len(self.merges) < self.vocab_size - len(self.vocab):
            pairs = self._get_stats(corpus)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            corpus = self._merge_pair(corpus, best_pair)
        
        # 5. 构建 id_to_token
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为 token id 列表"""
        tokens = []
        words = text.split()
        
        for word in words:
            # 添加 </w>
            word_tokens = list(word[:-1]) + [word[-1] + '</w>']
            
            # 应用所有合并规则
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i+1] == pair[1]:
                        word_tokens[i] = pair[0] + pair[1]
                        word_tokens.pop(i+1)
                    else:
                        i += 1
            
            # 转为 id
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # 未知 token 拆分为字符
                    for char in token:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """将 token id 列表解码为文本"""
        tokens = [self.id_to_token.get(id, f'<unk:{id}>') for id in token_ids]
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save(self, path: str):
        """保存模型"""
        data = {
            'vocab_size': self.vocab_size,
            'merges': self.merges,
            'vocab': self.vocab
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.merges = [tuple(m) for m in data['merges']]
        self.vocab = data['vocab']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def show_merges(self, n: int = 10):
        """显示前 n 个合并规则"""
        print(f"前 {n} 个合并规则:")
        for i, (a, b) in enumerate(self.merges[:n]):
            print(f"  {i+1}. '{a}' + '{b}' -> '{a}{b}'")



if __name__ == '__main__':
    # 1. 准备训练语料
    corpus = """
    low low low low low
    lower lower
    newest newest newest newest newest newest
    widest widest widest
    """

    # 2. 创建并训练 BPE
    bpe = BPE(vocab_size=50)    #
    bpe.train(corpus)

    # 3. 查看词表大小和合并规则
    print(f"词表大小：{bpe.get_vocab_size()}")
    bpe.show_merges(10)

    # 4. 编码示例
    text = "low newer widest"
    token_ids = bpe.encode(text)
    print(f"\n原文：{text}")
    print(f"Token IDs: {token_ids}")

    # 5. 解码示例
    decoded = bpe.decode(token_ids)
    print(f"解码结果：{decoded}")

    # 6. 保存和加载
    bpe.save('bpe_model.json')
    print("\n模型已保存到 bpe_model.json")

    bpe2 = BPE()    # 最后初始化，然后在加载一次
    bpe2.load('bpe_model.json')
    print(f"加载后词表大小：{bpe2.get_vocab_size()}")
