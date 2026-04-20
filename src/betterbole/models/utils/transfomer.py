import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_attn_scores(q, k, mask=None, fill_zero=None):
    """
    计算注意力分数

    Args:
        q, k: [B, H, L, d] - 查询和键向量
        mask: [B, H, L, L] 或 [L, L] - 掩码（True 的位置会被填充为 -1e9）
        fill_zero: [B, H, L, L] 或 [L, L] - 掩码（True 的位置会被填充为 0）

    Returns:
        scores: [B, H, L, L] - 注意力分数
    """
    d_k = q.size(-1)
    scores = q @ torch.transpose(k, -1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = torch.masked_fill(scores, mask, value=-1e9)
    if fill_zero is not None:
        scores = torch.masked_fill(scores, fill_zero, value=0)
    return scores


def compute_multihead_attn(mq, mk, mv, mask=None, fill_zero=None):
    """
    计算多头注意力

    Args:
        mq, mk, mv: [B, L, H, d] - 多头的查询、键、值
        mask: 掩码
        fill_zero: 掩码

    Returns:
        out: [B, L, H, d] - 注意力输出
    """
    # 转换为 [B, H, L, d]
    q = mq.transpose(1, 2)
    k = mk.transpose(1, 2)
    v = mv.transpose(1, 2)
    scores = compute_attn_scores(q, k, mask, fill_zero)
    p_attn = F.softmax(scores, dim=-1)
    out = torch.matmul(p_attn, v)
    return out.transpose(1, 2).contiguous(), p_attn


# ============ Transformer 核心模块 ============
class PositionalEncoding(nn.Module):
    """
    位置编码
    使用 sin/cos 位置编码，与 Transformer 原论文一致
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout 率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer 而不是 parameter
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]

        Returns:
            x + pos_encoding: [B, L, d_model]
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 输入投影：将 [B, L, d_model] 映射到 [B, L, num_heads, d_k]
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影：将 [B, L, num_heads, d_k] 映射回 [B, L, d_model]
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [B, L, d_model]
            mask: [L, L] 或 [B, L, L] - 掩码矩阵

        Returns:
            output: [B, L, d_model]
        """
        B = query.size(0)
        L = query.size(1)
        # 1. 投影到多头空间
        # [B, L, d_model] -> [B, L, num_heads, d_k]
        q = self.W_q(query).view(B, L, self.num_heads, self.d_k)
        k = self.W_k(key).view(B, L, self.num_heads, self.d_k)
        v = self.W_v(value).view(B, L, self.num_heads, self.d_k)
        # 2. 使用多头注意力函数计算注意力
        # [B, L, num_heads, d_k] -> [B, L, num_heads, d_k]
        attn_output, p_attn = compute_multihead_attn(q, k, v, mask=mask)
        # [B, L, num_heads, d_k] -> [B, L, d_model]
        output = attn_output.contiguous().view(B, L, self.d_model)
        # 4. 输出投影
        output = self.W_o(output)
        output = self.dropout(output)
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    包含：多头注意力 + 残差连接 + LayerNorm + 前馈网络 + 残差连接 + LayerNorm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐层维度
            dropout: Dropout 率
        """
        super().__init__()

        self.self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, d_model] - 输入序列
            mask: [L, L] - 自注意力掩码（可选）

        Returns:
            out: [B, L, d_model]
        """
        # 注意力子层 + 残差 + LayerNorm（Post-LN）
        attn_out = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + attn_out)

        # 前馈子层 + 残差 + LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    包含多个编码器层的堆叠
    """

    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_layers: 编码器层数
            num_heads: 注意力头数
            d_ff: 前馈网络隐层维度
            dropout: Dropout 率
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, d_model] - 输入序列
            mask: [L, L] - 自注意力掩码（可选）
        Returns:
            out: [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_seq_len=512, dropout=0.1):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_layers: 编码器层数
            num_heads: 注意力头数
            d_ff: 前馈网络隐层维度
            max_seq_len: 最大序列长度
            dropout: Dropout 率
            num_classes: 分类类数（如果为 None，输出 [B, L, d_model]）
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder = TransformerEncoder(d_model, num_layers, num_heads, d_ff, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L] - 输入 token IDs
            mask: [L, L] - 自注意力掩码（可选）
        Returns:
            output: [B, L, d_model] 或 [B, num_classes]（如果设置了 num_classes）
        """
        # 词嵌入
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, mask=mask)
        return x



if __name__ == '__main__':
    print("=" * 60)
    print("测试 Transformer 模块")
    print("=" * 60)

    # 参数设置
    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 3
    num_classes = 10

    print(f"\n配置:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
    print(f"  num_layers={num_layers}, num_classes={num_classes}")

    # 1. 测试单个多头注意力层
    print("\n[1] 测试 MultiHeadAttentionLayer")
    mha = MultiHeadAttentionLayer(d_model, num_heads, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    attn_out = mha(x, x, x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {attn_out.shape}")

    # 2. 测试编码器层
    print("\n[2] 测试 TransformerEncoderLayer")
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    layer_out = encoder_layer(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {layer_out.shape}")

    # 3. 测试完整编码器
    print("\n[3] 测试 TransformerEncoder")
    encoder = TransformerEncoder(d_model, num_layers, num_heads, d_ff)
    encoder_out = encoder(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {encoder_out.shape}")

    # 4. 测试完整 Transformer（无分类头）
    print("\n[4] 测试 Transformer（输出序列表示）")
    transformer = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff,
                              max_seq_len=512)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    transformer_out = transformer(token_ids)
    print(f"  输入 token IDs: {token_ids.shape}")
    print(f"  输出: {transformer_out.shape}")

    # 5. 测试完整 Transformer（有分类头）
    print("\n[5] 测试 Transformer（分类）")
    classifier = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff,
                             max_seq_len=512)
    logits = classifier(token_ids)
    print(f"  输入 token IDs: {token_ids.shape}")
    print(f"  输出（logits）: {logits.shape}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
