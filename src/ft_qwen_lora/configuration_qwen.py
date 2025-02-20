from transformers import PretrainedConfig, AutoTokenizer, AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenConfig(PretrainedConfig):
    model_type = "qwen_2"

    def __init__(self, vocab_size: int = 151646, hidden_size: int = 4096, num_layers: int = 32,
                 layer_norm_epsilon: float = 1e-5, use_cache: bool = True, bos_token_id: int = 151644,
                 eos_token_id: int = 151645, pad_token_id: int = 151643, max_sequence_length: int = 2048,
                 inner_hidden_size: int = 16384, position_encoding_2d: bool = True, quantization_bits: int = 0,
                 pre_seq_len: int = None, prefix_projection: bool = False, **kwargs):
        """
        自定义qwen配置
        :param vocab_size: 词汇表大小 默认为 151643
        :param hidden_size: Transformer 隐藏层维度（决定模型容量）
        :param num_layers: Transformer 层数（模型深度）7B 模型为 32，14B 模型为 40，需与官方配置对齐
        :param layer_norm_epsilon: Layer Normalization 的 epsilon 值（数值稳定性）
        :param use_cache: 是否使用 KV 缓存加速生成（推理优化）
        :param bos_token_id: 开始符
        :param eos_token_id: 结束符
        :param pad_token_id: 填充符
        :param max_sequence_length: 最大序列长度 包括输入输出
        :param inner_hidden_size: FFN 层中间维度（通常为 4*hidden_size）
        :param position_encoding_2d: 是否使用二维位置编码
        :param quantization_bits: 模型权重量化位数
        :param pre_seq_len: 前缀微调（P-Tuning）中可训练的前缀长度
        :param prefix_projection: 是否对前缀进行投影（增强前缀微调灵活性）
        :param kwargs:
        """

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        self.inner_hidden_size = inner_hidden_size
        self.position_encoding_2d = position_encoding_2d
        self.quantization_bits = quantization_bits
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        super().__init__(pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         **kwargs)

# 注册到 AutoConfig
AutoConfig.register("qwen_2", QwenConfig)
