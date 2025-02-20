from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/attention layer we are going to use.
    预训练模型所需要的参数
    """
    model_name_or_path: Optional[str] = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co 本地模型路径或huggingface的模型"},
    )

    ptuning_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "ath to p-tuning v2 checkpoints 检查点的保存路径"}
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name 配置"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name 分词器"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from 希望下载模型后保存的路径"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. 是否使用rust实现的快速分词器"}
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id). 指定特定模型版本"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models). 如果要加载 Hugging Face 上的私有模型或数据集，需要提供 API 令牌"},
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds the model's position embeddings. 调整位置嵌入层的大小"}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "指定用于量化的位数, 决定了量化后模型的精度"}
    )
    pre_seq_len: Optional[int] = field(
        default=None,
        metadata={"help": "输入序列前添加一个可学习的前缀序列的长度"}
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "是否用投影层,将输入的前缀向量映射到与模型输入一致的维度，确保它们可以与主模型的输入有效地结合"}
    )
    trainable: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "指定哪些模块是可训练的"}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "lora微调的矩阵的秩"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "训练过程中随机“丢弃”一些神经元，减少模型对训练数据的过度拟合"}
    )
    lora_alpha: Optional[float] = field(
        default=32.,
        metadata={
            "help": "缩放因子 增大使得低秩矩阵对模型的输出影响更大使得微调效果更显著,反之在训练数据有限的情况下有助于避免过拟合"}
    )
    modules_to_save: Optional[str] = field(
        default='embed_tokens,lm_head',
        metadata={"help": "模型保存时需要保留的模块或子模块的名称"}
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "是否启用调试模式"}
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "指定加载预训练或微调模型的位置"}
    )


@dataclass
class DataTrainingArguments:
    lang: Optional[str] = field(
        default=None,
        metadata={"help": "Language id for summarization. 指定处理文本时所使用的语言"}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library). 数据集路径"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library). 指定数据集的配置名称，通常用于指定不同版本、子任务或数据拆分"}
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization). 从哪个列中提取原始文本进行摘要"},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization). 指定数据集中的哪一列包含期望的摘要文本"},
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the history of chat. 指定数据集中包含聊天历史记录的列名"},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file). 训练集路径"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file). 验证集路径"},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file). 测试集路径"},
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets 是否覆盖缓存的训练和评估数据集"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
                "最大输出长度 超出则截断 不足则填充"
            )
        },
    )

    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
                "最大输出长度 超出截断 不足填充"
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
                "表示在验证阶段，目标文本序列（例如摘要、翻译结果等）的最大长度"
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
                "是否将所有输入样本填充到模型的最大句子长度"
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
                "束搜索是一种启发式搜索算法，通常用于生成任务中，它通过在每一步生成多个候选并选择最优的路径来生成更加准确的输出"
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not. 控制在计算损失时是否忽略填充（padding）标记"
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={"help":
            (
                "A prefix to add before every source text (useful for T5 models)."
                "用于在每个源文本前添加一个前缀。这对于某些特定模型（如 T5 模型）特别有用，因为 T5 模型通常使用不同的任务前缀来指示模型进行不同的任务"
                "输入文本可以是“翻译英语到法语：Hello, how are you?”，前缀“翻译英语到法语”告诉模型它需要执行翻译任务。这种方法使得模型能够处理不同的任务，而不需要为每个任务训练一个单独的模型"
            )
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
                "在生成序列时，尤其是对于多语言模型（如 mBART），需要在生成的文本开始时强制指定一个特定的令牌。例如，在翻译任务中，可能需要确保模型生成的文本以目标语言的标记开头，以便模型知道要生成哪种语言的文本。"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            # 如果训练集不为空
            if self.train_file is not None:
                # 获取文件名后缀
                extension = self.train_file.split(".")[-1]
                # 检查后缀是否为csv或者json文件
                assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
                # 如果验证集不为空
            if self.validation_file is not None:
                # 获取文件名后缀
                extension = self.validation_file.split(".")[-1]
                # 检查后缀是否为csv或者json文件
                assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
