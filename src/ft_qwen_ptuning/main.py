import logging
import os
import sys
sys.path.append("../../../CCKS2023-PromptCBLUE")


import transformers.utils.logging
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from src.ft_qwen_lora.arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


def get_command_args(argv):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args


def set_logger_config(log_level: int):
    # 控制该logger处理哪些级别的日志信息
    logger.setLevel(log_level)
    # transformers内部会输出log_level及以上级别的日志
    transformers.utils.logging.set_verbosity(log_level)
    #日志通过 默认处理器（如控制台输出） 传递
    transformers.utils.logging.enable_default_handler()
    # 为 Transformers 库的日志 定义明确的格式 时间戳 日志级别（INFO/WARNING 等） 模块名称 日志内容
    transformers.utils.logging.enable_explicit_format()



def main():
    # 解析命令行参数
    model_args, data_args, training_args = get_command_args(sys.argv)

    # 设置日志参数
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    # 日志基础配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # 配置日志
    set_logger_config(training_args.get_process_log_level())

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

if __name__ == "__main__":
    main()