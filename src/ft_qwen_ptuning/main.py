import logging
import os
import sys

import jieba
import numpy as np
import torch
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge

sys.path.append("../../../CCKS2023-PromptCBLUE")

import transformers.utils.logging
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed, Qwen2Config, AutoModelForCausalLM, \
    AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer

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
    # 日志通过 默认处理器（如控制台输出） 传递
    transformers.utils.logging.enable_default_handler()
    # 为 Transformers 库的日志 定义明确的格式 时间戳 日志级别（INFO/WARNING 等） 模块名称 日志内容
    transformers.utils.logging.enable_explicit_format()


def load_datasets(data_args: DataTrainingArguments, model_args: ModelArguments):
    """
    加载和预处理数据集
    返回标准化的 DatasetDict 对象, 后续可直接用于模型训练
    :param data_args:
    :param model_args:
    :return:
    """
    data_files = {}
    if data_args.train_file is not None:
        data_files['train'] = data_args.train_file
    if data_args.validation_file is not None:
        data_files['validation'] = data_args.validation_file
    if data_args.test_file is not None:
        data_files['test'] = data_args.test_file

    raw_datasets = load_dataset("json",
                                data_files=data_files,
                                cache_dir=model_args.cache_dir)
    return raw_datasets


def load_model_config(model_args):
    config = Qwen2Config.from_pretrained(
        model_args.model_name_or_path
    )

    # prefix_tuning的参数设置
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    return config


def load_tokenizer(model_args: ModelArguments):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )
    return tokenizer


def load_model(model_args: ModelArguments, config):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    return model


def load_tuning_config(model_args: ModelArguments, model):
    if model_args.ptuning_checkpoint is not None:
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    if model_args.quantization_bit is not None:
        logger.info(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        model = model.float()

    return model


def load_data_collator(model, tokenizer, data_args):
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    logger.info(
        f"tokenizer pad_token_id: {label_pad_token_id}, tokenizer bos_token: {tokenizer.bos_token}, tokenizer eos_token: {tokenizer.eos_token}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True
    )
    return data_collator


def compute_metrics(eval_predictions, data_args, tokenizer):
    prediction, labels = eval_predictions
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    decoded_predictions = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_predictions, decoded_labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()

        hypothesis = ' '.join(hypothesis)
        if not hypothesis:
            hypothesis = "-"
        scores = rouge.get_scores(hypothesis, ' '.join(reference))
        result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


def get_column_names(training_args, raw_datasets):
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    return column_names


def preprocess_function_train(examples, data_args: DataTrainingArguments, tokenizer):
    # 总序列最大长度 = 输入长度 + 输出长度
    max_seq_length = data_args.max_source_length + data_args.max_target_length
    input_ids, labels, attention_masks = [], [], []

    # 前缀
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # 指定原始文本列
    prompt_column = data_args.prompt_column
    # 指定预期的响应列
    response_column = data_args.response_column
    # 指定上文历史列
    history_column = data_args.history_column
    logger.info(f"原始文本列为:{prompt_column}, 预期响应列为:{response_column}, 上文历史列为:{history_column}")
    # 最长输出序列长度
    max_target_length = data_args.max_target_length
    max_source_length = data_args.max_source_length

    logger.info(f"前两个examples[prompt_column] 为 {examples[prompt_column][0:2]}")

    for i in range(len(examples[prompt_column])):
        # 一组数据的input 和 labels 都不为空
        if examples[prompt_column][i] and examples[response_column][i]:
            query, answer = examples[prompt_column][i], examples[response_column][i]
            # logger.info(f"query is : {query}, answer: {answer}")

            if history_column is None:
                prompt = query
            else:
                prompt = ""
                history = examples[history_column][i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += f"[Round {turn_idx}]\n问：{old_query}\n答：{response}\n"
                prompt += f"[Round {len(history)}]\n问：{query}\n答："
            # 前缀提示词+输入
            prompt = prefix + prompt

            # lora_A 和 lora_B 两个低秩矩阵
            a_ids = tokenizer.encode(prompt, add_special_tokens=False)
            b_ids = tokenizer.encode(answer, add_special_tokens=False)

            # 输入序列后面添加一个eos，那么原本的max_source_length需要留出一个位置给这个eos，所以截断时用max_source_length -1
            if len(a_ids) > max_source_length - 1:
                a_ids = a_ids[:max_source_length - 1]
            # logger.info(f"a_ids is {a_ids}")

            # 输出可能在前后都添加标记，比如前面加bos，后面加eos，这样就需要预留两个位置，所以截断时用max_target_length -2
            if len(b_ids) > max_target_length - 1:
                b_ids = b_ids[:max_target_length - 1]
            # logger.info(f"b_ids is {b_ids}")

            # 模型的input_ids
            input_id = a_ids + b_ids + [tokenizer.eos_token_id]
            # logger.info(f"input_id is {input_id}")
            # 输入结束时的文本长度
            query_length = len(a_ids)
            # 模型的labels 忽略输入，输入不参与梯度更新
            label = [-100] * query_length + input_id[query_length:]

            # 填充长度
            pad_len = max_seq_length - len(input_id)

            # 对input填充
            input_ids.append(input_id + [tokenizer.pad_token_id] * pad_len)
            # 对labels填充
            labels.append(label + [-100] * pad_len)
            # 填充attention_mask
            attention_masks.append([1] * len(input_id) + [0] * pad_len)
    # logger.info(f"input_ids: {input_ids}")
    return {
        "input_ids": torch.LongTensor(input_ids),
        "attention_mask": torch.LongTensor(attention_masks),
        "labels": torch.LongTensor(labels),
    }


def preprocess_function_eval(examples, tokenizer, data_args):
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    max_target_length = data_args.max_target_length
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    inputs, targets = [], []
    for i in range(len(examples[prompt_column])):
        if not examples[response_column][i]:
            targets.append("filled in !")
        else:
            targets.append(examples[response_column][i])

        if examples[prompt_column][i]:
            query = examples[prompt_column][i]
            if history_column is None or len(examples[history_column][i]) == 0:
                prompt = query
            else:
                prompt = ""
                history = examples[history_column][i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
            inputs.append(prompt)

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs,
                             max_length=data_args.max_source_length,
                             truncation=True,
                             padding="max_length")
    labels = tokenizer(targets,
                       max_length=max_target_length,
                       truncation=True,
                       padding="max_length")

    if data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def load_preprocessed_datas(training_args, tokenizer, data_args, raw_datasets):
    train_dataset, eval_dataset, predict_dataset = None, None, None
    column_names = get_column_names(training_args, raw_datasets)
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
                fn_kwargs={  # 传递额外参数
                    "data_args": data_args,
                    "tokenizer": tokenizer
                }
            )
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length

        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
                fn_kwargs={  # 传递额外参数
                    "data_args": data_args,
                    "tokenizer": tokenizer
                }
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]

        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

        # print("predict_dataset: ", predict_dataset)
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
                fn_kwargs={  # 传递额外参数
                    "data_args": data_args,
                    "tokenizer": tokenizer
                }
            )
    return train_dataset, eval_dataset, predict_dataset


def load_trainer(training_args,
                 data_args,
                 model,
                 tokenizer,
                 data_collator,
                 raw_datasets):
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    train_dataset, eval_dataset, predict_dataset = load_preprocessed_datas(training_args, tokenizer, data_args,
                                                                           raw_datasets)
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        # save_prefixencoder= model_args.pre_seq_len if model_args.pre_seq_len is not None else False
    )
    return trainer, train_dataset, eval_dataset, predict_dataset


def train(training_args, data_args, trainer, train_dataset, model):
    if training_args.do_train:
        checkpoint = None
        # 模型检查保存点
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # 开始训练
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(
            train_dataset)

        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        return trainer


def eval(training_args, data_args, eval_dataset, trainer):
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval",
                                   do_sample=True,
                                   top_p=0.7,
                                   max_length=512,
                                   temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def predict(training_args, data_args, predict_dataset, tokenizer, trainer):
    logger.info("*** Predict ***")

    # 读取原 test file（添加错误处理）
    list_test_samples = []
    try:
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    logger.warning(f"跳过空行：第 {line_num} 行")
                    continue
                try:
                    data = json.loads(line)
                    list_test_samples.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"测试文件第 {line_num} 行,line为:{line} JSON 解析失败: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"测试文件不存在: {data_args.test_file}")

    # 执行预测
    predict_results = trainer.predict(
        predict_dataset,
        metric_key_prefix="predict",
        max_new_tokens=512,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    metrics = predict_results.metrics
    logger.info(f"最后执行预测的指标为: {metrics}")
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    is_world_process_zero = trainer.is_world_process_zero()
    logger.info(f"is_world_process_zero is {is_world_process_zero}")
    if is_world_process_zero:
        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            labels = tokenizer.batch_decode(
                predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            labels = [label.strip() for label in labels]
            logger.info(f"预测:{predictions}和期望:{labels}")
            assert len(labels) == len(list_test_samples)

            logger.info(f"预测指标输出的文件夹路径为:{training_args.output_dir}")
            output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")
            logger.info(f"预测指标输出的文件路径为:{output_prediction_file}")

            # 写入预测结果（添加序列化检查）
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for idx, (p, l) in enumerate(zip(predictions, labels)):
                    if idx >= len(list_test_samples):
                        raise IndexError("预测结果数量超过测试样本数")
                    samp = list_test_samples[idx]
                    samp["target"] = p
                    try:
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")
                    except TypeError as e:
                        raise ValueError(f"无法序列化第 {idx} 个样本: {str(e)}")


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

    # 设置固定种子保证实验的可重复性
    set_seed(training_args.seed)

    # Load dataset
    raw_datasets = load_datasets(data_args, model_args)

    # 加载配置 包括p_tuning 的配置
    config = load_model_config(model_args)

    # 加载分词器
    tokenizer = load_tokenizer(model_args)

    # 加载模型
    model = load_model(model_args, config)

    # p_tuning_config
    model = load_tuning_config(model_args, model)

    # DataCollator 批处理数据
    data_collator = load_data_collator(model, tokenizer, data_args)

    # 加载trainer
    trainer, train_dataset, eval_dataset, predict_dataset = load_trainer(training_args, data_args, model, tokenizer,
                                                                         data_collator, raw_datasets)
    # 模型训练
    trainer = train(training_args, data_args, trainer, train_dataset, model)

    # trainer = create_mock_trainer()

    # 模型评估
    eval(training_args, data_args, eval_dataset, trainer)

    # 模型预测
    predict(training_args, data_args, predict_dataset, tokenizer, trainer)


if __name__ == "__main__":
    main()
