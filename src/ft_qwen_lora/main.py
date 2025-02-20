import json
import logging
import os
import sys
from functools import partial

import jieba
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from rouge_chinese import Rouge

sys.path.append("../../../CCKS2023-PromptCBLUE")

from datasets import load_dataset
from src.ft_qwen_lora.arguments import ModelArguments, DataTrainingArguments

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    logging as hf_logging, Qwen2Config, Seq2SeqTrainer, AutoModelForCausalLM,
)

# 创建一个日志记录器，名称为当前模块的名称
logger = logging.getLogger(__name__)


def load_datas(data_args: DataTrainingArguments, model_args: ModelArguments):
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    # load_data加载数据
    raw_datasets = load_dataset(
        path="json",
        data_files=data_files,
        cache_dir=model_args.cache_dir
    )

    logger.info(f"加载数据为：{raw_datasets}")
    return raw_datasets


def load_qwen_model_and_tokenizer(model_args: ModelArguments):
    # 加载模型配置
    config = Qwen2Config.from_pretrained(
        model_args.model_name_or_path
    )
    # 输入序列前添加一个可学习的前缀序列的长度
    config.pre_seq_len = model_args.pre_seq_len
    # 该参数指示是否在输入序列前添加一个投影层
    config.prefix_projection = model_args.prefix_projection
    logger.info(f"加载完成自定义配置之后config is {config}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info(f"加载分词器 tokenizer is{tokenizer}")

    # 半精度加载模型
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config).half().cuda()
    logger.info(f"半精度加载模型 model is{model}")
    return model, tokenizer


def load_lora_model(model, model_args: ModelArguments):
    if model_args.peft_path is not None:
        logger.info("加载Peft模型")
        lora_model = PeftModel.from_pretrained(model, model_args.peft_path)
    else:
        logger.info("初始化peft微调模型")
        # lora加载模型的哪些层
        target_modules = model_args.trainable.split(",")
        # 模型保存时需要保留的模块或子模块的名称
        modules_to_save = model_args.modules_to_save.split(
            ",") if model_args.modules_to_save is not None else target_modules
        # loraAB的秩
        lora_rank = model_args.lora_rank
        # 正则化随机丢弃神经元的概率
        lora_dropout = model_args.lora_dropout
        # 缩放因子
        lora_alpha = model_args.lora_alpha
        # 打印日志 输出lora模型的参数
        logger.info(
            f"lora训练的模块为:{target_modules}, 保留的模块或子模块的名称为:{modules_to_save}, loraAB的秩为:{lora_rank}, 正则化随机丢弃神经元的概率: {lora_dropout}, 缩放因子: {lora_alpha}")
        # 加载lora配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,  # 是否使用推理模式
            r=lora_rank,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save,
        )

        # 加载lora模型
        lora_model = get_peft_model(model, peft_config)
    return lora_model


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
                             padding=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    if data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def print_dataset_example(example, tokenizer):
    print("input_ids: ", example["input_ids"])
    print("inputs: ", tokenizer.decode(example["input_ids"]))
    print("label_ids: ", example["labels"])
    valid_ids = [tid for tid in example["labels"] if tid != -100]
    print("labels: ", tokenizer.decode(valid_ids, skip_special_tokens=True))


def process_raw_data(data_args: DataTrainingArguments, training_args: Seq2SeqTrainingArguments, raw_datasets, model,
                     tokenizer):
    logger.info(f"开始处理数据 从命令行加载的Seq2SeqTrainingArguments参数为:{training_args}")
    if training_args.do_train:
        column_names = raw_datasets['train'].column_names
    elif training_args.do_eval:
        column_names = raw_datasets['validation'].column_names
    elif training_args.do_predict:
        column_names = raw_datasets['test'].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    logger.info(f"json文件中的column_names为:{column_names}")

    # 预处理数据
    inputs = preprocess_function_train(raw_datasets["train"], data_args, tokenizer)

    return inputs


def train_dataset_for_model(raw_datasets, tokenizer, data_args: DataTrainingArguments,
                            training_args: Seq2SeqTrainingArguments):
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")

    # 取出训练数据
    train_dataset = raw_datasets["train"]

    # 最大训练样本数量
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    """
    让主进程先执行某些操作，比如数据预处理，然后再由其他进程处理。
    比如在分布式训练中，可能每个进程都会加载数据集，但预处理步骤如果由所有进程各自执行的话，可能会有重复，浪费资源。所以让主进程先处理，然后将结果共享给其他进程，这样可以节省时间和内存
    """
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
            fn_kwargs={  # 传递额外参数
                "data_args": data_args,
                "tokenizer": tokenizer
            }
        )
    return train_dataset


def valid_dataset_for_model(raw_datasets, tokenizer, data_args: DataTrainingArguments,
                            training_args: Seq2SeqTrainingArguments):
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")

    # 取出训练数据
    eval_dataset = raw_datasets["validation"]

    # 最大训练样本数量
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    """
    让主进程先执行某些操作，比如数据预处理，然后再由其他进程处理。
    比如在分布式训练中，可能每个进程都会加载数据集，但预处理步骤如果由所有进程各自执行的话，可能会有重复，浪费资源。所以让主进程先处理，然后将结果共享给其他进程，这样可以节省时间和内存
    """
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
            fn_kwargs={
                "data_args": data_args,
                "tokenizer": tokenizer
            }
        )
    return eval_dataset


def predict_dataset_for_model(raw_datasets, tokenizer, data_args: DataTrainingArguments,
                              training_args: Seq2SeqTrainingArguments):
    if "test" not in raw_datasets:
        raise ValueError("--do_eval requires a test dataset")

    # 取出训练数据
    predict_dataset = raw_datasets["test"]

    # 最大训练样本数量
    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))

    """
    让主进程先执行某些操作，比如数据预处理，然后再由其他进程处理。
    比如在分布式训练中，可能每个进程都会加载数据集，但预处理步骤如果由所有进程各自执行的话，可能会有重复，浪费资源。所以让主进程先处理，然后将结果共享给其他进程，这样可以节省时间和内存
    """
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
            fn_kwargs={
                "data_args": data_args,
                "tokenizer": tokenizer
            }
        )
    return predict_dataset


def compute_metrics(eval_predicts, tokenizer, data_args):
    predicts, labels = eval_predicts
    if isinstance(predicts, tuple):
        predicts = predicts[0]
    decoded_predicts = tokenizer.batch_decode(predicts, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_predicts, decoded_labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        rouge = Rouge()
        hypothesis = ' '.join(hypothesis)
        if not hypothesis:
            hypothesis = "-"
        scores = rouge.get_scores(hypothesis, ''.join(reference))
        result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

        # blue_score
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


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
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=512,
                                   temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def predict(training_args, data_args, predict_dataset, tokenizer, trainer):
    logger.info("*** Predict ***")

    # 读取原test file
    list_test_samples = []
    with open(data_args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            list_test_samples.append(line)

    predict_results = trainer.predict(
        predict_dataset,
        metric_key_prefix="predict",
        # max_tokens=512,
        max_new_tokens=data_args.max_target_length,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        # top_p=0.7,
        # temperature=0.95,
        # repetition_penalty=1.1
    )
    metrics = predict_results.metrics
    print(metrics)
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            labels = tokenizer.batch_decode(
                predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            labels = [label.strip() for label in labels]
            assert len(labels) == len(list_test_samples)

            output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for idx, (p, l) in enumerate(zip(predictions, labels)):
                    samp = list_test_samples[idx]
                    samp["target"] = p
                    res = json.dumps(samp, ensure_ascii=False)
                    writer.write(f"{res}\n")

    return results


def main():
    # HFArgumentParser 主要处理命令行的参数 ModelArguments为模型相关参数 DataTrainingArguments为数据训练相关参数
    # Seq2SeqTrainingArguments为模型预测等相关参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    print(f"命令行相关参数为:{sys.argv}")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 解析json文件中的参数
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        if model_args.debug_mode:
            print(
                f"解析json文件得到的参数为: model_args:{model_args}, \n data_args:{data_args}, \n training_args:{training_args}")
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        if model_args.debug_mode:
            print(
                f"解析命令行参数得到的参数为: model_args:{model_args}, \n data_args:{data_args}, \n training_args:{training_args}")

    # 配置日志相关参数
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # 打印到控制台
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # should_log为true 表示用户想在训练中打印日志
    if training_args.should_log:
        hf_logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # 设置 transformers 库的日志级别 决定哪些日志信息显示
    hf_logging.set_verbosity(log_level)
    # 启用默认的日志处理器（将日志输出到控制台）
    hf_logging.enable_default_handler()
    # 启用明确的日志格式
    hf_logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 设置固定的种子确保代码的可重复性
    # 不设置种子会导致结果的不可重复性 不利于排查问题
    set_seed(training_args.seed)

    # 加载数据
    raw_datasets = load_datas(data_args, model_args)

    # 加载模型和分词器
    model, tokenizer = load_qwen_model_and_tokenizer(model_args)

    # 加载lora模型
    model = load_lora_model(model, model_args)

    # 打印训练参数比
    model.print_trainable_parameters()

    # 训练&验证&测试 预处理数据
    train_dataset, eval_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        train_dataset = train_dataset_for_model(raw_datasets, tokenizer, data_args, training_args)
    if training_args.do_eval:
        eval_dataset = valid_dataset_for_model(raw_datasets, tokenizer, data_args, training_args)
    if training_args.do_predict:
        predict_dataset = predict_dataset_for_model(raw_datasets, tokenizer, data_args, training_args)

    # 加载数据集 DataCollator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    sample_batch = data_collator([train_dataset[0]])
    logger.info(f"sample_batch keys is {sample_batch.keys()}")

    # 初始化训练器
    training_args.generation_max_length = training_args.generation_max_length if training_args.generation_max_length is not None else data_args.val_max_target_length
    training_args.generation_num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func if training_args.predict_with_generate else None
    )

    # 开始训练
    trainer = train(training_args, data_args, trainer, train_dataset, model)

    # 模型评估
    eval(training_args, data_args, eval_dataset, trainer)

    # 模型预测
    results = predict(training_args, data_args, eval_dataset, tokenizer, trainer)

    logger.info(f"results is {results}")

if __name__ == "__main__":
    main()
