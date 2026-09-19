import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/models/"
    "Qwen2.5-1.5B-Instruct"
)

DEFAULT_TRAIN_PATH = "data/lora/train.jsonl"
DEFAULT_VALIDATION_PATH = "data/lora/validation.jsonl"

DEFAULT_OUTPUT_DIR = (
    "lora_outputs/"
    "qwen2.5-1.5b-router"
)

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name):
    if dtype_name == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"

        return torch.float16, "float16"

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    dtype = mapping[dtype_name]

    if (
        dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError(
            "当前GPU不支持bfloat16，"
            "请改用float16"
        )

    return dtype, dtype_name


def load_jsonl(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{path}"
        )

    samples = []
    seen_ids = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}第{line_number}行"
                    f"不是合法JSON：{error}"
                ) from error

            sample_id = sample.get("id")
            messages = sample.get("messages")

            if not isinstance(sample_id, str):
                raise ValueError(
                    f"{path}第{line_number}行"
                    "缺少合法id"
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"{path}存在重复id："
                    f"{sample_id}"
                )

            seen_ids.add(sample_id)

            if (
                not isinstance(messages, list)
                or len(messages) != 3
            ):
                raise ValueError(
                    f"样本{sample_id}必须包含"
                    "system、user、assistant"
                    "三条消息"
                )

            roles = [
                message.get("role")
                for message in messages
            ]

            if roles != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValueError(
                    f"样本{sample_id}角色顺序"
                    f"错误：{roles}"
                )

            samples.append(sample)

    if not samples:
        raise ValueError(
            f"数据文件为空：{path}"
        )

    return samples


def check_no_id_overlap(
    train_samples,
    validation_samples,
):
    train_ids = {
        sample["id"]
        for sample in train_samples
    }

    validation_ids = {
        sample["id"]
        for sample in validation_samples
    }

    overlap = (
        train_ids & validation_ids
    )

    if overlap:
        raise ValueError(
            "训练集和验证集存在数据泄漏："
            + ", ".join(sorted(overlap))
        )


class RouterDataset(Dataset):
    def __init__(
        self,
        samples,
        tokenizer,
        max_length,
    ):
        self.items = []

        for sample in tqdm(
            samples,
            desc="Tokenizing",
        ):
            item = self._encode_sample(
                sample=sample,
                tokenizer=tokenizer,
                max_length=max_length,
            )

            self.items.append(item)

    @staticmethod
    def _encode_sample(
        sample,
        tokenizer,
        max_length,
    ):
        messages = sample["messages"]

        prompt_messages = messages[:2]

        prompt_ids = (
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

        full_ids = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )

        if len(full_ids) > max_length:
            raise ValueError(
                f"样本{sample['id']}长度"
                f"{len(full_ids)}超过"
                f"max_length={max_length}"
            )

        if (
            full_ids[:len(prompt_ids)]
            != prompt_ids
        ):
            raise ValueError(
                f"样本{sample['id']}的"
                "Prompt不是完整对话的前缀，"
                "无法正确建立assistant标签"
            )

        assistant_token_count = (
            len(full_ids) - len(prompt_ids)
        )

        if assistant_token_count < 1:
            raise ValueError(
                f"样本{sample['id']}没有"
                "可用于训练的assistant Token"
            )

        # system与user部分不计算损失，
        # 只训练assistant输出的路由JSON。
        labels = (
            [-100] * len(prompt_ids)
            + full_ids[len(prompt_ids):]
        )

        return {
            "id": sample["id"],
            "input_ids": full_ids,
            "labels": labels,
            "length": len(full_ids),
            "assistant_tokens": (
                assistant_token_count
            ),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class RouterDataCollator:
    def __init__(
        self,
        pad_token_id,
    ):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_length = max(
            len(item["input_ids"])
            for item in batch
        )

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            item_length = len(
                item["input_ids"]
            )

            padding_length = (
                max_length - item_length
            )

            input_ids.append(
                item["input_ids"]
                + [self.pad_token_id]
                * padding_length
            )

            attention_masks.append(
                [1] * item_length
                + [0] * padding_length
            )

            labels.append(
                item["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


def move_batch_to_device(
    batch,
    device,
):
    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


def autocast_context(dtype):
    enabled = dtype in {
        torch.float16,
        torch.bfloat16,
    }

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=enabled,
    )


def evaluate_validation_loss(
    model,
    data_loader,
    device,
    dtype,
):
    model.eval()

    total_loss = 0.0
    total_target_tokens = 0

    with torch.inference_mode():
        for batch in tqdm(
            data_loader,
            desc="Validation",
            leave=False,
        ):
            batch = move_batch_to_device(
                batch,
                device,
            )

            target_tokens = (
                batch["labels"] != -100
            ).sum().item()

            with autocast_context(dtype):
                outputs = model(**batch)
                loss = outputs.loss

            total_loss += (
                loss.item()
                * target_tokens
            )

            total_target_tokens += (
                target_tokens
            )

    model.train()

    if total_target_tokens == 0:
        raise ValueError(
            "验证集中没有训练目标Token"
        )

    return (
        total_loss
        / total_target_tokens
    )


def save_json(
    data,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")


def save_adapter(
    model,
    tokenizer,
    output_path,
):
    output_path = Path(output_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_path,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        output_path
    )


def get_gpu_memory():
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": round(
            torch.cuda.memory_allocated()
            / 1024**3,
            3,
        ),
        "reserved_gb": round(
            torch.cuda.memory_reserved()
            / 1024**3,
            3,
        ),
        "peak_allocated_gb": round(
            torch.cuda.max_memory_allocated()
            / 1024**3,
            3,
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "使用Qwen进行工具路由LoRA微调"
        )
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--train-data",
        default=DEFAULT_TRAIN_PATH,
    )

    parser.add_argument(
        "--validation-data",
        default=DEFAULT_VALIDATION_PATH,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "auto",
            "float16",
            "bfloat16",
            "float32",
        ],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "限制优化器更新次数，"
            "适合短暂试运行"
        ),
    )

    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
    )

    return parser.parse_args()


def validate_arguments(args):
    if args.max_length < 1:
        raise ValueError(
            "max-length必须大于0"
        )

    if args.epochs < 1:
        raise ValueError(
            "epochs必须大于0"
        )

    if args.batch_size < 1:
        raise ValueError(
            "batch-size必须大于0"
        )

    if (
        args.gradient_accumulation_steps
        < 1
    ):
        raise ValueError(
            "gradient-accumulation-steps"
            "必须大于0"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate必须大于0"
        )

    if not 0 <= args.warmup_ratio < 1:
        raise ValueError(
            "warmup-ratio必须在"
            "[0, 1)之间"
        )

    if args.lora_rank < 1:
        raise ValueError(
            "lora-rank必须大于0"
        )

    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError(
                "max-steps必须大于0"
            )


def main():
    args = parse_arguments()
    validate_arguments(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA不可用，不能开始LoRA训练"
        )

    output_directory = Path(
        args.output_dir
    ).resolve()

    if (
        output_directory.exists()
        and any(output_directory.iterdir())
    ):
        raise FileExistsError(
            "输出目录已经存在且不为空："
            f"{output_directory}\n"
            "请更换--output-dir，避免覆盖实验"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = (
        True
    )

    dtype, dtype_name = resolve_dtype(
        args.dtype
    )

    device = torch.device("cuda:0")

    print("训练配置")
    print(f"模型：{args.model_path}")
    print(f"训练集：{args.train_data}")
    print(
        f"验证集：{args.validation_data}"
    )
    print(f"输出目录：{output_directory}")
    print(f"精度：{dtype_name}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")

    train_samples = load_jsonl(
        args.train_data
    )

    validation_samples = load_jsonl(
        args.validation_data
    )

    check_no_id_overlap(
        train_samples,
        validation_samples,
    )

    print(
        f"训练样本：{len(train_samples)}"
    )
    print(
        f"验证样本："
        f"{len(validation_samples)}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    train_dataset = RouterDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    validation_dataset = RouterDataset(
        samples=validation_samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    data_collator = RouterDataCollator(
        pad_token_id=(
            tokenizer.pad_token_id
        )
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )

    print("正在加载基础模型……")

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            local_files_only=True,
        )
    )

    model.config.use_cache = False

    if not (
        args.disable_gradient_checkpointing
    ):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.to(device)
    model.print_trainable_parameters()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader)
        / args.gradient_accumulation_steps
    )

    planned_steps = (
        updates_per_epoch * args.epochs
    )

    total_training_steps = (
        min(
            planned_steps,
            args.max_steps,
        )
        if args.max_steps is not None
        else planned_steps
    )

    warmup_steps = round(
        total_training_steps
        * args.warmup_ratio
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(
                total_training_steps
            ),
        )
    )

    use_grad_scaler = (
        dtype == torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )

    training_config = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_path": args.model_path,
        "train_data": args.train_data,
        "validation_data": (
            args.validation_data
        ),
        "output_dir": str(
            output_directory
        ),
        "train_samples": len(
            train_samples
        ),
        "validation_samples": len(
            validation_samples
        ),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "effective_batch_size": (
            args.batch_size
            * args.gradient_accumulation_steps
        ),
        "learning_rate": (
            args.learning_rate
        ),
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "planned_steps": planned_steps,
        "total_training_steps": (
            total_training_steps
        ),
        "max_steps": args.max_steps,
        "dtype": dtype_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": TARGET_MODULES,
        "trainable_parameters": (
            trainable_parameter_count
        ),
        "total_parameters": (
            total_parameter_count
        ),
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
    }

    save_json(
        training_config,
        output_directory
        / "training_config.json",
    )

    print(
        f"每轮优化步数："
        f"{updates_per_epoch}"
    )
    print(
        f"总优化步数："
        f"{total_training_steps}"
    )
    print(
        f"Warmup步数：{warmup_steps}"
    )

    history = []
    best_validation_loss = float("inf")
    global_step = 0
    stop_training = False

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.reset_peak_memory_stats()

    training_started_at = (
        time.perf_counter()
    )

    model.train()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_started_at = (
            time.perf_counter()
        )

        epoch_loss_sum = 0.0
        epoch_micro_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        )

        for micro_step, batch in enumerate(
            progress_bar,
            start=1,
        ):
            batch = move_batch_to_device(
                batch,
                device,
            )

            with autocast_context(dtype):
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"训练Loss异常：{loss.item()}"
                )

            epoch_loss_sum += loss.item()
            epoch_micro_batches += 1

            scaled_loss = (
                loss
                / args.gradient_accumulation_steps
            )

            scaler.scale(
                scaled_loss
            ).backward()

            should_update = (
                micro_step
                % args.gradient_accumulation_steps
                == 0
                or micro_step == len(train_loader)
            )

            if not should_update:
                continue

            scaler.unscale_(optimizer)

            gradient_norm = clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            scheduler.step()

            global_step += 1

            current_learning_rate = (
                scheduler.get_last_lr()[0]
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                step=global_step,
                lr=f"{current_learning_rate:.2e}",
            )

            if (
                global_step
                % args.logging_steps
                == 0
            ):
                print(
                    f"\nstep={global_step}，"
                    f"loss={loss.item():.6f}，"
                    f"grad_norm="
                    f"{float(gradient_norm):.4f}，"
                    f"lr="
                    f"{current_learning_rate:.8f}"
                )

            if (
                global_step
                >= total_training_steps
            ):
                stop_training = True
                break

        train_loss = (
            epoch_loss_sum
            / max(epoch_micro_batches, 1)
        )

        validation_loss = (
            evaluate_validation_loss(
                model=model,
                data_loader=(
                    validation_loader
                ),
                device=device,
                dtype=dtype,
            )
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_started_at
        )

        epoch_result = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(
                train_loss,
                6,
            ),
            "validation_loss": round(
                validation_loss,
                6,
            ),
            "epoch_seconds": round(
                epoch_seconds,
                2,
            ),
            "learning_rate": (
                scheduler.get_last_lr()[0]
            ),
            "gpu_memory": (
                get_gpu_memory()
            ),
        }

        history.append(epoch_result)

        print("\n" + "=" * 70)
        print(f"Epoch：{epoch}")
        print(
            f"训练Loss：{train_loss:.6f}"
        )
        print(
            "验证Loss："
            f"{validation_loss:.6f}"
        )
        print(
            f"耗时：{epoch_seconds:.2f}秒"
        )
        print(
            f"显存：{get_gpu_memory()}"
        )

        checkpoint_directory = (
            output_directory
            / f"checkpoint-epoch-{epoch}"
        )

        save_adapter(
            model=model,
            tokenizer=tokenizer,
            output_path=(
                checkpoint_directory
            ),
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            save_adapter(
                model=model,
                tokenizer=tokenizer,
                output_path=(
                    output_directory
                    / "best_adapter"
                ),
            )

            print(
                "已保存新的最佳Adapter，"
                f"validation_loss="
                f"{validation_loss:.6f}"
            )

        save_json(
            {
                "best_validation_loss": (
                    best_validation_loss
                ),
                "global_step": global_step,
                "history": history,
            },
            output_directory
            / "training_history.json",
        )

        if stop_training:
            break

    save_adapter(
        model=model,
        tokenizer=tokenizer,
        output_path=(
            output_directory
            / "last_adapter"
        ),
    )

    total_seconds = (
        time.perf_counter()
        - training_started_at
    )

    final_report = {
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "global_step": global_step,
        "best_validation_loss": (
            best_validation_loss
        ),
        "total_seconds": round(
            total_seconds,
            2,
        ),
        "gpu_memory": get_gpu_memory(),
        "best_adapter": str(
            output_directory
            / "best_adapter"
        ),
        "last_adapter": str(
            output_directory
            / "last_adapter"
        ),
        "history": history,
    }

    save_json(
        final_report,
        output_directory
        / "training_result.json",
    )

    print("\n" + "=" * 70)
    print("LoRA训练完成")
    print(f"总优化步数：{global_step}")
    print(
        "最佳验证Loss："
        f"{best_validation_loss:.6f}"
    )
    print(
        f"总耗时：{total_seconds:.2f}秒"
    )
    print(
        "最佳Adapter："
        f"{output_directory / 'best_adapter'}"
    )
    print(
        "最终Adapter："
        f"{output_directory / 'last_adapter'}"
    )


if __name__ == "__main__":
    main()