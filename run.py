import logging
import os
from pathlib import Path

from transformers import AutoTokenizer, HfArgumentParser, set_seed

from src.utils.arguments import ModelArguments, DataArguments, TrainArguments
from src.utils.trainer import CustomTrainer


def main(json_path):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(json_path))

    if training_args.training_goal == "drug":
        from src.utils.data import DrugDataset as CustomDataset
        from src.utils.data import DrugCollator as CustomCollator
        from src.model.drugModel import Model
    elif training_args.training_goal == "disease":
        from src.utils.data import DiseaseDataset as CustomDataset
        from src.utils.data import DiseaseCollator as CustomCollator
        from src.model.diseaseModel import Model

    if model_args.train_with_qlora:
        training_args.optim = "paged_adamw_32bit"
        training_args.adam_epsilon = 1e-4

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1)
    )

    # Set seed
    set_seed(training_args.seed)

    logging.info("load model...")
    model = Model(model_args)
    model.train()
    tokenizer = model.tokenizer

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    logging.info("load dataset...")
    train_dataset = CustomDataset(args=data_args)

    logging.info("prepare trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CustomCollator(
            tokenizer=tokenizer,
            input_max_len=data_args.input_max_len
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main("./arguments.json")
