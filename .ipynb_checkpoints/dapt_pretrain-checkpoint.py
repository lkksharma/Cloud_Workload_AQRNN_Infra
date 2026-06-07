import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

def main():
    parser = argparse.ArgumentParser(description="Run Domain-Adaptive Pretraining (DAPT) on XLM-R.")
    parser.add_argument("--lang", type=str, required=True, choices=["ar", "zh"], help="Target language (e.g., 'ar' or 'zh')")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=100000, help="Number of sentences to use")
    args = parser.parse_args()

    lang = args.lang
    model_name = "xlm-roberta-base"
    output_dir = f"./xlm-roberta-dapt-{lang}"

    print(f"\n{'='*50}")
    print(f"--- STARTING DAPT PIPELINE FOR {lang.upper()} ---")
    print(f"{'='*50}")
    
    print(f"[1/4] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    print(f"[2/4] Downloading raw {lang.upper()} Wikipedia data...")
    # Loading the standard Wikipedia dataset for the specific language
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")

    print(f"      Sampling {args.samples} sentences for feasible compute time...")
    dataset = dataset.shuffle(seed=42).select(range(min(args.samples, len(dataset))))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("[3/4] Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["id", "url", "title", "text"]
    )

    # The magic happens here: This randomly masks 15% of the tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32, # Drop to 16 if you hit an Out-Of-Memory (OOM) error
        save_steps=10_000,
        save_total_limit=1,
        logging_steps=500,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(), # Enables mixed precision for faster GPU training
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )

    print(f"\n[4/4] Starting Masked Language Modeling (MLM)...")
    trainer.train()

    print(f"\nSaving new DAPT backbone to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("--- DAPT COMPLETE ---")

if __name__ == "__main__":
    main()