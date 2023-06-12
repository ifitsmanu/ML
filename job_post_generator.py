
import os

from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          GPT2Tokenizer, TextDataset, Trainer,
                          TrainingArguments)


def fine_tune_gpt(text_file, model_name='gpt2', output_dir='./fine_tuned_job_post_model'):
    """
    Fine-tune GPT-2 on the given text file.

    Args:
        text_file (str): Path to the text file containing the preprocessed job posts.
        model_name (str, optional): The name of the GPT-2 model to be fine-tuned. Defaults to 'gpt2'.
        output_dir (str, optional): The directory where the fine-tuned model will be saved. Defaults to './fine_tuned_job_post_model'.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_file,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def generate_job_post(prompt, model_path='./fine_tuned_job_post_model', max_length=300):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    text_file = 'job_postings.txt'
    fine_tune_gpt(text_file)

    prompt = "CNA job post"
    generated_job_post = generate_job_post(prompt)
    print(generated_job_post)
