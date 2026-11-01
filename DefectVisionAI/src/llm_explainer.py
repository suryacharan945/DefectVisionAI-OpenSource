
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

def load_explainer_pipeline(device, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=local_files_only)

    if device.type == "cuda":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quant_config, device_map="auto",
            low_cpu_mem_usage=True, local_files_only=local_files_only
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True,
            local_files_only=local_files_only
        )
        model = model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe, tokenizer


def build_explanation_prompt(category, defect_type, severity_level, severity_score,
                              defect_area_pct, confidence_pct):
    system_msg = (
        "You are an industrial quality inspection assistant. Given structured "
        "defect detection results, write a concise 2-3 sentence explanation for "
        "a factory operator. Only use the facts provided. Do not invent causes, "
        "dates, machine names, or numbers not given to you. If the cause is not "
        "certain from the data, say the likely cause is uncertain and describe "
        "the defect pattern instead. Stop after the explanation - do not write "
        "anything else, including unrelated questions or examples."
    )
    user_msg = (
        f"Product category: {category}\n"
        f"Detected defect type: {defect_type}\n"
        f"Severity level: {severity_level} (score: {severity_score}/100)\n"
        f"Estimated defect area: {defect_area_pct}% of surface\n"
        f"Model confidence this is defective: {confidence_pct}%\n\n"
        f"Write a short explanation of this inspection result for the operator."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def generate_explanation(pipe, tokenizer, messages, max_new_tokens=120):
    # Phi-3 signals end-of-turn with <|end|>, not just the standard EOS token.
    # Without listing both, generation runs past the intended stopping point.
    terminators = [tokenizer.eos_token_id]
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if end_id is not None and end_id != tokenizer.unk_token_id:
        terminators.append(end_id)

    output = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False,
    )
    text = output[0]["generated_text"]
    if isinstance(text, list):
        text = text[-1]["content"]
    return text.strip()
