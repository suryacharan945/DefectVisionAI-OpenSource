
import re

def extract_numbers(text):
    return set(re.findall(r"\d+\.?\d*", str(text)))

def numbers_are_grounded(answer_text, result_data):
    answer_nums = extract_numbers(answer_text)
    source_nums = extract_numbers(str(result_data))
    source_floats = [float(n) for n in source_nums]

    for num_str in answer_nums:
        if num_str in source_nums:
            continue
        try:
            num_val = float(num_str)
        except ValueError:
            return False
        close_matches = [src for src in source_floats if abs(num_val - src) < 0.5]
        if close_matches:
            if num_val == 0 and not any(src == 0 for src in source_floats):
                return False
            continue
        return False
    return True
