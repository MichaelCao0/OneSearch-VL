"""Answer extraction and fixed evaluation prompts for OneSearch and VideoDR."""

import re


# Report-contains-answer protocol derived from OpenSearch-VL evaluation.
JUDGE_PROMPT = (
    "You are an impartial judge evaluating whether a deep research report contains the correct answer.\n\n"
    "[Question]\n{question}\n\n"
    "[Correct Answer]\n{correct_answer}\n\n"
    "[Deep Research Report]\n{response}\n\n"
    "Task: Determine if the deep research report contains the correct answer anywhere in its content.\n\n"
    "Instructions:\n"
    "1. Read through the entire research report carefully\n"
    "2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)\n"
    "3. Check if the information in the report is consistent with the correct answer\n"
    '4. The answer does NOT need to be in a specific format or labeled as "final answer"\n'
    "5. Provide your reasoning\n"
    '6. Answer with "yes" if the report contains the correct answer, "no" if it doesn\'t or contradicts it\n\n'
    "Output format:\n"
    "correct: [yes/no]\n"
    "reasoning: [your explanation]"
)

# VideoDR-Benchmark/llm_as_judge/src/judge_prompts.py: official evaluator text.
OFFICIAL_SYSTEM_MESSAGE = (
    "You are an expert answer evaluator with strong analytical capabilities. "
    "Your role is to carefully analyze and compare answers using a systematic evaluation framework. "
    "You should apply critical thinking to determine correctness based on semantic equivalence and factual accuracy, "
    "rather than relying solely on exact textual matching. Always provide clear, detailed reasoning for your judgments."
)

def official_judge_prompt(question: str, standard_answer: str, model_answer: str) -> str:
    return (
        "You are an expert answer evaluator. Your task is to determine whether a model's answer is "
        "correct by comparing it with a standard answer.\n\n"
        f"Question: {question}\n\n"
        f"Standard Answer: {standard_answer}\n\n"
        f"Model Answer: {model_answer}\n\n"
        "**Evaluation Framework:**\n\n"
        "1. **Identify the answer type**: Determine whether the answer is numerical, categorical, "
        "descriptive, or a combination. Consider what type of information the question is asking for.\n\n"
        "2. **Extract core information**: Identify the essential information in both answers, "
        "ignoring formatting, stylistic differences, and non-essential details.\n\n"
        "3. **Semantic equivalence assessment**:\n"
        "   - For numerical answers: Check if the values are equivalent considering units, "
        "precision, and representation formats\n"
        "   - For categorical answers: Check if they refer to the same entity or concept, "
        "accounting for synonyms and alternative names\n"
        "   - For descriptive answers: Check if they convey the same meaning and key information\n\n"
        "4. **Completeness check**: Determine if the model answer contains all critical information "
        "required by the question, or if missing information would make the answer incorrect.\n\n"
        "5. **Contextual relevance**: Consider the question's intent and whether the model answer "
        "adequately addresses what was asked.\n\n"
        "**Decision Principles:**\n"
        "- Prioritize semantic correctness over exact textual matching\n"
        "- Allow reasonable variations in format, units, and expression\n"
        "- Be strict about factual accuracy and completeness of essential information\n"
        "- Distinguish between related but distinct entities or concepts\n"
        "- Consider the question's context and requirements when making judgments\n\n"
        "**Output Format:**\n"
        "Provide your judgment in JSON format:\n"
        "{\n"
        '    "is_correct": true/false,\n'
        '    "reasoning": "Your detailed reasoning explaining: (1) the answer type identified, '
        '(2) how you compared the answers, (3) what differences you found (if any), and (4) why '
        'you concluded the answer is correct or incorrect (in English)"\n'
        "}\n"
    )

def extract_boxed_content(text: str) -> str | None:
    """Extract content from \\boxed{...} handling nested braces.

    Accept both ``boxed{...}`` and ``\\boxed{...}``, preferring the latter.
    """
    idx = text.find("\\boxed{")
    if idx != -1:
        start = idx + len("\\boxed{")
    else:
        # Fallback: accept boxed{...} without leading backslash
        m = re.search(r"(?<!\\)boxed\{", text)
        if not m:
            return None
        start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    content = text[start:i-1].strip()
    content = re.sub(r'\\text\{([^}]*)\}', r'\1', content)
    content = content.replace("\\#", "#").replace("\\%", "%")
    return content

def extract_final_answer(text: str) -> str:
    """Prefer boxed content, then answer tags, response tags, and final text."""
    if not text:
        return ""

    boxed = extract_boxed_content(text)
    if boxed:
        return boxed

    answer_tag = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip()

    response_tag = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
    if response_tag:
        return response_tag.group(1).strip()

    parts = text.split("</think>")
    if len(parts) > 1:
        last_part = parts[-1].strip()
        last_part = re.sub(r'<tool_call>.*?</tool_call>', '', last_part, flags=re.DOTALL).strip()
        if last_part:
            return last_part[:2000]

    return text[-2000:] if len(text) > 2000 else text
