# # agents/critic.py

# CRITIC_SYSTEM_PROMPT = """
# You are a Critic agent.

# Your ONLY job is to evaluate the answer provided by another agent.

# Evaluate the answer based on:
# 1. Correctness – no factual errors or hallucinations
# 2. Completeness – fully answers the question
# 3. Clarity – well-structured and understandable
# 4. Consistency – no contradictions

# You MUST output your evaluation in EXACTLY this format
# (use the same line breaks and symbols):

# VERDICT: <ACCEPT or REJECT>
# CONFIDENCE: <number between 0 and 1>
# ISSUES:
# - <list each issue on its own line starting with '-', or write '- NONE'>
# RECOMMENDATION: <one short sentence>

# Rules:
# - 'ISSUES:' must be on its own line
# - Always include at least one '-' line under ISSUES
# - Do NOT rewrite the answer
# - Do NOT add extra text
# """


# def parse_critic_output(text: str) -> dict:
#     """
#     Parse critic output into a structured decision.
#     """
#     lines = [l.strip() for l in text.splitlines() if l.strip()]

#     verdict = next(l for l in lines if l.startswith("VERDICT:")) \
#         .split("VERDICT:")[1].strip()

#     confidence = float(
#         next(l for l in lines if l.startswith("CONFIDENCE:"))
#         .split("CONFIDENCE:")[1].strip()
#     )

#     issues_index = lines.index("ISSUES:")

#     issues = []
#     for line in lines[issues_index + 1:]:
#         if line.startswith("RECOMMENDATION:"):
#             break
#         issue = line.lstrip("-").strip()
#         if issue != "NONE":
#             issues.append(issue)

#     recommendation = next(
#         l for l in lines if l.startswith("RECOMMENDATION:")
#     ).split("RECOMMENDATION:")[1].strip()

#     return {
#         "verdict": verdict,
#         "confidence": confidence,
#         "issues": issues,
#         "recommendation": recommendation
#     }


# import json
# import re

# CRITIC_SYSTEM_PROMPT = """You are a Critic that validates answers with confidence scoring.

# Evaluate answers on:
# 1. ACCURACY - Is the information correct?
# 2. COMPLETENESS - Does it fully answer the question?
# 3. CLARITY - Is it well-explained?

# Respond in this JSON format:
# {
#   "verdict": "ACCEPT" or "REJECT",
#   "confidence": 0.85,
#   "accuracy_score": 0.9,
#   "completeness_score": 0.8,
#   "clarity_score": 0.85,
#   "issues": ["list of problems"],
#   "recommendation": "suggestion"
# }

# Confidence guide:
# - 0.80-1.00: Strong answer, accept
# - 0.50-0.79: Needs improvement, retry
# - 0.00-0.49: Poor answer, reject

# Always be specific and constructive."""


# def parse_critic_output(content: str) -> dict:
#     json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
    
#     if json_match:
#         try:
#             data = json.loads(json_match.group())
#             confidence = max(0.0, min(1.0, float(data.get('confidence', 0.0))))
            
#             return {
#                 'verdict': data.get('verdict', 'REJECT').upper(),
#                 'confidence': confidence,
#                 'accuracy_score': float(data.get('accuracy_score', 0.0)),
#                 'completeness_score': float(data.get('completeness_score', 0.0)),
#                 'clarity_score': float(data.get('clarity_score', 0.0)),
#                 'issues': data.get('issues', []),
#                 'recommendation': data.get('recommendation', '')
#             }
#         except (json.JSONDecodeError, ValueError):
#             pass
    
#     # Fallback parsing
#     confidence = 0.5
#     confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', content.lower())
#     if confidence_match:
#         try:
#             confidence = float(confidence_match.group(1))
#             if confidence > 1.0:
#                 confidence = confidence / 100.0
#         except ValueError:
#             pass
    
#     verdict = "REJECT"
#     if any(word in content.upper() for word in ["ACCEPT", "APPROVED", "VALID"]):
#         verdict = "ACCEPT"
    
#     return {
#         'verdict': verdict,
#         'confidence': confidence,
#         'accuracy_score': confidence,
#         'completeness_score': confidence,
#         'clarity_score': confidence,
#         'issues': ['See feedback above'],
#         'recommendation': 'Improve answer quality'
#     }


# def should_retry(confidence: float, retry_count: int) -> bool:
#     if confidence >= 0.80:
#         return False
#     if confidence < 0.50:
#         return False
#     return retry_count == 0



CRITIC_SYSTEM_PROMPT = """
You are a Critic agent.

You MUST evaluate the given answer and respond in EXACTLY this format:

VERDICT: ACCEPT or REJECT
CONFIDENCE: <number between 0.0 and 1.0>
ISSUES:
- <list issues or write NONE>
RECOMMENDATION: <short guidance>

Rules:
- CONFIDENCE is REQUIRED.
- Use:
  - ≥ 0.80 for strong answers
  - 0.50–0.79 for partial answers
  - < 0.50 for weak or incorrect answers
- If ISSUES is NONE, write exactly:
  ISSUES:
  - NONE
- Do not omit any field.
- Do not add extra text.
"""
def parse_critic_output(text: str) -> dict:
    """
    Parse critic output into a structured decision.
    """
    result = {
        "verdict": "REJECT",
        "confidence": 0.0,
        "issues": [],
        "recommendation": "",
    }

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    current_section = None

    for line in lines:
        if line.startswith("VERDICT:"):
            result["verdict"] = line.split(":", 1)[1].strip()

        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                result["confidence"] = 0.0

        elif line.startswith("ISSUES:"):
            current_section = "issues"

        elif line.startswith("RECOMMENDATION:"):
            result["recommendation"] = line.split(":", 1)[1].strip()
            current_section = None

        elif line.startswith("-") and current_section == "issues":
            issue = line[1:].strip()
            if issue.upper() != "NONE":
                result["issues"].append(issue)

    return result

def should_retry(confidence: float, retry_count: int, max_retries: int = 1) -> bool:
    """
    Decide whether a retry should occur based on confidence.
    """
    if retry_count >= max_retries:
        return False

    return 0.50 <= confidence < 0.80
