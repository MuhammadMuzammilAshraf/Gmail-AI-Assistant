import logging

from langchain.prompts import PromptTemplate

log = logging.getLogger(__name__)

DRAFT_PROMPT = PromptTemplate.from_template(
    """
You are a professional email assistant.
Write a clear, polite, concise reply to the email below.
Rules: under 150 words, professional but friendly, directly address sender,
end with a clear next step. No subject line. No greeting like "Dear" — start directly.

From: {sender}
Subject: {subject}
Email: {body}
Reply:
"""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    """
Summarize this email in 1-2 short sentences for a voice notification.
Write naturally as if speaking aloud. Start with the sender's first name.
Focus on the key point. Maximum 40 words.

From: {sender_name}
Subject: {subject}
Body: {body}
Spoken summary:
"""
)

REFINE_PROMPT = PromptTemplate.from_template(
    """
You are a professional email writing assistant.
The user has spoken a rough, conversational message.
Rewrite it into a clean, grammatically correct, and natural sentence.

Rules:
- Fix grammar, spelling, and typos
- Keep the original meaning exactly
- Do not add new information
- Output ONLY the corrected sentence
- No conversational filler, intro text, greetings, or sign-offs

User's rough spoken message: {raw_text}

Refined message:
"""
)


def build_draft_chain(llm):
    return DRAFT_PROMPT | llm


def build_summary_chain(llm):
    return SUMMARY_PROMPT | llm


def build_refine_chain(llm):
    return REFINE_PROMPT | llm


def generate_draft_reply(chain, email):
    log.info(f"Generating draft for: {email['subject'][:60]}")
    try:
        return chain.invoke(
            {
                "sender": email["sender"],
                "subject": email["subject"],
                "body": email["body"],
            }
        ).strip()
    except Exception as e:
        log.error(f"Draft LLM error: {e}")
        return None


def summarize_email(chain, email):
    log.info(f"Summarizing email from: {email['sender_name']}")
    try:
        return chain.invoke(
            {
                "sender_name": email["sender_name"],
                "subject": email["subject"],
                "body": email["body"][:500],
            }
        ).strip()
    except Exception as e:
        log.error(f"Summary LLM error: {e}")
        return f"{email['sender_name']} sent an email about {email['subject']}"


def refine_message(chain, raw_text):
    log.info("Refining message with model...")
    try:
        return chain.invoke({"raw_text": raw_text}).strip()
    except Exception as e:
        log.error(f"Refine LLM error: {e}")
        return None


def generate_subject(chain, body):
    log.info("Generating subject line...")
    try:
        return chain.invoke({"body": body[:300]}).strip()
    except Exception as e:
        log.error(f"Subject LLM error: {e}")
        return "Draft Email"
