"""Fallback utilities for degraded API key state or local-only query responses."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


GREETING_RESPONSES = {
    "hi": "Hi there! I'm Job Advisor AI. I can help you with job search, resume tips, interview prep, and career advice. What would you like to know?",
    "hello": "Hello! Welcome to Job Advisor AI. I'm here to help with jobs, skills, interviews, and career growth. How can I assist you?",
    "hey": "Hey! I'm your AI job advisor. Ask me anything about careers, skills, or job searching.",
    "how are you": "I'm doing great, thanks for asking! I'm ready to help you with any job or career questions you have.",
    "good morning": "Good morning! Hope you're having a productive day. Let me know if you need help with anything career-related.",
    "good afternoon": "Good afternoon! I'm here if you need help with resumes, interviews, or career planning.",
    "good evening": "Good evening! I can help you with job search tips, resume advice, or interview preparation.",
    "thank you": "You're welcome! Feel free to ask if you have more questions.",
    "thanks": "You're welcome! Let me know if there's anything else I can help with.",
}


@dataclass
class FallbackSummary:
    query: str
    summary: str
    sources: list[str]


def _extract_skills_from_chunks(chunks: Iterable[object]) -> list[str]:
    """Extract skill mentions from dataset chunks."""
    skills = []
    for chunk in chunks:
        text = getattr(chunk, "text", "")
        match = re.search(r"skills:\s*([^|]+)", text, re.IGNORECASE)
        if match:
            skill_list = [s.strip() for s in match.group(1).split(",") if s.strip()]
            skills.extend(skill_list)
    return list(dict.fromkeys(skills))[:10]


def _extract_careers_from_chunks(chunks: Iterable[object]) -> list[str]:
    """Extract career labels from dataset chunks."""
    careers = []
    for chunk in chunks:
        text = getattr(chunk, "text", "")
        match = re.search(r"career_label:\s*(.+?)(?:\s*\||$)", text, re.IGNORECASE)
        if match:
            career = match.group(1).strip()
            if career:
                careers.append(career)
    return list(dict.fromkeys(careers))


def _format_dataset_info(chunks: Iterable[object]) -> str:
    """Format dataset chunks as readable information."""
    lines = []
    for chunk in chunks:
        text = getattr(chunk, "text", "")
        parts = [p.strip() for p in text.split("|") if p.strip()]
        formatted = ", ".join(parts)
        if len(formatted) > 200:
            formatted = formatted[:200] + "..."
        lines.append(f"  - {formatted}")
    return "\n".join(lines[:5])


def extractive_summary_from_chunks(query: str, chunks: Iterable[object], top_n: int = 3) -> FallbackSummary:
    """Create a conversational fallback answer from retrieved dataset chunks.

    The engine only requires chunk objects with `.text`, `.source`, and `.row_id`.
    """
    query_lower = query.lower().strip()
    sources = []
    chunks_list = list(chunks)

    for chunk in chunks_list:
        sources.append(f"{getattr(chunk, 'source', 'unknown')} row {getattr(chunk, 'row_id', -1)}")

    # Check for greetings
    for greeting, response in GREETING_RESPONSES.items():
        if greeting in query_lower:
            return FallbackSummary(query=query, summary=response, sources=[])

    # Check for skill-related questions
    skill_keywords = ["skill", "technolog", "programming", "language", "tool"]
    if any(kw in query_lower for kw in skill_keywords):
        skills = _extract_skills_from_chunks(chunks_list)
        if skills:
            summary = f"Based on the dataset, here are relevant skills for your question:\n\n"
            summary += ", ".join(f"**{s}**" for s in skills)
            summary += "\n\nThese skills appear frequently in job profiles. Focus on building proficiency in the ones most relevant to your target role."
            return FallbackSummary(query=query, summary=summary, sources=sources)

    # Check for career/role questions
    career_keywords = ["career", "job", "role", "position", "become", "work as"]
    if any(kw in query_lower for kw in career_keywords):
        careers = _extract_careers_from_chunks(chunks_list)
        if careers:
            summary = f"Here are some related career paths from the dataset:\n\n"
            for c in careers:
                summary += f"- {c}\n"
            summary += "\nWould you like to know more about any of these roles?"
            return FallbackSummary(query=query, summary=summary, sources=sources)

    # Check for education questions
    edu_keywords = ["education", "degree", "university", "college", "study", "qualification"]
    if any(kw in query_lower for kw in edu_keywords):
        info = _format_dataset_info(chunks_list)
        if info:
            summary = f"Based on the dataset, here are relevant education profiles:\n\n{info}\n\nDifferent roles have varying education requirements. What specific role are you targeting?"
            return FallbackSummary(query=query, summary=summary, sources=sources)

    # General fallback with dataset context
    if chunks_list:
        info = _format_dataset_info(chunks_list)
        summary = f"Here's what I found in our job database related to your question:\n\n{info}\n\nFor more detailed help, I'd recommend asking about specific skills, career paths, or job requirements. What would you like to explore?"
    else:
        summary = "I don't have specific data matching your query in my local database. I can help with:\n\n- Job search strategies\n- Resume and interview tips\n- Career transition advice\n- Skill development recommendations\n\nCould you rephrase your question or ask about one of these topics?"

    return FallbackSummary(query=query, summary=summary, sources=sources)
