"""
LLM Prompts for DAGnostics

This module contains all the prompts used by the LLM engine for error analysis.
Prompts are designed to be provider-agnostic and can be customized as needed.
"""

from typing import Optional

# Error extraction prompt for full analysis
ERROR_EXTRACTION_PROMPT = """
You are an expert ETL engineer analyzing Airflow task failure logs. Your job is to identify the root cause error from noisy log data.

Log Context:
{log_context}

DAG ID: {dag_id}
Task ID: {task_id}

Instructions:
1. Identify the PRIMARY error that caused the task failure
2. Ignore informational, debug, or warning messages unless they're the root cause
3. Focus on the MOST RELEVANT error line(s)
4. Provide confidence score (0.0-1.0)
5. Suggest error category and severity

Respond in JSON format:
{{
    "error_message": "Exact error message that caused the failure",
    "confidence": 0.85,
    "category": "resource_error|data_quality|dependency_failure|configuration_error|permission_error|timeout_error|unknown",
    "severity": "low|medium|high|critical",
    "reasoning": "Brief explanation of why this is the root cause",
    "error_lines": ["specific log lines that contain the error"]
}}
"""

# Provider-specific additions for error extraction
GEMINI_ERROR_EXTRACTION_ADDITION = """
IMPORTANT: Respond with valid JSON only. Do not include any markdown formatting or code blocks.
"""

# Error categorization prompt
ERROR_CATEGORIZATION_PROMPT = """
Categorize this error into one of the following categories:

Error: {error_message}
Context: {context}

Categories:
- resource_error: Memory, CPU, disk space, connection limits
- data_quality: Bad data, schema mismatches, validation failures
- dependency_failure: Upstream task failures, external service unavailable
- configuration_error: Wrong settings, missing parameters, bad configs
- permission_error: Access denied, authentication failures
- timeout_error: Operations taking too long, deadlocks
- unknown: Cannot determine category

Respond with just the category name (e.g., "resource_error").
"""

# Resolution suggestion prompt
RESOLUTION_SUGGESTION_PROMPT = """
Based on the following error analysis, provide 3-5 specific, actionable resolution steps:

Error: {error_message}
Category: {category}
Severity: {severity}

Provide resolution steps as a numbered list. Be specific and technical.
Focus on root cause resolution, not just symptoms.

Resolution Steps:
"""

# SMS error extraction prompt (lightweight LLM analysis for notifications)
SMS_ERROR_EXTRACTION_PROMPT = """
Extract the most important error message from these Airflow task logs for SMS notification.

Log Context:
{log_context}

DAG ID: {dag_id}
Task ID: {task_id}

Instructions:
1. Find the PRIMARY error that caused the task failure
2. Return ONLY the exact error message
3. Keep it concise and actionable (max 160 chars for SMS)
4. Ignore informational and debug messages
5. Focus on the root cause, not symptoms

Return just the error message, nothing else.
"""

# Prompt templates for different use cases
PROMPT_TEMPLATES = {
    "error_extraction": ERROR_EXTRACTION_PROMPT,
    "error_categorization": ERROR_CATEGORIZATION_PROMPT,
    "resolution_suggestion": RESOLUTION_SUGGESTION_PROMPT,
    "sms_error_extraction": SMS_ERROR_EXTRACTION_PROMPT,
}

# Provider-specific prompt modifications
PROVIDER_MODIFICATIONS = {
    "gemini": {
        "error_extraction": GEMINI_ERROR_EXTRACTION_ADDITION,
    }
}


def get_prompt(prompt_name: str, provider_type: Optional[str] = None, **kwargs) -> str:
    """
    Get a prompt template with optional provider-specific modifications.

    Args:
        prompt_name: Name of the prompt template
        provider_type: Type of LLM provider (e.g., 'gemini', 'openai')
        **kwargs: Variables to format the prompt template

    Returns:
        Formatted prompt string
    """
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {prompt_name}")

    prompt = PROMPT_TEMPLATES[prompt_name]

    # Add provider-specific modifications
    if provider_type and provider_type in PROVIDER_MODIFICATIONS:
        if prompt_name in PROVIDER_MODIFICATIONS[provider_type]:
            prompt += PROVIDER_MODIFICATIONS[provider_type][prompt_name]

    # Format the prompt with provided variables
    return prompt.format(**kwargs)


def get_error_extraction_prompt(
    log_context: str, dag_id: str, task_id: str, provider_type: Optional[str] = None
) -> str:
    """Get the error extraction prompt for full analysis."""
    return get_prompt(
        "error_extraction",
        provider_type=provider_type,
        log_context=log_context,
        dag_id=dag_id,
        task_id=task_id,
    )


def get_categorization_prompt(error_message: str, context: str = "") -> str:
    """Get the error categorization prompt."""
    return get_prompt(
        "error_categorization", error_message=error_message, context=context
    )


def get_resolution_prompt(error_message: str, category: str, severity: str) -> str:
    """Get the resolution suggestion prompt."""
    return get_prompt(
        "resolution_suggestion",
        error_message=error_message,
        category=category,
        severity=severity,
    )


def get_sms_error_prompt(
    log_context: str, dag_id: str, task_id: str, provider_type: Optional[str] = None
) -> str:
    """Get the SMS error extraction prompt for lightweight LLM analysis."""
    return get_prompt(
        "sms_error_extraction",
        provider_type=provider_type,
        log_context=log_context,
        dag_id=dag_id,
        task_id=task_id,
    )
