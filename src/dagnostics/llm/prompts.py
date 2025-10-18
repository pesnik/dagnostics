"""
LLM Prompts for DAGnostics

This module contains all the prompts used by the LLM engine for error analysis.
Prompts are designed to be provider-agnostic and can be customized as needed.
"""

from typing import Optional

from dagnostics.core.models import AppConfig, FewShotExample

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
5. Categorize into exactly ONE category from the list below
6. Assign severity level

Available Categories (choose exactly ONE):
- resource_error: Memory, CPU, disk space, connection pool exhaustion
- data_quality: Bad data format, schema mismatches, validation failures
- dependency_failure: Upstream task failures, external service unavailable, network issues
- configuration_error: Wrong settings, missing parameters, misconfigured services
- permission_error: Access denied, authentication failures, authorization issues
- timeout_error: Operations taking too long, deadlocks, connection timeouts
- unknown: Cannot determine category with confidence

Respond in STRICT JSON format (no additional text or markdown):
{{
    "error_message": "Exact error message that caused the failure",
    "confidence": 0.85,
    "category": "dependency_failure",
    "severity": "high",
    "reasoning": "Brief explanation of why this is the root cause",
    "error_lines": ["specific log lines that contain the error"]
}}

CRITICAL: The "category" field must contain EXACTLY ONE category name from the list above. Do not use pipe symbols or combine multiple categories.
"""

# Provider-specific additions for error extraction
GEMINI_ERROR_EXTRACTION_ADDITION = """
IMPORTANT: Respond with valid JSON only. Do not include any markdown formatting or code blocks.
"""

# Error categorization prompt
ERROR_CATEGORIZATION_PROMPT = """
Categorize this error into EXACTLY ONE of the following categories:

Error: {error_message}
Context: {context}

Categories (choose only ONE):
- resource_error: Memory, CPU, disk space, connection limits
- data_quality: Bad data, schema mismatches, validation failures
- dependency_failure: Upstream task failures, external service unavailable, network issues
- configuration_error: Wrong settings, missing parameters, bad configs
- permission_error: Access denied, authentication failures
- timeout_error: Operations taking too long, deadlocks
- unknown: Cannot determine category

IMPORTANT: Respond with ONLY the category name as a single word.
Examples of correct responses:
- "dependency_failure"
- "timeout_error"
- "configuration_error"

Do NOT combine categories with pipes (|) or commas. Choose the MOST relevant single category.
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


def get_prompt(
    prompt_name: str,
    provider_type: Optional[str] = None,
    config: Optional[AppConfig] = None,
    **kwargs,
) -> str:
    """
    Get a prompt template with optional provider-specific modifications and config overrides.

    Args:
        prompt_name: Name of the prompt template
        provider_type: Type of LLM provider (e.g., 'gemini', 'openai')
        config: App configuration with prompt overrides
        **kwargs: Variables to format the prompt template

    Returns:
        Formatted prompt string
    """
    # First check for config-based template override
    if config and config.prompts and prompt_name in config.prompts.templates:
        prompt = config.prompts.templates[prompt_name]
    else:
        # Fallback to default templates
        if prompt_name not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown prompt template: {prompt_name}")
        prompt = PROMPT_TEMPLATES[prompt_name]

    # Add few-shot examples if available
    if config and config.prompts and prompt_name in config.prompts.few_shot_examples:
        examples = config.prompts.few_shot_examples[prompt_name]
        few_shot_text = _format_few_shot_examples(examples)
        kwargs["few_shot_examples"] = few_shot_text

    # Add provider-specific modifications
    if provider_type and provider_type in PROVIDER_MODIFICATIONS:
        if prompt_name in PROVIDER_MODIFICATIONS[provider_type]:
            prompt += PROVIDER_MODIFICATIONS[provider_type][prompt_name]

    # Format the prompt with provided variables
    return prompt.format(**kwargs)


def get_error_extraction_prompt(
    log_context: str,
    dag_id: str,
    task_id: str,
    provider_type: Optional[str] = None,
    config: Optional[AppConfig] = None,
) -> str:
    """Get the error extraction prompt for full analysis."""
    return get_prompt(
        "error_extraction",
        provider_type=provider_type,
        config=config,
        log_context=log_context,
        dag_id=dag_id,
        task_id=task_id,
    )


def get_categorization_prompt(
    error_message: str, context: str = "", config: Optional[AppConfig] = None
) -> str:
    """Get the error categorization prompt."""
    return get_prompt(
        "error_categorization",
        config=config,
        error_message=error_message,
        context=context,
    )


def get_resolution_prompt(
    error_message: str, category: str, severity: str, config: Optional[AppConfig] = None
) -> str:
    """Get the resolution suggestion prompt."""
    return get_prompt(
        "resolution_suggestion",
        config=config,
        error_message=error_message,
        category=category,
        severity=severity,
    )


def get_sms_error_prompt(
    log_context: str,
    dag_id: str,
    task_id: str,
    provider_type: Optional[str] = None,
    config: Optional[AppConfig] = None,
) -> str:
    """Get the SMS error extraction prompt for lightweight LLM analysis."""
    return get_prompt(
        "sms_error_extraction",
        provider_type=provider_type,
        config=config,
        log_context=log_context,
        dag_id=dag_id,
        task_id=task_id,
    )


def _format_few_shot_examples(examples: list[FewShotExample]) -> str:
    """Format few-shot examples for inclusion in prompts."""
    formatted_examples = []

    for i, example in enumerate(examples[:5], 1):  # Limit to 5 examples max
        formatted_examples.append(
            f"Example {i}:\n"
            f"Log Context:\n{example.log_context}\n\n"
            f"Expected Response:\n{example.extracted_response}\n"
        )

    return "\n" + "\n---\n".join(formatted_examples) + "\n---\n"
