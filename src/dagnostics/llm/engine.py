import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import requests
from pydantic import HttpUrl

from dagnostics.core.models import ErrorAnalysis, ErrorCategory, ErrorSeverity, LogEntry

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider"""

    def __init__(
        self,
        base_url: Union[str, HttpUrl] = "http://localhost:11434",
        model: str = "mistral",
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = model

    def generate_response(self, prompt: str, **kwargs) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False, **kwargs}
        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def generate_response(self, prompt: str, **kwargs) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_output_tokens: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self._configure_gemini()

    def _configure_gemini(self):
        """Configure Gemini API"""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)  # type: ignore

            generation_config = {}
            if self.max_output_tokens is not None:
                generation_config["max_output_tokens"] = self.max_output_tokens
            if self.top_p is not None:
                generation_config["top_p"] = self.top_p
            if self.top_k is not None:
                generation_config["top_k"] = self.top_k

            self.gemini_model = genai.GenerativeModel(  # type: ignore
                self.model,
                generation_config=generation_config if generation_config else None,  # type: ignore
            )

        except (ImportError, AttributeError):
            logger.error(
                "google.generativeai package not installed. Install with: pip install google-generativeai"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            # Extract generation config parameters, with method-level overrides
            generation_config = {}

            # Use instance defaults first, then override with kwargs
            if self.max_output_tokens is not None:
                generation_config["max_output_tokens"] = self.max_output_tokens
            if self.top_p is not None:
                generation_config["top_p"] = self.top_p
            if self.top_k is not None:
                generation_config["top_k"] = self.top_k

            # Override with method-level parameters
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")
            if "max_output_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs.pop("max_output_tokens")
            if "top_p" in kwargs:
                generation_config["top_p"] = kwargs.pop("top_p")
            if "top_k" in kwargs:
                generation_config["top_k"] = kwargs.pop("top_k")

            # Create a new model instance if we need to override the generation config
            if generation_config and any(
                key in kwargs
                for key in ["temperature", "max_output_tokens", "top_p", "top_k"]
            ):
                import google.generativeai as genai

                temp_model = genai.GenerativeModel(  # type: ignore
                    self.model, generation_config=generation_config  # type: ignore
                )
                response = temp_model.generate_content(prompt)
            else:
                # Use the pre-configured model
                response = self.gemini_model.generate_content(prompt)

            # Handle potential response issues
            if not response.text:
                logger.warning("Gemini returned empty response")
                return ""

            return response.text

        except Exception as e:
            # More specific error handling for common Gemini issues
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate limit" in error_msg:
                logger.error(f"Gemini quota/rate limit exceeded: {e}")
            elif "safety" in error_msg or "blocked" in error_msg:
                logger.error(f"Gemini content blocked by safety filters: {e}")
            elif "api key" in error_msg:
                logger.error(f"Gemini API key issue: {e}")
            else:
                logger.error(f"Gemini request failed: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            # Extract anthropic-specific parameters
            max_tokens = kwargs.pop("max_tokens", 1000)
            temperature = kwargs.pop("temperature", 0.1)

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            return response.content[0].text

        except ImportError:
            logger.error(
                "anthropic package not installed. Install with: pip install anthropic"
            )
            raise
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise


class LLMEngine:
    """Provider-agnostic LLM interface for error analysis"""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.error_extraction_prompt = self._load_error_extraction_prompt()
        self.categorization_prompt = self._load_categorization_prompt()

    def extract_error_message(self, log_entries: List[LogEntry]) -> ErrorAnalysis:
        """Extract and analyze error from log entries"""

        # Prepare log context
        log_context = "\n".join(
            [
                f"[{entry.timestamp}] {entry.level}: {entry.message}"
                for entry in log_entries[-10:]  # Last 10 entries for context
            ]
        )

        prompt = self.error_extraction_prompt.format(
            log_context=log_context,
            dag_id=log_entries[0].dag_id if log_entries else "unknown",
            task_id=log_entries[0].task_id if log_entries else "unknown",
        )

        try:
            # Use provider-specific parameters for better results
            kwargs = {"temperature": 0.1}
            if isinstance(self.provider, GeminiProvider):
                # Gemini-specific optimizations for structured output
                kwargs.update({"temperature": 0.1, "top_p": 0.8, "top_k": 40})
            elif isinstance(self.provider, OpenAIProvider):
                # OpenAI-specific optimizations
                kwargs.update({"temperature": 0.1})
                if "gpt-3.5" not in self.provider.model.lower():
                    kwargs["response_format"] = {"type": "json_object"}  # type: ignore

            response = self.provider.generate_response(prompt, **kwargs)
            return self._parse_error_analysis_response(response, log_entries)
        except Exception as e:
            logger.error(f"Error extraction failed: {e}")
            return self._create_fallback_analysis(log_entries, str(e))

    def categorize_error(self, error_message: str, context: str = "") -> ErrorCategory:
        """Categorize error into predefined categories"""

        prompt = self.categorization_prompt.format(
            error_message=error_message, context=context
        )

        try:
            response = self.provider.generate_response(prompt, temperature=0.0)
            return self._parse_category_response(response)
        except Exception as e:
            logger.error(f"Error categorization failed: {e}")
            return ErrorCategory.UNKNOWN

    def suggest_resolution(self, error_analysis: ErrorAnalysis) -> List[str]:
        """Suggest resolution steps based on error analysis"""

        resolution_prompt = f"""
Based on the following error analysis, provide 3-5 specific, actionable resolution steps:

Error: {error_analysis.error_message}
Category: {error_analysis.category.value}
Severity: {error_analysis.severity.value}

Provide resolution steps as a numbered list. Be specific and technical.
Focus on root cause resolution, not just symptoms.

Resolution Steps:
"""

        try:
            # Use slightly higher temperature for more creative resolution suggestions
            kwargs = {"temperature": 0.2}
            if isinstance(self.provider, GeminiProvider):
                kwargs.update({"temperature": 0.3, "top_p": 0.9})

            response = self.provider.generate_response(resolution_prompt, **kwargs)
            return self._parse_resolution_steps(response)
        except Exception as e:
            logger.error(f"Resolution suggestion failed: {e}")
            return [
                "Manual investigation required",
                "Check system logs",
                "Contact support",
            ]

    def _load_error_extraction_prompt(self) -> str:
        # Enhanced prompt with better instructions for different LLM providers
        base_prompt = """
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
}}"""

        # Add provider-specific instructions
        if isinstance(self.provider, GeminiProvider):
            return (
                base_prompt
                + "\n\nIMPORTANT: Respond with valid JSON only. Do not include any markdown formatting or code blocks."
            )

        return base_prompt

    def _load_categorization_prompt(self) -> str:
        return """
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

    def _parse_error_analysis_response(
        self, response: str, log_entries: List[LogEntry]
    ) -> ErrorAnalysis:
        """Parse LLM response into ErrorAnalysis object"""
        try:
            # Clean response for better JSON parsing (especially for Gemini)
            cleaned_response = response.strip()

            # Remove markdown code blocks if present
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            data = json.loads(cleaned_response)

            return ErrorAnalysis(
                error_message=data.get("error_message", "Unknown error"),
                confidence=float(data.get("confidence", 0.5)),
                category=ErrorCategory(data.get("category", "unknown")),
                severity=ErrorSeverity(data.get("severity", "medium")),
                suggested_actions=[],  # Will be populated by suggest_resolution
                related_logs=log_entries,
                raw_error_lines=data.get("error_lines", []),
                llm_reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response}")
            return self._create_fallback_analysis(log_entries, f"Parse error: {e}")

    def _parse_category_response(self, response: str) -> ErrorCategory:
        """Parse category from LLM response"""
        category_str = response.strip().lower()

        # Handle potential markdown or extra formatting
        category_str = re.sub(r"^```.*\n", "", category_str)
        category_str = re.sub(r"\n```$", "", category_str)
        category_str = category_str.strip()

        try:
            return ErrorCategory(category_str)
        except ValueError:
            logger.warning(f"Unknown category returned: {category_str}")
            return ErrorCategory.UNKNOWN

    def _parse_resolution_steps(self, response: str) -> List[str]:
        """Parse resolution steps from LLM response"""
        lines = response.strip().split("\n")
        steps = []

        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("*")
            ):
                # Remove numbering/bullets
                clean_line = re.sub(r"^\d+\.?\s*", "", line)
                clean_line = re.sub(r"^[-*]\s*", "", clean_line)
                if clean_line:
                    steps.append(clean_line)

        return steps if steps else ["Manual investigation required"]

    def _create_fallback_analysis(
        self, log_entries: List[LogEntry], error_msg: str
    ) -> ErrorAnalysis:
        """Create fallback analysis when LLM fails"""
        return ErrorAnalysis(
            error_message=f"Analysis failed: {error_msg}",
            confidence=0.1,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            suggested_actions=["Manual analysis required", "Check logs manually"],
            related_logs=log_entries,
            raw_error_lines=[],
            llm_reasoning="LLM analysis failed",
        )
