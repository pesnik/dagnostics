import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import requests
from pydantic import HttpUrl

from dagnostics.core.models import ErrorAnalysis, ErrorCategory, ErrorSeverity, LogEntry
from dagnostics.llm.prompts import (
    get_categorization_prompt,
    get_error_extraction_prompt,
    get_resolution_prompt,
    get_sms_error_prompt,
)

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
                    self.model,
                    generation_config=generation_config,  # type: ignore
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

    def extract_error_message(self, log_entries: List[LogEntry]) -> ErrorAnalysis:
        """Extract and analyze error from log entries"""

        # Prepare log context
        log_context = "\n".join(
            [
                f"[{entry.timestamp}] {entry.level}: {entry.message}"
                for entry in log_entries[-10:]  # Last 10 entries for context
            ]
        )

        # Determine provider type for prompt customization
        provider_type = None
        if isinstance(self.provider, GeminiProvider):
            provider_type = "gemini"
        elif isinstance(self.provider, OpenAIProvider):
            provider_type = "openai"
        elif isinstance(self.provider, AnthropicProvider):
            provider_type = "anthropic"
        elif isinstance(self.provider, OllamaProvider):
            provider_type = "ollama"

        prompt = get_error_extraction_prompt(
            log_context=log_context,
            dag_id=log_entries[0].dag_id if log_entries else "unknown",
            task_id=log_entries[0].task_id if log_entries else "unknown",
            provider_type=provider_type,
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

    def extract_error_line(self, log_entries: List[LogEntry]) -> str:
        """Extract the exact error line for SMS notifications from filtered log candidates"""

        if not log_entries:
            return "No error details available"

        # Prepare log context from filtered candidates (these are already relevant)
        log_context = "\n".join(
            [
                f"[{entry.timestamp}] {entry.level}: {entry.message}"
                for entry in log_entries[
                    -5:
                ]  # Last 5 entries since they're already filtered
            ]
        )

        # Determine provider type for prompt customization
        provider_type = None
        if isinstance(self.provider, GeminiProvider):
            provider_type = "gemini"
        elif isinstance(self.provider, OpenAIProvider):
            provider_type = "openai"
        elif isinstance(self.provider, AnthropicProvider):
            provider_type = "anthropic"
        elif isinstance(self.provider, OllamaProvider):
            provider_type = "ollama"

        # Get SMS-specific prompt
        prompt = get_sms_error_prompt(
            log_context=log_context,
            dag_id=log_entries[0].dag_id if log_entries else "unknown",
            task_id=log_entries[0].task_id if log_entries else "unknown",
            provider_type=provider_type,
        )

        try:
            # Use lower temperature for more consistent SMS error extraction
            kwargs = {"temperature": 0.0}
            if isinstance(self.provider, GeminiProvider):
                kwargs.update({"temperature": 0.0, "top_p": 0.8, "top_k": 40})
            elif isinstance(self.provider, OpenAIProvider):
                kwargs.update({"temperature": 0.0})

            response = self.provider.generate_response(prompt, **kwargs)

            # Clean and validate response
            error_message = response.strip()
            if not error_message or error_message.lower() in [
                "no error",
                "none",
                "unknown",
            ]:
                # Fallback to heuristic if LLM returns empty/unknown
                return self._extract_error_heuristic(log_entries)

            # Truncate if too long for SMS (160 chars)
            if len(error_message) > 160:
                error_message = error_message[:157] + "..."

            return error_message

        except Exception as e:
            logger.error(f"SMS error extraction failed: {e}")
            # Fallback to heuristic method
            return self._extract_error_heuristic(log_entries)

    def _extract_error_heuristic(self, log_entries: List[LogEntry]) -> str:
        """Fallback heuristic error extraction when LLM fails"""
        # Simple heuristic: find the first ERROR/CRITICAL/FATAL level log
        for entry in log_entries:
            if entry.level.upper() in ["ERROR", "CRITICAL", "FATAL"]:
                return entry.message

        # If no ERROR level found, look for error keywords
        error_keywords = [
            "error",
            "exception",
            "failed",
            "failure",
            "fatal",
            "critical",
        ]
        for entry in log_entries:
            message_lower = entry.message.lower()
            if any(keyword in message_lower for keyword in error_keywords):
                return entry.message

        # Fallback: return the last log entry
        if log_entries:
            return log_entries[-1].message

        return "No error details available"

    def categorize_error(self, error_message: str, context: str = "") -> ErrorCategory:
        """Categorize error into predefined categories"""

        prompt = get_categorization_prompt(error_message=error_message, context=context)

        try:
            response = self.provider.generate_response(prompt, temperature=0.0)
            return self._parse_category_response(response)
        except Exception as e:
            logger.error(f"Error categorization failed: {e}")
            return ErrorCategory.UNKNOWN

    def suggest_resolution(self, error_analysis: ErrorAnalysis) -> List[str]:
        """Suggest resolution steps based on error analysis"""

        resolution_prompt = get_resolution_prompt(
            error_message=error_analysis.error_message,
            category=error_analysis.category.value,
            severity=error_analysis.severity.value,
        )

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
