from typing import Any, Dict, List, Optional, Protocol

import requests


class BaseSMSGateway(Protocol):
    def send_sms(self, recipient: str, message: str) -> Dict[str, Any]: ...


class URLBuilder:
    """Builder for constructing SMS Gateway URLs using config dict"""

    def __init__(self):
        self.reset()

    def reset(self) -> "URLBuilder":
        self._config: Dict[str, Any] = {}
        self._dynamic_params: Dict[str, Any] = {}
        return self

    def load_config(self, config: Dict[str, Any]) -> "URLBuilder":
        """Load configuration from a config dict (from config.yaml)"""
        self._config = config
        return self

    def with_message(self, message: str) -> "URLBuilder":
        self._dynamic_params["message"] = message
        return self

    def with_recipient(self, recipient: str) -> "URLBuilder":
        self._dynamic_params["recipient"] = recipient
        return self

    def build(self) -> str:
        if not self._config:
            raise ValueError("Configuration not loaded from config dict")
        base_url = self._config["base_url"].rstrip("/")
        path = self._config.get("path", "").lstrip("/")
        static_params = self._config.get("static_params", {})
        param_mapping = self._config.get("param_mapping", {})
        url = f"{base_url}/{path}" if path else base_url
        all_params: Dict[str, Any] = {}
        all_params.update(static_params)
        for param_key, param_value in self._dynamic_params.items():
            mapped_key = param_mapping.get(param_key, param_key)
            all_params[mapped_key] = param_value
        if all_params:
            import urllib.parse

            query_string = urllib.parse.urlencode(all_params)
            url = f"{url}?{query_string}"
        return url


class CustomSMSGateway:
    """SMS Gateway handler using config dict (from config.yaml)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.builder = URLBuilder()

    def send_sms(self, recipient: str, message: str) -> Dict[str, Any]:
        try:
            url = (
                self.builder.reset()
                .load_config(self.config)
                .with_recipient(recipient)
                .with_message(message)
                .build()
            )
            response = requests.get(url, timeout=30)
            return {
                "success": response.status_code < 300,
                "response": response.text,
                "url": url,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class TwilioSMSGateway:
    """Twilio SMS Gateway handler"""

    def __init__(self, config: Dict[str, Any]):
        self.account_sid = config.get("account_sid", "")
        self.auth_token = config.get("auth_token", "")
        self.from_number = config.get("from_number", "")
        if not all([self.account_sid, self.auth_token, self.from_number]):
            raise ValueError(
                "Twilio configuration missing required fields: account_sid, auth_token, from_number"
            )

    def send_sms(self, recipient: str, message: str) -> Dict[str, Any]:
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            data = {"To": recipient, "From": self.from_number, "Body": message}
            response = requests.post(
                url, data=data, auth=(self.account_sid, self.auth_token), timeout=30
            )
            return {
                "success": response.status_code < 300,
                "response": response.text,
                "url": url,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_sms_gateway(config: Dict[str, Any]) -> BaseSMSGateway:
    provider = config.get("provider", "custom")
    if provider == "custom":
        return CustomSMSGateway(config)
    elif provider == "twilio":
        return TwilioSMSGateway(config)
    else:
        raise ValueError(f"Unknown SMS provider: {provider}")


def send_sms_alert(
    message: str,
    recipients: List[str],
    config: Dict[str, Any],
    raddr: Optional[str] = None,
) -> None:
    gateway = get_sms_gateway(config)
    for msisdn in recipients:
        response = gateway.send_sms(recipient=msisdn, message=message)
        if not response["success"]:
            raise Exception(response.get("response") or response.get("error"))
    return
