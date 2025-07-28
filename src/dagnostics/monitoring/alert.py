# from abc import ABC, abstractmethod
# from typing import Dict, Any
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from logging import Logger

# from dagnostics.core.models import AnalysisResult, ErrorSeverity

# logger = Logger(__name__)

# class AlertProvider(ABC):
#     """Abstract base class for alert providers"""

#     @abstractmethod
#     async def send_alert(self, message: str, recipient: str, **kwargs):
#         pass

# class EmailAlertProvider(AlertProvider):
#     """Email alert provider"""

#     def __init__(self, smtp_server: str, smtp_port: int, username: str,
#                  password: str, from_address: str):
#         self.smtp_server = smtp_server
#         self.smtp_port = smtp_port
#         self.username = username
#         self.password = password
#         self.from_address = from_address

#     async def send_alert(self, message: str, recipient: str, subject: str = "DAGnostics Alert"):
#         """Send email alert"""
#         try:
#             msg = MIMEMultipart()
#             msg['From'] = self.from_address
#             msg['To'] = recipient
#             msg['Subject'] = subject

#             msg.attach(MIMEText(message, 'html'))

#             with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
#                 server.starttls()
#                 server.login(self.username, self.password)
#                 server.send_message(msg)

#             logger.info(f"Email alert sent to {recipient}")
#         except Exception as e:
#             logger.error(f"Failed to send email alert: {e}")
#             raise

# class SMSAlertProvider(AlertProvider):
#     """SMS alert provider using Twilio"""

#     def __init__(self, account_sid: str, auth_token: str, from_number: str):
#         self.account_sid = account_sid
#         self.auth_token = auth_token
#         self.from_number = from_number

#     async def send_alert(self, message: str, recipient: str, **kwargs):
#         """Send SMS alert"""
#         try:
#             from twilio.rest import Client

#             client = Client(self.account_sid, self.auth_token)

#             message = client.messages.create(
#                 body=message,
#                 from_=self.from_number,
#                 to=recipient
#             )

#             logger.info(f"SMS alert sent to {recipient}: {message.sid}")
#         except Exception as e:
#             logger.error(f"Failed to send SMS alert: {e}")
#             raise

# class AlertManager:
#     """Manages different alert providers and routing"""

#     def __init__(self, config: dict):
#         self.providers = {}
#         self.config = config
#         self._setup_providers()

#     def _setup_providers(self):
#         """Setup alert providers based on configuration"""
#         if self.config.get('alerts', {}).get('email', {}).get('enabled'):
#             email_config = self.config['alerts']['email']
#             self.providers['email'] = EmailAlertProvider(
#                 smtp_server=email_config['smtp_server'],
#                 smtp_port=email_config['smtp_port'],
#                 username=email_config['username'],
#                 password=email_config['password'],
#                 from_address=email_config['from_address']
#             )

#         if self.config.get('alerts', {}).get('sms', {}).get('enabled'):
#             sms_config = self.config['alerts']['sms']
#             self.providers['sms'] = SMSAlertProvider(
#                 account_sid=sms_config['account_sid'],
#                 auth_token=sms_config['auth_token'],
#                 from_number=sms_config['from_number']
#             )

#     async def send_alert(self, analysis_result: AnalysisResult):
#         """Send appropriate alerts based on analysis result"""
#         if not analysis_result.analysis:
#             return

#         severity = analysis_result.analysis.severity
#         error_msg = analysis_result.analysis.error_message

#         # Create alert message
#         alert_message = self._create_alert_message(analysis_result)

#         # Send SMS for critical/high severity
#         if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] and 'sms' in self.providers:
#             recipients = self._get_sms_recipients(analysis_result)
#             for recipient in recipients:
#                 await self.providers['sms'].send_alert(
#                     message=self._create_sms_message(analysis_result),
#                     recipient=recipient
#                 )

#         # Send email for all severities
#         if 'email' in self.providers:
#             recipients = self._get_email_recipients(analysis_result)
#             for recipient in recipients:
#                 await self.providers['email'].send_alert(
#                     message=alert_message,
#                     recipient=recipient,
#                     subject=f"DAGnostics Alert: {analysis_result.dag_id}.{analysis_result.task_id}"
#                 )

#     def _create_alert_message(self, result: AnalysisResult) -> str:
#         """Create HTML alert message"""
#         return f"""
#         <html>
#         <body>
#             <h2>🚨 DAGnostics Alert</h2>
#             <p><strong>Task:</strong> {result.dag_id}.{result.task_id}</p>
#             <p><strong>Run ID:</strong> {result.run_id}</p>
#             <p><strong>Error:</strong> {result.analysis.error_message}</p>
#             <p><strong>Category:</strong> {result.analysis.category.value}</p>
#             <p><strong>Severity:</strong> {result.analysis.severity.value}</p>
#             <p><strong>Confidence:</strong> {result.analysis.confidence:.1%}</p>

#             <h3>Suggested Actions:</h3>
#             <ul>
#             {''.join(f'<li>{action}</li>' for action in result.analysis.suggested_actions)}
#             </ul>

#             <p><em>Generated at {result.timestamp}</em></p>
#         </body>
#         </html>
#         """

#     def _create_sms_message(self, result: AnalysisResult) -> str:
#         """Create short SMS message"""
#         return (f"DAGnostics Alert: {result.dag_id}.{result.task_id} failed. "
#                 f"Error: {result.analysis.error_message[:100]}... "
#                 f"Severity: {result.analysis.severity.value}")[:160]

#     def _get_sms_recipients(self, result: AnalysisResult) -> List[str]:
#         """Get SMS recipients based on DAG/severity"""
#         # This would be configurable per DAG or team
#         return ["+1234567890"]  # Placeholder

#     def _get_email_recipients(self, result: AnalysisResult) -> List[str]:
#         """Get email recipients based on DAG/severity"""
#         # This would be configurable per DAG or team
#         return ["alerts@company.com"]  # Placeholder
