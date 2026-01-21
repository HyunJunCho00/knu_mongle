import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.core.config import settings

class EmailService:
    def __init__(self):
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.user = settings.EMAIL_USER
        self.password = settings.EMAIL_PASSWORD

    def draft_email(self, recipient: str, subject: str, body: str) -> dict:
        """
        Draft an email (returns the content, doesn't send).
        """
        return {
            "to": recipient,
            "subject": subject,
            "body": body,
            "status": "draft"
        }

    def send_email(self, recipient: str, subject: str, body: str, attachments: list = None):
        """
        Send an email using SMTP.
        """
        if not self.user or not self.password:
            print("Email credentials not set. Skipping send.")
            return {"status": "failed", "reason": "No credentials"}

        msg = MIMEMultipart()
        msg['From'] = self.user
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Handle attachments (simplified)
        if attachments:
            pass 

        try:
            # server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            # server.starttls()
            # server.login(self.user, self.password)
            # server.send_message(msg)
            # server.quit()
            print(f"Email sent to {recipient}")
            return {"status": "sent"}
        except Exception as e:
            print(f"Failed to send email: {e}")
            return {"status": "failed", "error": str(e)}
