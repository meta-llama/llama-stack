from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from bs4 import BeautifulSoup
import os
import pytz
import base64
import pickle
from datetime import datetime, timezone
import ollama
from pypdf import PdfReader
from pathlib import Path
from shared import memory

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.compose']
user_email = None
service = None
user_id = 'me'

def authenticate_gmail(user_email):
    creds = None
    token_file = f'token_{user_email}.pickle'  # Unique token file for each user
    
    # Load the user's token if it exists
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, prompt the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_console()
        
        # Save the new credentials to a user-specific token file
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    # Build the Gmail API service
    service = build('gmail', 'v1', credentials=creds)
    return service

def num_of_emails(query=''):
    response = service.users().messages().list(
        userId='me', 
        q=query, 
        maxResults=1).execute()
    return response.get('resultSizeEstimate', 0)

def list_emails(query='', max_results=100):
    emails = []
    next_page_token = None

    while True:
        response = service.users().messages().list(
            userId=user_id,
            maxResults=max_results,
            pageToken=next_page_token,
            q=query
        ).execute()
        
        if 'messages' in response:
            for msg in response['messages']:
                sender, subject, received_time = get_email_info(msg['id'])
                emails.append(
                    {
                        "message_id": msg['id'],
                        "sender": sender,
                        "subject": subject,
                        "received_time": received_time
                    }
                )
        
        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break
    
    return emails

def get_email_detail(detail, which):
    message_id = None
    # pre-processing
    if 'from ' in which:
        sender = which.split('from ')[-1]
        for email in memory['emails']:
            if email['sender'].find(sender) != -1:
                message_id = email['message_id']
                break
    elif 'subject:' in which:
        subject = which.split('subject:')[-1]
        # exact match beats substring
        for email in memory['emails']:
            if email['subject'].upper() == subject.upper():
                message_id = email['message_id']
                break
            elif email['subject'].upper().find(subject.upper()) != -1:
                message_id = email['message_id']

    elif 'id_' in which:
        message_id = which.split('id_')[-1]
    else:
        message_id = memory['emails'][-1]['message_id']

    if detail == 'body':
        return get_email_body(message_id)
    elif detail == 'attachment':
        return get_email_attachments(message_id)

def get_email_body(message_id):
    try:
        message = service.users().messages().get(
            userId=user_id, 
            id=message_id, 
            format='full').execute()

        # Recursive function to extract the parts
        def extract_parts(payload):
            text_body = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    return extract_parts(part)
            else:
                mime_type = payload.get('mimeType')
                body = payload.get('body', {}).get('data')
                if mime_type == 'text/html':
                    decoded_body = base64.urlsafe_b64decode(body).decode('utf-8')
                    soup = BeautifulSoup(decoded_body, 'html.parser')
                    text_body = soup.get_text().strip()
                elif mime_type == 'text/plain':
                    decoded_body = base64.urlsafe_b64decode(body).decode('utf-8')
                    text_body = decoded_body

                return text_body

        return extract_parts(message['payload'])

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def parse_message(message):
    payload = message['payload']
    headers = payload.get("headers")

    subject = None
    sender = None
    for header in headers:
        if header['name'] == 'Subject':
            subject = header['value']
        elif header['name'] == 'From':
            sender = header['value']    

    internal_date = message.get('internalDate')  
    utc_time = datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc)
    
    # Convert UTC to the specified timezone
    local_timezone = pytz.timezone("America/Los_Angeles")
    local_time = utc_time.astimezone(local_timezone)
    
    # Format the local time as a string
    received_time = local_time.strftime('%Y-%m-%d %H:%M:%S %Z')

    # Check if the email is plain text or multipart
    if 'parts' in payload:
        # Multipart message - find the text/plain or text/html part
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' or part['mimeType'] == 'text/html':  # You can also look for 'text/html'
                data = part['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8')
                return sender, subject, received_time, body
            elif part['mimeType'] in ['multipart/related', 'multipart/mixed', 'multipart/alternative']:
                return sender, subject, received_time, get_email_body(message.get('id'))
    else:
        # Single part message
        data = payload['body']['data']
        body = base64.urlsafe_b64decode(data).decode('utf-8')
        return sender, subject, received_time, body

def get_email_info(msg_id):
    message = service.users().messages().get(
        userId=user_id, 
        id=msg_id, 
        format='full').execute()

    sender, subject, received_time, body = parse_message(message)
    
    return sender, subject, received_time

def reply_email(message_id, reply_text):
    # Fetch the original message
    original_message = service.users().messages().get(
        userId=user_id, 
        id=message_id, 
        format='full').execute()
    
    # Get headers
    headers = original_message['payload']['headers']
    subject = None
    to = None
    for header in headers:
        if header['name'] == 'Subject':
            subject = header['value']
        if header['name'] == 'From':
            to = header['value']
    
    # Create the reply subject
    if not subject.startswith("Re: "):
        subject = "Re: " + subject

    # Compose the reply message
    reply_message = MIMEText(reply_text)
    reply_message['to'] = to
    reply_message['from'] = user_id
    reply_message['subject'] = subject
    reply_message['In-Reply-To'] = message_id
    
    # Encode and send the message
    raw_message = base64.urlsafe_b64encode(reply_message.as_bytes()).decode("utf-8")
    body = {'raw': raw_message, 
            'threadId': original_message['threadId']}
    sent_message = service.users().messages().send(
        userId=user_id, 
        body=body).execute()
    print("Reply sent. Message ID:", sent_message['id'])

def forward_email(message_id, forward_to, email_body=None):
    """
    Forwards an email, preserving the original MIME type, including multipart/related.
    """
    # Get the original message in 'full' format
    original_message = service.users().messages().get(
        userId=user_id,
        id=message_id,
        format='full').execute()

    # Extract the payload and headers
    payload = original_message.get('payload', {})
    headers = payload.get('headers', [])
    parts = payload.get('parts', [])
    # Get the Subject
    subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')

    # Create a new MIME message for forwarding
    mime_message = MIMEMultipart(payload.get('mimeType', 'mixed').split('/')[-1])
    mime_message['To'] = forward_to
    mime_message['Subject'] = f"Fwd: {subject}"

    # Add the optional custom email body
    if email_body:
        mime_message.attach(MIMEText(email_body, 'plain'))

    # Function to fetch attachment data by attachmentId
    def fetch_attachment_data(attachment_id, message_id):
        attachment = service.users().messages().attachments().get(
            userId=user_id, messageId=message_id, id=attachment_id
        ).execute()
        return base64.urlsafe_b64decode(attachment['data'])

    # Rebuild MIME structure
    def rebuild_parts(parts):
        """
        Recursively rebuild MIME parts.
        """
        if not parts:
            return None

        for part in parts:
            part_mime_type = part.get('mimeType', 'text/plain')
            part_body = part.get('body', {})
            part_data = part_body.get('data', '')
            part_parts = part.get('parts', [])  # Sub-parts for multipart types
            filename = part.get('filename')
            attachment_id = part_body.get('attachmentId')

            if part_mime_type.startswith('multipart/'):
                # Rebuild nested multipart
                sub_multipart = MIMEMultipart(part_mime_type.split('/')[-1])
                sub_parts = rebuild_parts(part_parts)
                if sub_parts:
                    for sub_part in sub_parts:
                        sub_multipart.attach(sub_part)
                yield sub_multipart
            elif filename and attachment_id:
                # Handle attachments
                decoded_data = fetch_attachment_data(attachment_id, message_id)
                attachment = MIMEBase(*part_mime_type.split('/'))
                attachment.set_payload(decoded_data)
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                yield attachment
            else:
                if part_data:
                    # Decode and attach non-multipart parts
                    decoded_data = base64.urlsafe_b64decode(part_data)

                    if part_mime_type == 'text/plain':
                        yield MIMEText(decoded_data.decode('utf-8'), 'plain')
                    elif part_mime_type == 'text/html':
                        yield MIMEText(decoded_data.decode('utf-8'), 'html')

    # Rebuild the main MIME structure
    rebuilt_parts = rebuild_parts(parts)
    if rebuilt_parts:
        for rebuilt_part in rebuilt_parts:
            mime_message.attach(rebuilt_part)

    # Encode the MIME message to base64
    raw = base64.urlsafe_b64encode(mime_message.as_bytes()).decode('utf-8')

    # Send the email
    forward_body = {'raw': raw}
    sent_message = service.users().messages().send(userId=user_id, body=forward_body).execute()

    print(f"Message forwarded successfully! Message ID: {sent_message['id']}")

def send_email(action, to, subject, body="", email_id=""):
    if action == "compose":
        message = MIMEText(body)
        message['to'] = to
        message['from'] = user_id
        message['subject'] = subject
        
        # Encode and send the message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        body = {'raw': raw_message}
        sent_message = service.users().messages().send(
            userId=user_id, 
            body=body).execute()
        return sent_message['id']
    elif action == "reply": # reply or forward; a message id is needed
        reply_email(email_id, body)
    elif action == "forward":
        forward_email(email_id, to, body)

def create_draft(action, to, subject, body="", email_id=""):
    if action == "new":
        message = MIMEText(body)
        message['to'] = to
        message['from'] = user_id
        message['subject'] = subject
        
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        draft_body = {'message': {'raw': encoded_message}}
        draft = service.users().drafts().create(
            userId=user_id, 
            body=draft_body).execute()
        print(f"Draft created with ID: {draft['id']}")
        return draft['id']
    elif action == "reply":
        return create_reply_draft(email_id, body)
    elif action == "forward":
        return create_forward_draft(email_id, to, body)
    else:
        return

def create_reply_draft(message_id, reply_text):
    # Fetch the original message
    original_message = service.users().messages().get(
        userId=user_id,
        id=message_id,
        format='full').execute()

    # Get headers
    headers = original_message['payload']['headers']
    subject = None
    to = None
    for header in headers:
        if header['name'] == 'Subject':
            subject = header['value']
        if header['name'] == 'From':
            to = header['value']

    # Create the reply subject
    if not subject.startswith("Re: "):
        subject = "Re: " + subject

    # Compose the reply message
    reply_message = MIMEText(reply_text)
    reply_message['to'] = to
    reply_message['from'] = user_id
    reply_message['subject'] = subject
    reply_message['In-Reply-To'] = message_id

    encoded_message = base64.urlsafe_b64encode(reply_message.as_bytes()).decode()
    draft_body = {'message': {'raw': encoded_message, 'threadId': original_message['threadId']}}
    draft = service.users().drafts().create(userId=user_id, body=draft_body).execute()
    return draft['id']

def create_forward_draft(message_id, recipient_email, custom_message=None):
    # Get the original message
    original_message = service.users().messages().get(
        userId=user_id,
        id=message_id,
        format='raw').execute()

    # Decode the raw message
    raw_message = base64.urlsafe_b64decode(original_message['raw'].encode('utf-8'))

    # Prepare the forward header and optional custom message
    forward_header = f"----- Forwarded message -----\nFrom: {recipient_email}\n\n"
    if custom_message:
        forward_header += f"{custom_message}\n\n"

    # Combine the forward header with the original message
    new_message = forward_header + raw_message.decode('utf-8')

    # Encode the combined message into base64 format
    encoded_message = base64.urlsafe_b64encode(new_message.encode('utf-8')).decode('utf-8')

    draft_body = {'message': {'raw': encoded_message, 'threadId': original_message['threadId']}}
    draft = service.users().drafts().create(userId=user_id, body=draft_body).execute()
    print(f"Forward draft created with ID: {draft['id']}")
    return draft['id']

def send_draft(id):
    sent_message = service.users().drafts().send(
        userId=user_id, 
        body={'id': memory['draft_id']}
        ).execute()
    return f"Draft sent with email ID: {sent_message['id']}"

def get_pdf_summary(file_name):
    text = pdf2text(file_name)
    print("Calling Llama to generate a summary...")
    response = llama31(text, "Generate a summary of the input text in 5 sentences.")
    return response

def get_email_attachments(message_id, mime_type='application/pdf'):
    attachments = []

    # Helper function to process email parts
    def process_parts(parts):
        for part in parts:
            if part['mimeType'] in ['multipart/related', 'multipart/mixed', 'multipart/alternative']:
                # Recursively process nested parts
                if 'parts' in part:
                    process_parts(part['parts'])
            elif 'filename' in part and part['filename']:
                if part['mimeType'] == mime_type:  # Check for the desired MIME type
                    attachment_id = part['body'].get('attachmentId')
                    if attachment_id:
                        # Get the attachment data
                        attachment = service.users().messages().attachments().get(
                            userId=user_id, 
                            messageId=message_id, 
                            id=attachment_id
                        ).execute()
                        
                        # Decode the attachment content
                        file_data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))

                        with open(part['filename'], "wb") as f:
                            f.write(file_data)
                        
                        # Save the attachment information
                        attachments.append(
                            {'filename': part['filename'], 
                            'data': file_data,
                            'size': attachment.get('size', 0)
                            })

    # Retrieve the email message
    message = service.users().messages().get(
        userId=user_id,
        id=message_id,
        format='full').execute()
    payload = message['payload']

    # Start processing the parts
    if 'parts' in payload:
        process_parts(payload['parts'])
    
    rslt = ""
    for a in attachments:        
        rslt += f"{a['filename']} - {a['size']} bytes\n"
    return rslt #attachments

def pdf2text(file):
    text = ''
    try:
        with Path(file).open("rb") as f:
            reader = PdfReader(f)
            text = "\n\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        raise f"Error reading the PDF file: {str(e)}"

    print(f"\nPDF text length: {len(text)}\n")

    return text

def set_email_service(gmail):
    global user_email
    global service

    user_email = gmail
    service = authenticate_gmail(user_email)

def llama31(user_prompt: str, system_prompt = ""):
    response = ollama.chat(model='llama3.1',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response['message']['content']
