from langchain_openai import ChatOpenAI
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from bs4 import BeautifulSoup

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MOCK_TOOL_PROMPT = """
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""

INPUT_PROMPT = """
Tool Docstring: {description}
Input: {input}
Generate a mock output for this tool.
"""


def Web_Scraper_Parse_URL_and_Extract_Data(url: str):
    """
    A simple web scraper using BeautifulSoup4 to extract all text content from a given URL.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: All text content extracted from the webpage.
    """
    try:
        import requests
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        return f"Error scraping URL {url}: {e}"

def Contact_Finder_Search_and_Retrieve_Contact(company_name: str, senior_executive_title: Optional[str] = None):
    """
    Tool to search for and retrieve contact information, specifically email addresses or LinkedIn profiles, of senior executives within a company.

    Args:
        company_name (str): The name of the company to search for contacts.
        senior_executive_title (Optional[str]): The title of the senior executive to search for (e.g., "CEO", "CTO").

    Returns:
        str: A JSON string containing the contact information.

    Example:
        Contact_Finder_Search_and_Retrieve_Contact(company_name="Google", senior_executive_title="CEO")
        # Expected output:
        # {
        #   "name": "Sundar Pichai",
        #   "email": "sundar.pichai@google.com",
        #   "linkedin_profile": "https://www.linkedin.com/in/sundarpichai"
        # }
    """
    class ContactInfo(BaseModel):
        name: str = Field(description="The name of the contact.")
        email: Optional[str] = Field(description="The email address of the contact.")
        linkedin_profile: Optional[str] = Field(description="The LinkedIn profile URL of the contact.")
        title: Optional[str] = Field(description="The job title of the contact.")

    input_str = f"company_name: {company_name}, senior_executive_title: {senior_executive_title}"
    description = Contact_Finder_Search_and_Retrieve_Contact.__doc__

    result = llm.with_structured_output(ContactInfo).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def Email_Sender_Send_Email(to: str, subject: str, body: str, cc: Optional[List[str]] = None, bcc: Optional[List[str]] = None, attachments: Optional[List[str]] = None):
    """
    Tool to send a personalized email to the identified senior executive.

    Args:
        to (str): The recipient's email address.
        subject (str): The subject of the email.
        body (str): The main content of the email.
        cc (Optional[List[str]]): A list of email addresses to CC.
        bcc (Optional[List[str]]): A list of email addresses to BCC.
        attachments (Optional[List[str]]): A list of file paths for attachments.

    Returns:
        str: A JSON string indicating the email sending status.

    Example:
        Email_Sender_Send_Email(to="recipient@example.com", subject="Job Application", body="Dear Sir/Madam, ...")
        # Expected output:
        # {
        #   "status": "success",
        #   "message": "Email sent successfully."
        # }
    """
    class EmailStatus(BaseModel):
        status: str = Field(description="The status of the email sending operation (e.g., 'success', 'failed').")
        message: str = Field(description="A descriptive message about the email sending status.")

    input_str = f"to: {to}, subject: {subject}, body: {body}, cc: {cc}, bcc: {bcc}, attachments: {attachments}"
    description = Email_Sender_Send_Email.__doc__

    result = llm.with_structured_output(EmailStatus).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)