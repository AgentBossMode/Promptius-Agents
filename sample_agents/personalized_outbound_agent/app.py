from typing import List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command
import re
from sample_agents.personalized_outbound_agent.mock_tools import Web_Scraper_Parse_URL_and_Extract_Data, Contact_Finder_Search_and_Retrieve_Contact, Email_Sender_Send_Email


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    The GraphState represents the state of the LangGraph workflow.
    It extends MessagesState to include conversation history.
    """
    job_link: Optional[str] = None
    job_details: Optional[dict] = None
    contact_info: Optional[dict] = None
    product_prd: Optional[str] = None
    email_content: Optional[str] = None
    approved: Optional[bool] = None
    email_sent_status: Optional[bool] = None

web_scrape_job_details_tools = [Web_Scraper_Parse_URL_and_Extract_Data]
def web_scrape_job_details(state: GraphState) -> GraphState:
    """
    Node purpose: Extracts job details (title, pay, duration, skills, company) from a given job link.
    Implementation reasoning: Uses a pre-built react agent with the Web Scraper tool to perform web scraping.
    """
    class JobDetailsOutput(BaseModel):
        job_title: str = Field(description="The title of the job.")
        pay: Optional[str] = Field(description="The pay or salary range for the job.", default=None)
        duration: Optional[str] = Field(description="The duration of the job, if specified.", default=None)
        skills: List[str] = Field(description="A list of required skills for the job.")
        company_name: str = Field(description="The name of the company offering the job.")

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: JobDetailsOutput

    job_link = state.get("job_link")
    if not job_link:
        # Extract job link from the initial human message if not already set
        if state["messages"] and isinstance(state["messages"][0], HumanMessage):
            job_link = state["messages"][0].content
            state["job_link"] = job_link
        else:
            raise ValueError("Job link is missing from the state and initial message.")

    agent = create_react_agent(
      model=llm,
      prompt=f"You are an expert web scraper. Extract the job title, pay, duration, skills, and company name from the following job link: {job_link}. Use the 'Web_Scraper_Parse_URL_and_Extract_Data' tool. If a field is not found, return None for that field.",
      tools=web_scrape_job_details_tools,
      state_schema=CustomStateForReact,
      response_format=JobDetailsOutput
    )

    result: JobDetailsOutput = agent.invoke({"messages" :state["messages"]})["structured_response"]
    
    job_details = result.model_dump()
    
    return {
        "job_details": job_details,
        "messages": [AIMessage(content=f"Job details extracted for {job_details.get('job_title', 'N/A')} at {job_details.get('company_name', 'N/A')}.")]
    }

find_contact_information_tools = [Contact_Finder_Search_and_Retrieve_Contact]
def find_contact_information(state: GraphState) -> GraphState:
    """
    Node purpose: Identifies and retrieves contact information for senior executives within a company based on extracted job details.
    Implementation reasoning: Uses a pre-built react agent with the Contact Finder tool to search for executive contacts.
    """
    class ContactInfoOutput(BaseModel):
        name: str = Field(description="The name of the senior executive.")
        email: Optional[str] = Field(description="The email address of the senior executive.", default=None)
        linkedin_profile: Optional[str] = Field(description="The LinkedIn profile URL of the senior executive.", default=None)
        title: Optional[str] = Field(description="The title of the senior executive.", default=None)

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: ContactInfoOutput

    job_details = state["job_details"]
    company_name = job_details.get("company_name")
    if not company_name:
        raise ValueError("Company name is missing from job details.")

    agent = create_react_agent(
      model=llm,
      prompt=f"Find contact information (email or LinkedIn profile) for a senior executive at {company_name}. Prioritize roles like CEO, CTO, Head of Product, or similar. Use the 'Contact_Finder_Search_and_Retrieve_Contact' tool.",
      tools=find_contact_information_tools,
      state_schema=CustomStateForReact,
      response_format=ContactInfoOutput
    )

    result: ContactInfoOutput = agent.invoke(state)["structured_response"]
    
    contact_info = result.model_dump()

    return {
        "contact_info": contact_info,
        "messages": [AIMessage(content=f"Contact information found for {contact_info.get('name', 'N/A')} at {company_name}.")]
    }

def generate_email_content(state: GraphState) -> GraphState:
    """
    Node purpose: Generates personalized email content based on job details and product PRD.
    Implementation reasoning: Uses an LLM with structured output to ensure the email content is well-formatted and includes all necessary components.
    """
    class EmailContent(BaseModel):
        subject: str = Field(description="The subject line of the email.")
        body: str = Field(description="The main body of the email, personalized for the recipient and job.")
        call_to_action: str = Field(description="A clear call to action for the recipient.")

    structured_llm = llm.with_structured_output(EmailContent)

    job_details = state["job_details"]
    contact_info = state["contact_info"]
    product_prd = state["product_prd"]
    
    if not job_details or not contact_info or not product_prd:
        # For testing purposes, provide a mock product PRD if not available
        if not product_prd:
            product_prd = "Our product is an AI-powered coding assistant that helps developers write better code faster."
            state["product_prd"] = product_prd
        else:
            raise ValueError("Missing job details, contact info, or product PRD for email generation.")

    prompt = f"""
    Generate a personalized outreach email.
    Recipient Name: {contact_info.get('name', 'Sir/Madam')}
    Recipient Title: {contact_info.get('title', 'Senior Executive')}
    Company: {job_details.get('company_name', 'N/A')}
    Job Title: {job_details.get('job_title', 'N/A')}
    Job Skills: {', '.join(job_details.get('skills', []))}
    Product PRD: {product_prd}

    The email should:
    - Be concise and professional.
    - Reference the job title and company.
    - Briefly explain how our product (based on the PRD) can benefit their company or address a need related to the job.
    - Include a clear call to action.
    """
    
    email_output: EmailContent = structured_llm.invoke([
        HumanMessage(content=prompt),
        AIMessage(content=state["messages"][-1].content)
    ])
    
    full_email_content = f"Subject: {email_output.subject}\n\nDear {contact_info.get('name', 'Sir/Madam')},\n\n{email_output.body}\n\n{email_output.call_to_action}\n\nBest regards,\n[Your Name]"

    return {
        "email_content": full_email_content,
        "messages": [AIMessage(content="Email content generated and ready for human approval.")]
    }

def human_approval(state: GraphState) -> Command[Literal["send_email", "__end__"]]:
    """
    Node purpose: Presents the generated email content to the user for approval. This node acts as a human-in-the-loop step.
    Implementation reasoning: Uses `interrupt` to pause the graph and await human input for approval or rejection.
    """
    email_content = state["email_content"]
    if not email_content:
        raise ValueError("Email content is missing for human approval.")

    approval_decision = interrupt(
        {
            "prompt_for_human": "Please review the generated email content. Do you approve sending this email? (Type 'yes' to approve, 'no' to reject)",
            "email_to_review": email_content
        }
    )
    
    if approval_decision and approval_decision.lower() == "yes":
        return Command(goto="send_email", update={"approved": True, "messages": [AIMessage(content="Email approved by human.")]})
    else:
        return Command(goto="__end__", update={"approved": False, "messages": [AIMessage(content="Email rejected by human. Workflow terminated.")]})

send_email_tools = [Email_Sender_Send_Email]
def send_email(state: GraphState) -> GraphState:
    """
    Node purpose: Sends the personalized email to the identified senior executive after user approval.
    Implementation reasoning: Uses a pre-built react agent with the Email Sender tool to dispatch the email.
    """
    class EmailSendResult(BaseModel):
        success: bool = Field(description="True if the email was sent successfully, False otherwise.")
        message: str = Field(description="A message indicating the outcome of the email sending attempt.")

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: EmailSendResult

    email_content = state["email_content"]
    contact_info = state["contact_info"]
    
    if not email_content or not contact_info:
        raise ValueError("Missing email content or contact information for sending email.")

    recipient_email = contact_info.get("email")
    if not recipient_email:
        return {
            "email_sent_status": False,
            "messages": [AIMessage(content="Email not sent: Recipient email address not found.")]
        }

    # Extract subject and body from the full email content
    subject_match = re.match(r"Subject: (.*)\n\nDear", email_content, re.DOTALL)
    subject = subject_match.group(1).strip() if subject_match else "No Subject"
    body = email_content[email_content.find("Dear"):].strip()

    agent = create_react_agent(
      model=llm,
      prompt=f"Send an email to {recipient_email} with the subject '{subject}' and the following body: '{body}'. Use the 'Email_Sender_Send_Email' tool.",
      tools=send_email_tools,
      state_schema=CustomStateForReact,
      response_format=EmailSendResult
    )

    result: EmailSendResult = agent.invoke(state)["structured_response"]
    
    return {
        "email_sent_status": result.success,
        "messages": [AIMessage(content=f"Email sending status: {result.message}")]
    }

workflow = StateGraph(GraphState)

workflow.add_node("web_scrape_job_details", web_scrape_job_details)
workflow.add_node("find_contact_information", find_contact_information)
workflow.add_node("generate_email_content", generate_email_content)
workflow.add_node("human_approval", human_approval)
workflow.add_node("send_email", send_email)

workflow.add_edge(START, "web_scrape_job_details")
workflow.add_edge("web_scrape_job_details", "find_contact_information")
workflow.add_edge("find_contact_information", "generate_email_content")
workflow.add_edge("generate_email_content", "human_approval")

workflow.add_edge("send_email", END)

app = workflow.compile(
)
