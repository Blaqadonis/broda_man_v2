# -------------------------------
# Imports and Initial Setup
# -------------------------------

import os
from uuid import uuid4
from typing import List, Union, Literal, Dict, Any
from pydantic import BaseModel, field_validator
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# -------------------------------
# Part 1: State and Utility Classes/Functions
# -------------------------------

from dotenv import load_dotenv
import os

# Load environment variables 
load_dotenv()

# Access the environment variables
api_key = os.getenv('GROQ_API_KEY')

class State(MessagesState):
    income: float
    budget_for_expenses: float
    expense: float
    expenses: List[Dict[str, Any]]  
    savings_goal: float
    savings: float
    summary: str
    currency: str

def show_assistant_output(message: str, flush: bool = False):
    """Display assistant's output."""
    print(message, flush=flush)

def get_user_input() -> str:
    """Capture user input."""
    return input("Your response: ")

# Initialize ChatGroq
GPT = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)

class QueryRouting(BaseModel):
    next_node: Literal['expert_registrar', 'budget_registrar', 'expense_registrar', 'irrelevant', 'END', 'intro']

GPT_ROUTER = GPT.with_structured_output(QueryRouting)

def assistant(state: dict) -> str:
    """
    Routes the user query to the appropriate registrar or handles session control.

    Args:
        state (dict): The application's current state, including messages and session information.

    Returns:
        str: The next node to route to or a session control indicator ('END').
    """
    messages = state.get("messages")

    query_prompt = (
        """
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses,
        and provide expert financial commentary on their financial status. Analyze the user's query and respond based on the context:
          * 'budget_registrar': If the query relates to budget allocation.
          * 'expense_registrar': If the query relates to expense tracking or logging.
          * 'expert_registrar': If the query seeks financial advice, recommendations, or insights about the user's financial status.
          * 'END': If the user wants to end or leave the session.
          * 'intro': If the query is a greeting or general inquiry about Broda Man's role.
          * 'irrelevant': For any other type of query not related to Broda Man's responsibilities.
        """
    )

    # Add the system message for the model
    messages.append(SystemMessage(content=query_prompt))

    # Invoke the model
    response = GPT_ROUTER.invoke(messages)

    next_node = response.next_node

    # Append the model's response to the conversation history
    messages.append(AIMessage(content=response.next_node))

    # Return the session control command
    return next_node

def intro(state: State) -> State:
    """
    Introduces Broda Man to the user and identifies their needs.
    """
    messages = state.get("messages")
    user_query = messages[0].content

    introduction_message = (
        f"""
        You are Broda Man, a personal financial assistant designed to help a user manage their budget, log expenses,
        and provide expert financial commentary on their financial status.

        Now, respond to user query following these instructions:
            * Respond politely to greetings if appropriate.
            * Briefly introduce yourself and explain to the user what you do.
            * If this is a new session, emphasize the importance of creating a budget first before proceeding with other tasks.
              Prompt the user to share their income details to set up a budget.
            * If it’s not a new session, ask the user how you can assist them.
        Keep responses concise, professional, and focused. Do not be chatty.

        User query: {user_query}
        """
    )
    messages.append(SystemMessage(content=introduction_message))
    response = GPT.invoke(introduction_message).content
    messages.append(AIMessage(content=response))

    # Display introduction
    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m", flush=True)

    # Capture user's response
    user_response = get_user_input()
    messages.append(HumanMessage(content=user_response))

    return state

class QueryResponse(BaseModel):
    next_node: Literal['expert_registrar', 'budget_registrar', 'expense_registrar', 'irrelevant', 'END']

GPT_EVALUATOR = GPT.with_structured_output(QueryResponse)

def evaluator(state: dict) -> str:
    """
    Evaluates the user's response to intro.

    Args:
        state (dict): The current application state, including messages and session information.

    Returns:
        str: The appropriate next node to route the user to or session control indicator ('END').
    """
    messages = state.get("messages")
    user_response = messages[-1].content
    last_ai_message = messages[-2].content

    # Define the query prompt for the evaluation
    query_prompt = (
        f"""
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses,
        and provide expert financial commentary on their financial status. Analyze the user's query and respond based on the context:
          * 'budget_registrar': If the query relates to budget allocation.
          * 'expense_registrar': If the query relates to expense tracking or logging.
          * 'expert_registrar': If the query seeks financial advice, recommendations, or insights about the user's financial status.
          * 'END': If the user wants to end or leave the session.
          * 'irrelevant': For any other type of query not related to Broda Man's responsibilities.
         AI system message:{last_ai_message}
         User response:{user_response}
        """
    )

    # Add the system message for the model
    messages.append(SystemMessage(content=query_prompt))

    # Invoke the model
    response = GPT_EVALUATOR.invoke(messages)

    next_node = response.next_node

    # Return the session control command
    return next_node

class BudgetStateUpdate(BaseModel):
    income: float
    savings_goal: float
    currency: str


BUDGET_STATE_UPDATE = GPT.with_structured_output(BudgetStateUpdate)

def budget_registrar(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles the budget registry process, prompting the user for income, savings goal,
    and currency details. Updates the state accordingly.

    Parameters:
        state (dict): The state containing conversation context and user data.

    Returns:
        dict: Updated state with user's income and savings goal.
    """
    messages = state.get("messages")
    user_income = state.get("income", 0.0)
    user_savings_goal = state.get("savings_goal", 0.0)
    user_expense = state.get("expense", 0.0)
    budget_for_expenses = state.get("budget_for_expenses", 0.0)
    user_currency = state.get("currency", "")
    summary = state.get("summary", "")

    # Generate a system message guiding the assistant's response
    budget_message = (
        f"""
          Broda Man is a personal financial assistant engaged in an ongoing conversation with the user. The user intends to 
          continue the session by allocating their budget.
          To proceed, the following details are required: Income amount, Savings goal, and Currency.
          Existing Data (if available):
            * Income: {user_income}
            * Savings goal: {user_savings_goal}
            * Expense: {user_expense}
            * Expense budget: {budget_for_expenses}
            * Currency: {user_currency}
            * Conversation summary: {summary}
          Instructions:
            1. Review Conversation History:
               Analyze the conversation to determine which required details (Income, Savings goal, Currency) are missing.
            2. Request Missing Details:
               If any detail is missing, guide the user to provide it in a friendly and supportive manner. Be dynamic and slightly 
               conversational, yet focused on Broda Man’s responsibilities.
               Example: "Could you let me know your savings goal? This will help me tailor your budget effectively."
            3. Display Data Only When Requested:
               Only display the user’s data if explicitly requested, and ensure it is formatted correctly and user-friendly.
               Example: "Your budget for expense is 5,000 {user_currency}"
            4. Confirm Existing Data:
               If all details are available, present the user's data to them in the following way (Do not forget the instructions
               for displaying data in step 3 above):
                  Last logged income: <display user_income in the format described in step 3 above which should of course
                  include the currency symbol>
                  Last logged savings target: <display user_savings_goal in the format described in step 3 above>
                  Then ask: "Does this look correct, or would you like to update any of these details?"
            5. Stay Concise and Relevant:
               Provide direct responses without unnecessary elaboration or conversational fillers. Ensure your tone encourages 
               participation while staying aligned with Broda Man's responsibilities.
          """
    )
    messages.append(SystemMessage(content=budget_message))

    # Invoke GPT to provide the assistant's response
    registrar_response = GPT.invoke(messages).content
    messages.append(AIMessage(content=registrar_response))

    # Display the assistant's response
    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m", flush=True)

    # Capture user's response and append to messages
    user_response = get_user_input()
    messages.append(HumanMessage(content=user_response))

    # Request structured output from GPT to extract income and savings goal
    state_update_prompt = "Extract the user's income and their savings goal from the conversation."
    messages.append(HumanMessage(content=state_update_prompt))
    response = BUDGET_STATE_UPDATE.invoke(messages)

    # Update the state with extracted values
    state["income"] = response.income
    state["savings_goal"] = response.savings_goal
    state["currency"] = response.currency

    return state

class BudgetRegistryEvaluation(BaseModel):
    next_node: Literal['budget_allocation_node', 'budget_registrar', 'END']

GPT_BUDGET_EVAL = GPT.with_structured_output(BudgetRegistryEvaluation)

def budget_registry_eval(state: dict) -> str:
    """
    Evaluates user's response to the budget registrar.

    Args:
        state (dict): Current state containing the conversation history.

    Returns:
        str: The next node determined by the user's response.
    """
    messages = state.get("messages")

    # Construct the system prompt
    prompt = (
        f"""
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses,
        and provide expert financial commentary on their financial status. Analyze the user's query and respond appropriately:
          * 'budget_allocation_node': If all the requirements are met and up-to-date(income, savings_goal, and currency).
          * 'budget_registrar': If some of the requirements are met but not all.
          * 'END': If the user wants to end or leave the session.
        """
    )
    messages.append(SystemMessage(content=prompt))

    # Invoke the model
    response = GPT_BUDGET_EVAL.invoke(messages)

    next_node = response.next_node

    # Extract and return the structured output
    return response.next_node

# -------------------------------
# Part 2: Expense Registrar and Expert Registrar
# -------------------------------

class ExpenseStateUpdate(BaseModel):
    expenses: List[float]
    currency: str

EXPENSE_STATE_UPDATE = GPT.with_structured_output(ExpenseStateUpdate)

def expense_registrar(state: State) -> State:
    """
    Handles the expense registry process, prompting the user for expense amounts and currency details.
    Updates the state accordingly.

    Parameters:
        state (State): The current state containing conversation context and user data.

    Returns:
        State: Updated state with the user's total expenses and currency.
    """
    messages = state.get("messages")
    user_income = state.get("income", 0.0)
    user_savings_goal = state.get("savings_goal", 0.0)
    user_expense = state.get("expense", 0.0)
    budget_for_expenses = state.get("budget_for_expenses", 0.0)
    user_currency = state.get("currency", "")
    summary = state.get("summary", "")

    # Define the system message guiding the assistant's response
    expense_message = f"""
          Broda Man is a personal financial assistant engaged in an ongoing conversation with the user. The user intends to 
          continue the session logging their expenses.
          To proceed, they must provide these expense amounts, and Currency.
          Existing Data (if available):
            * Income: {user_income}
            * Savings goal: {user_savings_goal}
            * Expense: {user_expense}
            * Expense budget: {budget_for_expenses}
            * Currency: {user_currency}
            * Conversation summary: {summary}
          Instructions:
            1. Review Conversation History:
               Analyze the conversation to determine which required details (expense amounts, and Currency) are missing.
            2. Request Missing Details:
               If any detail is missing, guide the user to provide it in a friendly and supportive manner. Be dynamic and slightly
               conversational, yet focused on Broda Man’s responsibilities.
               Example: "Could you let me know your savings goal? This will help me tailor your budget effectively."
            3. Display Data Only When Requested:
               Only display the user’s data if explicitly requested, and ensure it is formatted correctly and user-friendly.
                   For Example: "Your total savings is 3,000 {user_currency}" 
            4. Confirm Existing Data:
               If all details are available, present the user's data to them in the following way (Do not forget the instructions for
               displaying data in step 3 above):
                  * Last logged total expense: <display total_user_expenses in the format described in step 3 above which should of course
                  include the currency symbol>
                  Then ask: "Does this look correct, or would you like to update any of these details?"
            5. Stay Concise and Relevant:
               Provide direct responses without unnecessary elaboration or conversational fillers. Ensure your tone encourages participation 
               while staying aligned with Broda Man's responsibilities.
          """

    # Add the system message to the conversation
    messages.append(SystemMessage(content=expense_message))

    # Invoke GPT to generate the assistant's response
    registrar_response = GPT.invoke(messages).content
    messages.append(AIMessage(content=registrar_response))

    # Display the assistant's response
    show_assistant_output(f"\033[92m{registrar_response}\033[0m", flush=True)

    # Capture the user's response and append to the conversation
    user_response = get_user_input()
    messages.append(HumanMessage(content=user_response))

    # Use structured output to validate and extract user-provided expense data
    response = EXPENSE_STATE_UPDATE.invoke(messages)

    # Update the state with the validated expenses and currency
    state["expenses"] = response.expenses
    state["currency"] = response.currency

    return state

class ExpenseResponse(BaseModel):
    """
    A structured response model for expense evaluation.
    """
    next_node: Literal['expense_logging_node', 'expense_registrar', 'END']

# Initialize the GPT model with structured output
GPT_EXPENSE_EVAL = GPT.with_structured_output(ExpenseResponse)

def expense_registry_eval(state: dict) -> str:
    """
    Evaluates the user's response in the expense registrar and determines the next workflow node.

    Args:
        state (dict): The current application state, including messages.

    Returns:
        str: The determined next node (e.g., 'expense_logging_node', 'expense_registrar', 'END').
    """
    messages = state.get("messages")

    # Construct the evaluation prompt to guide the assistant
    prompt = (
        f"""
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses,
        and provide expert financial commentary on their financial status. Analyze the user's query and respond appropriately:
          * 'expense_logging_node': If all the requirements for logging expenses are met(expenses, and currency).
          * 'expense_registrar': If the user has partially provided expense-related information.
          * 'END': If the user wishes to end or leave the session.
        """
    )

    # Add the system message for evaluation to the conversation
    messages.append(SystemMessage(content=prompt))

    # Invoke the model
    response = GPT_EXPENSE_EVAL.invoke(messages)

    return response.next_node

class ExpertStateUpdate(BaseModel):
    income: float 
    expenses: float
    savings_goal: float
    currency: str

EXPERT_STATE_UPDATE = GPT.with_structured_output(ExpertStateUpdate)

def expert_registrar(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles the financial expert registry process, prompting the user for income, expenses,
    savings goals, and currency. Updates the state accordingly.

    Parameters:
        state (dict): The state containing conversation context and user data.

    Returns:
        dict: Updated state with the user's financial details.
    """
    # Extract existing data from the state
    messages = state.get("messages")
    user_expenses = state.get("expense", 0.0)
    user_expense_budget = state.get("budget_for_expenses", 0.0)
    user_income = state.get("income", 0.0)
    user_savings_goal = state.get("savings_goal", 0.0)
    user_savings = state.get("savings", 0.0)
    user_currency = state.get("currency", "")
    summary = state.get("summary", "")

    # Define the system message guiding the assistant's response
    expert_message = f"""
          Broda Man is a personal financial assistant engaged in an ongoing conversation with the user. 
          Analyze the conversation to identify which of the required details
          (Income, expense amounts, Savings target, and Currency) are missing.
          Existing Data (if available):
                    * User's total expense: {user_expenses}
                    * User's income: {user_income}
                    * User's expense budget (optional): {user_expense_budget}
                    * User's savings target: {user_savings_goal}
                    * User's savings (optional): {user_savings}
                    * User's currency: {user_currency}
                    * Conversation summary: {summary}
          Instructions:
            1. Review Conversation History:
               Analyze the conversation to determine which required details (Income, expense amounts, Savings target,
               and Currency) are missing.
            2. Request Missing Details:
               If any required detail is missing (Income, expense amounts, Savings target, and Currency), guide the user
               to provide it in a friendly and supportive manner.
               Be dynamic and slightly conversational, yet focused on Broda Man’s responsibilities.
               Example: "Could you let me know your savings goal? This will help me tailor your budget effectively."
            3. Display Data Only When Requested:
               Only display the user’s data if explicitly requested, and ensure it is formatted correctly and user-friendly.
               Example: "Your total savings is 210,000 {user_currency}"
            4. Confirm Existing Data:
               If all details are available, present the user's data to them in the following way (Do not forget the 
               instructions for displaying data in step 3 above):
                    * Last logged total expense: <display user_expenses in the format described in step 3 above which should of course
                  include the currency symbol>
                    * Last logged income: <display user_income in the format described in step 3 above which should of course
                  include the currency symbol>
                    * Last logged expense budget: <display user_expense_budget in the format described in step 3 above which should of course
                  include the currency symbol>
                    * Last logged savings target: <display user_savings_goal in the format described in step 3 above which should of course
                  include the currency symbol>
                    * Last logged savings: <display user_savings in the format described in step 3 above which should of course
                  include the currency symbol>
                  Then ask: "Does this look correct, or would you like to update any of these details?"
            5. Stay Concise and Relevant:
               Provide direct responses without unnecessary elaboration or conversational fillers. Ensure your tone encourages
               participation while staying aligned with Broda Man's responsibilities.
          """
    # Add the system message to the conversation
    messages.append(SystemMessage(content=expert_message))

    # Invoke GPT to generate the assistant's response
    registrar_response = GPT.invoke(messages).content
    messages.append(AIMessage(content=registrar_response))

    # Display the assistant's response
    show_assistant_output(f"\033[92m{registrar_response}\033[0m", flush=True)

    # Capture user's response and append to the conversation
    user_response = get_user_input()
    messages.append(HumanMessage(content=user_response))

    # Use structured output to extract user-provided financial data
    response = EXPERT_STATE_UPDATE.invoke(messages)

    # Update the state with the extracted data
    state["income"] = response.income 
    state["expenses"] = response.expenses 
    state["savings_goal"] = response.savings_goal 
    state["currency"] = response.currency 

    return state

class ExpertResponse(BaseModel):
    next_node: Literal['ask_expert', 'expert_registrar', 'END']

# Configure GPT with Pydantic validation
GPT_EXPERT_EVAL = GPT.with_structured_output(ExpertResponse)

def expert_registry_eval(state: dict) -> str:
    """
    Evaluates the user's response to the expert registrar.

    Args:
        state (dict): The current application state, including messages.

    Returns:
        str: The determined next node (e.g., 'ask_expert', 'expert_registrar', 'END', or 'irrelevant').
    """
    messages = state.get("messages")

    # Construct the evaluation prompt
    prompt = (
        f"""
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses,
        and provide expert financial commentary on their financial status. Analyze the user's query and respond appropriately:
          * 'ask_expert': If all the requirements are met and up-to-date(income, expenses, savings_goal, and currency).
          * 'expert_registrar': If some of the requirements are met but not all.
          * 'END': If the user wants to end or leave the session.
        User response:{messages[-1].content}
        Broda Man's last message:{messages[-2].content}
        """
    )

    # Add the system message for the model
    messages.append(SystemMessage(content=prompt))

    # Invoke the model with structured output validation
    response = GPT_EXPERT_EVAL.invoke(messages)

    next_node = response.next_node

    # Return the next node
    return next_node

# -------------------------------
# Part 2: Budget Allocation and Logging
# -------------------------------

from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional, Union
import json

class BudgetAllocation(BaseModel):
    income: float
    savings_goal: Union[float, str]  # Savings goal can be a float, percentage, ratio, or amount

MODEL = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)

ALLOCATION = MODEL.with_structured_output(BudgetAllocation)

def parse_savings_goal(savings_goal: str, income: float) -> float:
    """
    Parse the savings goal string into a float value.

    Args:
        savings_goal (str): The savings goal as a percentage, ratio, or amount.
        income (float): The user's income, used for percentage or ratio-based goals.

    Returns:
        float: The parsed savings amount.
    """
    if isinstance(savings_goal, (float, int)):
        return float(savings_goal)

    try:
        # Handle percentage savings goal
        if "%" in savings_goal:
            percentage = float(savings_goal.strip("%")) / 100
            return income * percentage

        # Handle ratio-based savings goal (e.g., "1:3")
        if ":" in savings_goal:
            ratio = savings_goal.split(":")
            savings_ratio = float(ratio[0]) / (float(ratio[0]) + float(ratio[1]))
            return income * savings_ratio

        # Handle direct amount as string
        return float(savings_goal)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse savings goal: {savings_goal}, Error: {e}")

def calculate_budget(income: float, savings_goal: Union[float, str]) -> tuple:
    """
    Calculate savings and expenses based on income and savings goal.

    Args:
        income (float): The user's income.
        savings_goal (Union[float, str]): The user's savings goal.

    Returns:
        tuple: A tuple containing (savings, expenses).
    """
    parsed_savings_goal = parse_savings_goal(savings_goal, income)

    if parsed_savings_goal > 0.3 * income:
        # Adjust savings and expenses proportionately if goal is too high
        savings = 0.3 * income
        expenses = income - savings
    else:
        # Use the specified savings goal
        savings = parsed_savings_goal
        expenses = income - savings

    return savings, expenses

def budget_allocation_node(state: dict) -> dict:
    """
    Help the user allocate a budget based on their income and savings goal.

    Args:
        state (dict): The current state of the app.

    Returns:
        dict: Updated state with budget allocation.
    """
    messages = state["messages"]
    user_income = state.get("income")
    savings_goal = state.get("savings_goal")
    user_response = messages[-1].content

    # Calculate savings and expenses
    savings, expenses = calculate_budget(user_income, savings_goal)

    # Update the state
    state["income"] = user_income
    state["savings_goal"] = savings_goal
    state["savings"] = savings
    state["budget_for_expenses"] = expenses

    # Log success message
    messages.append(AIMessage(content="BUDGET ALLOCATION DONE. STATE UPDATED."))
    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m", flush=True)

    return state

def calculate_total_expense(expenses: List[float]) -> float:
    """
    Calculate the total expenses from a list of numeric amounts.

    Args:
        expenses (List[float]): A list of numeric expense amounts.

    Returns:
        float: Total expense amount.
    """
    return sum(expenses)

def expense_logging_node(state: dict) -> dict:
    """
    Calculate the total expense based on logged expenses in the state.

    Args:
        state (dict): The current state of the app.

    Returns:
        dict: Updated state with the total expense.
    """
    # Retrieve the expenses list from the state
    expenses = state.get("expenses", [])
    messages = state.get("messages")

    # Calculate the total of all expenses directly
    total_expense = calculate_total_expense(expenses)

    # Update the state with the total expense
    state["expense"] = total_expense

    # Append a success message to the conversation history
    messages.append(AIMessage(content="Expense logged successfully!"))
    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m", flush=True)

    return state

# -------------------------------
# Part 3: Continue and Irrelevant Handlers
# -------------------------------

from enum import Enum

class Existence(Enum):
    EXIST = "EXIST"
    DOES_NOT_EXIST = "DOES NOT EXIST"

class MsgCheck(BaseModel):
    query: Existence

MSG_MODEL = GPT.with_structured_output(MsgCheck)

def ask_expert(state: State) -> State:
    messages = state.get("messages")

    # Define the prompt for analyzing the user's query
    prompt = f"""
    Broda Man is a personal financial assistant engaged in an ongoing conversation with the user. The user appears interested in querying the financial expert.
    Analyze the user's query and respond with one of the following:
      * 'EXIST' if the user has already made a query for the expert.
      * 'DOES NOT EXIST' if the user has expressed interest but has not yet asked a query.
    Message history: {messages}
    """

    # Invoke the model to check for query existence
    response = MSG_MODEL.invoke(messages)
    query_status = response.query

    if query_status == Existence.DOES_NOT_EXIST:
        # Prompt the user to make a query
        assistant_prompt = "What would you like to know?"
        messages.append(SystemMessage(content=assistant_prompt))

        # Display assistant's output
        show_assistant_output(f"\033[92m{assistant_prompt}\033[0m", flush=True)

        # Capture the user's response
        user_response = get_user_input()
        messages.append(HumanMessage(content=user_response))
        state["messages"] = messages

    return state

def financial_expert_advice(state: State) -> State:
    messages = state.get("messages")
    user_response = messages[-1].content
    last_ai_message = messages[-2].content
    user_income = state.get("income")
    user_savings_goal = state.get("savings_goal")
    user_expenses = state.get("expense")
    user_currency = state.get("currency")

    # Calculate savings and expense budget
    user_savings, user_expense_budget = calculate_budget(user_income, user_savings_goal)

    # Update the state with calculated values
    state["budget_for_expenses"] = user_expense_budget
    state["savings"] = user_savings


    # Add system message for financial advice
    prompt = (
        f"""
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses, and provide expert
        financial commentary on their financial status.
        
        Responding to User Queries:
        Use the following user details to craft your response:
        
        Last Broda Man message: {last_ai_message}
        User's response: {user_response}
        User's income: {user_income} {user_currency}
        User's expenses: {user_expenses} {user_currency}
        User's expense budget: {user_expense_budget} {user_currency}
        User's savings target: {user_savings_goal} {user_currency}
        User's savings: {user_savings} {user_currency}
        
        Instructions for Your Response:
        1. Be Brief and Precise: Your response should be concise, direct, and focused on addressing the user's query.
        2. Avoid Chattiness: Refrain from unnecessary elaboration or conversational fillers.
        3. No Calculations or Assumptions: Just answer the user query based on provided information. Do not compute, only deduce.
        4. Contextual Relevance: Base your response strictly on the provided user details and the context of their query.
        5. Missing Information: If any required detail for crafting an accurate response is missing, clearly state: 
           "I don't have enough information to answer that accurately based on the provided details."
        """
    )

    # Generate response for financial advice
    messages.append(SystemMessage(content=prompt))
    response = GPT.invoke(prompt).content
    messages.append(AIMessage(content=response))
    show_assistant_output(response, flush=True)

    return state

class IrrelevantResponse(BaseModel):
    next_node: Literal['budget_registrar', 'expense_registrar', 'expert_registrar', 'END']

EXIRRELEVANT = GPT.with_structured_output(IrrelevantResponse)

def irrelevant_eval(state: dict) -> str:
    """
    Evaluates user's response to system-generated irrelevant message.
    """
    messages = state.get("messages")
    user_response = messages[-1].content
    last_ai_message = messages[-2].content

    # Construct the query prompt
    query_prompt = (
        f"""
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses,
        and provide expert financial commentary on their financial status. Analyze the user's query and respond appropriately:
          * 'budget_registrar': If the query relates to budget allocation.
          * 'expense_registrar': If the query relates to expense tracking or logging.
          * 'expert_registrar': If the query seeks financial advice, recommendations, or insights about the user's financial status.
          * Otherwise, return 'END'.
        User response:{user_response}
        Broda Man's last message:{last_ai_message}
        """
    )

    # Add the system message to the conversation
    messages.append(SystemMessage(content=query_prompt))

    # Invoke the model and get the response
    response = EXIRRELEVANT.invoke(messages)

    next_node = response.next_node

    # Append the validated response to the messages
    messages.append(AIMessage(content=next_node))

    # Return the validated action
    return next_node

def Continue(state: State) -> State:
    """
    Determine if user wants to continue the session or end it.
    """
    messages = state.get("messages")

    introduction_message = (
        """Ask the user:
        'Do you want to continue the session or end it?'"""
    )
    messages.append(SystemMessage(content=introduction_message))
    response = GPT.invoke(messages).content
    messages.append(AIMessage(content=response))

    # Display introduction
    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m", flush=True)

    # Capture user's response
    user_response = get_user_input()
    messages.append(HumanMessage(content=user_response))
    return state

class EvaluationResponse(BaseModel):
    next_node: Literal['budget_registrar', 'expense_registrar', 'expert_registrar', 'END', 'irrelevant']

CONTINUE_EVAL = GPT.with_structured_output(EvaluationResponse)

def continue_eval(state: dict) -> str:
    """
    Evaluates user's response to Continue
    """
    messages = state.get("messages")

    # Construct the prompt for the model
    prompt = (
        f"""
            Broda Man is a personal financial assistant designed to help users with the following tasks:
              * Manage their budget by tracking income, expenses, and savings.
              * Log expenses accurately and consistently.
              * Provide expert financial advice, insights, and recommendations based on the user's financial situation.
            
            Task:
              * Analyze the user's response and determine the appropriate next action based on its context:
                - Return 'budget_registrar' if the response indicates the user wants to **create, modify, or allocate a budget**.
                  Example: "I want to perform budget allocation" or "Let’s create a budget for my expenses."
                - Return 'expense_registrar' if the response indicates the user wants to **log new expenses**.
                  Example: "I need to log an expense for lunch." 
                - Return 'expert_registrar' if the response seeks **financial advice, insights, or commentary** on the user’s financial 
                  status or future planning.
                  Example: "How much is my limit for budget?" or "How do I make sure I don’t overspend?" or 
                  "What are my total expenses so far?"
                - Return 'END' if the user expresses a desire to **cancel, end, or leave the session**.
                  Example: "I’m done for now, goodbye." or "End session."
                - Return 'irrelevant' for any query that falls **outside the scope** of Broda Man’s responsibilities.
                  Example: "What’s the weather like today?" or "Can you tell me a joke?"
            
            Note:
              * Provide only the relevant response based on the user's input. Do not add additional commentary or explanations.
        """
    )

    # Add the system message for the model
    messages.append(SystemMessage(content=prompt))

    # Invoke the model and get the response
    response = CONTINUE_EVAL.invoke(messages)

    next_node = response.next_node

    # Append the model response to the messages
    messages.append(AIMessage(content=next_node))

    # Return the validated action
    return next_node

def irrelevant(state: State) -> State:
    """
    Captures user's response to system-generated irrelevant message
    """
    messages = state.get("messages")

    messages += [
        AIMessage(
            """
            I am Broda Man, your AI personal financial assistant. I can help you with the following:
              * Tracking your expenses
              * Allocating your budget
              * Providing expert financial advice
            Which one would you like to do?
            """
        )
    ]
    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m", flush=True)

    # Capture user's response
    user_response = get_user_input()
    messages.append(HumanMessage(content=user_response))

    return state

class IrrelevantResponse(BaseModel):
    next_node: Literal['budget_registrar', 'expense_registrar', 'expert_registrar', 'END']

IRRELEVANT = GPT.with_structured_output(IrrelevantResponse)

def irrelevant_eval(state: dict) -> str:
    """
    Evaluates user's response to system-generated irrelevant message.
    """
    messages = state.get("messages")
    user_response = messages[-1].content
    last_ai_message = messages[-2].content

    # Construct the query prompt
    query_prompt = (
        f"""
        Broda Man is a personal financial assistant designed to help users manage their budget, log expenses,
        and provide expert financial commentary on their financial status. Analyze the user's query and respond appropriately:
          * 'budget_registrar': If the query relates to budget allocation.
          * 'expense_registrar': If the query relates to expense tracking or logging.
          * 'expert_registrar': If the query seeks financial advice, recommendations, or insights about the user's financial status.
          * Otherwise, return 'END'.
        User response:{user_response}
        Broda Man's last message:{last_ai_message}
        """
    )

    # Add the system message to the conversation
    messages.append(SystemMessage(content=query_prompt))

    # Invoke the model and get the response
    response = IRRELEVANT.invoke(query_prompt)

    next_node = response.next_node

    # Append the validated response to the messages
    messages.append(AIMessage(content=next_node))

    # Return the validated action
    return next_node

def BudgetFirst(state: State) -> State:
    """
    Ensures that the user creates a budget before proceeding with any other tasks.
    If the user's income is not provided, the assistant encourages them to prioritize budget creation.
    """
    messages = state.get("messages")
    income = state.get("income")

    if not income:
        msg = (
            "It looks like we need to set up your budget first before we can proceed with your request. "
            "Let me help you create a budget tailored to your financial goals and needs. "
            "Please share your income details to get started."
        )
        messages.append(AIMessage(content=msg))
        show_assistant_output(f"\033[92m{msg}\033[0m", flush=True)

    return state

# -------------------------------
# Part 3: Summary and Workflow Setup
# -------------------------------

class SummarizeConversation(BaseModel):
    summary: str

def summarize_conversation(state: State) -> State:
    """
    Summarizes the conversation if it exceeds a certain length.

    Args:
        state (State): The current state of the application.

    Returns:
        State: Updated state with the conversation summary.
    """
    messages = state.get("messages")
    summary = state.get("summary", "")

    # Create a summarization prompt
    if len(messages) > 6:
        if summary:
            # If there's an existing summary, extend it
            summary_message = (
                f"This is the summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages below: \n\n"
                f"messages: {messages}"
            )
        else:
            # No existing summary, create one
            summary_message = "Create a summary of this conversation:"

        # Initialize the model
        GPT = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)

        # Generate a summary response
        response = GPT.invoke(summary_message)

        # Remove all but the 3 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-3]]

        # Update the summary and messages
        state["summary"] = response.content
        state["messages"] = delete_messages

    # If there are 8 or fewer messages, don't summarize
    return state

# -------------------------------
# Part 3: Workflow Graph Setup
# -------------------------------

# Initialize the StateGraph
workflow = StateGraph(State)

# Define the nodes in the graph
workflow.add_node("intro", intro)
workflow.add_node("budget_allocation_node", budget_allocation_node)
workflow.add_node("expense_logging_node", expense_logging_node)
workflow.add_node("ask_expert", ask_expert)
workflow.add_node("financial_expert_advice", financial_expert_advice)
workflow.add_node("irrelevant", irrelevant)
workflow.add_node("budget_registrar", budget_registrar)
workflow.add_node("expense_registrar", expense_registrar)
workflow.add_node("expert_registrar", expert_registrar)
workflow.add_node("Continue", Continue)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("BudgetFirst", BudgetFirst)

# Define edges (connections between nodes)
#workflow.add_edge(START, "assistant")
workflow.set_conditional_entry_point(
    assistant,
    {
        "budget_registrar": "budget_registrar",
        "expense_registrar": "BudgetFirst",
        "expert_registrar": "BudgetFirst",
        "irrelevant": "irrelevant",
        "END": END,
        "intro": "intro"
    },
)

workflow.add_conditional_edges(
    "intro", evaluator,{
        "budget_registrar": "budget_registrar",
        "expense_registrar": "BudgetFirst",
        "expert_registrar": "BudgetFirst",
        "irrelevant": "irrelevant",
        "END": END})
workflow.add_edge("BudgetFirst", "budget_registrar")
workflow.add_conditional_edges(
    "budget_registrar", budget_registry_eval,{
        "budget_registrar": "budget_registrar",
        "budget_allocation_node": "budget_allocation_node",
        "END": END})
workflow.add_edge("budget_allocation_node", "summarize_conversation")
workflow.add_conditional_edges(
    "expense_registrar", expense_registry_eval,{
        "expense_registrar": "expense_registrar",
        "expense_logging_node": "expense_logging_node",
        "END": END})
workflow.add_edge("expense_logging_node", "summarize_conversation")
workflow.add_conditional_edges(
    "expert_registrar", expert_registry_eval,{
        "expert_registrar": "expert_registrar",
        "ask_expert": "ask_expert",
        "END": END})
workflow.add_edge("ask_expert", "financial_expert_advice")
workflow.add_edge("financial_expert_advice", "summarize_conversation")

workflow.add_edge("summarize_conversation", "Continue")
workflow.add_conditional_edges(
    "Continue", continue_eval,{
        "budget_registrar": "budget_registrar",
        "expense_registrar": "expense_registrar",
        "expert_registrar": "expert_registrar",
        "END": END,
        "irrelevant": "irrelevant"})
workflow.add_conditional_edges(
    "irrelevant", irrelevant_eval,{
        "budget_registrar": "budget_registrar",
        "expense_registrar": "expense_registrar",
        "expert_registrar": "expert_registrar",
        "END": END})

# Compile the workflow
checkpointer = MemorySaver()
broda_man_v2 = workflow.compile(checkpointer)

# Specify a thread (if needed)
#config = {"configurable": {"thread_id": "1"}}

# Start the conversation with user input
#messages = [HumanMessage(content=input("You: "))]
#result = broda_man_v2.invoke({"messages": messages}, config)

# Display the conversation result
#for message in result["messages"]:
    #print(f"Broda Man: {message.content}")