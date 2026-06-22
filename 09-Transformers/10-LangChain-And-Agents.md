# 🤖 LangChain & Autonomous Agents

> **Prerequisites**: RAG & Vector Databases, GPT & Decoder Models | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Evolution from Text Generators to Agents](#1-the-evolution-from-text-generators-to-agents)
2. [Introduction to LangChain](#2-introduction-to-langchain)
3. [The ReAct Paradigm (Reason + Act)](#3-the-react-paradigm-reason--act)
4. [Equipping LLMs with Tools](#4-equipping-llms-with-tools)
5. [LangGraph: Stateful Multi-Agent Workflows](#5-langgraph-stateful-multi-agent-workflows)
6. [Implementation Example: A Math & Search Agent](#6-implementation-example-a-math--search-agent)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Evolution from Text Generators to Agents

Vanilla LLMs like GPT-4 are excellent at generating text, but they are isolated brains. They cannot browse the live internet, calculate math deterministically, or interact with APIs.

An **Autonomous Agent** is an LLM given the ability to:
1. **Reason** about a complex goal.
2. **Break it down** into sub-tasks.
3. **Use External Tools** (like a calculator, web browser, or SQL database) to gather information or take actions.
4. **Observe** the outcome of those actions and adjust its plan.

---

## 2. Introduction to LangChain

LangChain is the industry-standard framework for orchestrating LLMs. It provides several core abstractions:
- **Models**: Unified interfaces for LLMs and Chat Models (OpenAI, Anthropic, HuggingFace).
- **Prompts**: Template management.
- **Memory**: Giving the LLM state (e.g., remembering chat history).
- **Chains**: Sequences of calls (e.g., Prompt -> LLM -> Output Parser).
- **Agents**: Systems where the LLM decides the sequence of actions itself.

---

## 3. The ReAct Paradigm (Reason + Act)

The breakthrough in Agent architectures came from the **ReAct** paper (Reason + Act). 
Instead of just asking an LLM for an answer, we prompt it to think out loud.

The prompt structure forces the LLM to cycle through three phases:
1. **Thought**: What do I need to do next?
2. **Action**: Which tool should I use and with what input?
3. **Observation**: What did the tool return?

*(Cycle repeats until the LLM decides it has enough information to provide the `Final Answer`)*.

---

## 4. Equipping LLMs with Tools

A "Tool" in LangChain is simply a Python function with a highly descriptive docstring. The LLM reads the docstring to understand what the tool does, and LangChain maps the LLM's requested action to execute the Python function.

Examples of Tools:
- `WikipediaQueryRun`: Searches Wikipedia.
- `PythonREPL`: Executes Python code.
- `SQLDatabaseChain`: Translates natural language to SQL, runs it against a DB, and returns the result.

---

## 5. LangGraph: Stateful Multi-Agent Workflows

While standard LangChain Agents are great for simple tasks, they often get stuck in infinite loops on complex tasks.

**LangGraph** is a newer extension of LangChain built for creating complex, cyclic agent workflows as **Graphs** (nodes and edges). 
It allows you to define:
- **State**: A shared Python dictionary that updates as it passes between nodes.
- **Nodes**: Python functions (usually calling an LLM or a tool) that update the state.
- **Edges**: Conditional logic that determines which node to go to next.

This allows for **Multi-Agent Systems** where, for example, a "Researcher Agent" gathers data and passes it to a "Writer Agent", who then passes it to an "Editor Agent".

---

## 6. Implementation Example: A Math & Search Agent

Here is a simple example using LangChain to create an agent that can do math and search Wikipedia.

```python
# pip install langchain langchain-openai wikipedia
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.chains.llm_math.base import LLMMathChain

llm = ChatOpenAI(temperature=0, model="gpt-4")

# 1. Define Tools
wikipedia = WikipediaAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to answer questions about current events or historical facts."
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math."
    )
]

# 2. Initialize the Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

# 3. Run the Agent
# The agent will realize it needs to search Wikipedia for the birth year, then use the Calculator to subtract.
agent.run("What year was Albert Einstein born, and what is that year multiplied by 5?")
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Create a LangChain agent equipped with a `Requests` tool and a `BeautifulSoup` tool to scrape a website and summarize its contents.
- 🟡 **Intermediate**: Build a SQL Agent that connects to a local SQLite database (like a Northwind trades database) and can answer natural language queries like "Who are our top 5 customers by sales?"
- 🔴 **Advanced**: Use LangGraph to build a multi-agent debate system where two LLMs debate a topic and a third LLM acts as the judge, updating a shared state until a consensus is reached.

### What's Next
| Next | Why |
|------|-----|
| [ML Pipeline](../15-ML-In-Production/01-ML-Pipeline.md) | Now that you understand models and agents, it's time to learn how to put them into production. |

---

[← RAG & Vector Databases](09-RAG-And-Vector-Databases.md) | [Back to Index](../README.md) | [Next: Autoencoders →](../10-Generative/01-Autoencoders.md)
