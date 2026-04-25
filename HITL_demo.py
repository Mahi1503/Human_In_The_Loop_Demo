from typing import TypedDict
from typing_extensions import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    
    decision = interrupt({
        "type" : "approval",
        "reason" : "model is about to answer a user question.",
        "question" : state["messages"][-1].content,
        "instruction" : "Approve this question? yes/no" 
    })

    if decision["approved"] == "no":
        return {"messages":[AIMessage(content="Not Approved.")]}
    
    else:
        response = llm.invoke(state["messages"])
        return {"messages":[response]}

    

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_edge(START,"chat")
graph.add_edge("chat",END)

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

  
config = {"configurable" : {"thread_id": "1234"}}
initial_input = {
    "messages": [
        ("user", "Explain gradient descent in very simple terms.")
    ]
}

result = app.invoke(initial_input, config=config)

message = result["__interrupt__"][0].value
user_input = input(f"\nBackend message - {message} \n Approve this question? (y/n): ")

final_result = app.invoke(
    Command(resume={"approved": user_input}),
    config=config,
)

print(final_result["messages"][-1].content)