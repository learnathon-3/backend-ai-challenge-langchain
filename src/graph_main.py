from typing import Annotated, Iterator
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    todo: str


def route(state: AgentState):
    template = """당신은 사용자의 질문에 따라 카테고리를 반환하는 역할을 합니다.
사용자의 질문: {question}

아래는 응답해야하는 답변 입니다. 하나만 선택해서 반환하고, 그 외의 다른 설명은 하지 마세요.
1. "summary"
2. "quiz"
3. "check"

"""

    question = state["messages"][-1].content
    llm = ChatOpenAI(model="gpt-4")

    prompt = PromptTemplate(template=template,
                            input_variables=["question"])
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    return {"todo": response.content}
    
def summary(state: AgentState):
    prompt = """
    AI 뉴스를 도구를 사용해서 가져와서 요약해 주세요.
"""

    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def route_condition(state):
    #if state["todo"] == 'summary':
        return "summary"
    #elif state["todo"] == 'quiz':
    #    return "quiz"
    #else:
    #    return "check"


class Agent():
    def __init__(self):
        self.graph = self.initialize_graph()
        self.state = AgentState(messages=[], todo="")  # todo 초기값 추가

    def initialize_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node ("route", route)
        graph.add_node ("summary", summary)
        #graph.add_node ("quiz", quiz)
        #graph.add_node ("check", check)

        
        graph.add_conditional_edges(
            START,
            route_condition,
            {
                "summary": "summary",
                #"quiz": "quiz",
                #"check": "check"
            }
        )

        graph.add_edge ("summary", END)
        #graph.add_edge ("quiz", END)
        #graph.add_edge ("check", END)

        #return graph.compile(checkpointer=MemorySaver())
        return graph.compile()

    def run(self, user_input) -> Iterator[str]:
        # config 설정
        config = RunnableConfig(
            recursion_limit=30,
        )

        self.state["messages"] = [HumanMessage(content=user_input)]
        self.state["todo"] = ""  # todo 초기화

        response_stream = self.graph.stream(
                                            self.state,
                                            stream_mode="messages",
                                            config=config
                                            )
        
        for chunk_msg, metadata in response_stream:
            if chunk_msg.content:
                yield chunk_msg.content


