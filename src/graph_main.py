from typing import Annotated, Iterator
from typing_extensions import TypedDict

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain.schema import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings


# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os

# API 키 정보 로드
load_dotenv()

summary_data = ""
quiz_data = ""
vectorstore = Milvus(
        embedding_function=OpenAIEmbeddings(),
        connection_args={"uri": os.environ["MILVUS_URI"], "token": os.environ["MILVUS_TOKEN"]},
    )

retriever = vectorstore.as_retriever()
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    name="ai_report_search",  # 도구의 이름을 입력합니다.
    description="use this tool to search ai report information from the PDF document",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)
tools = [retriever_tool]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    todo: str


def route(state: AgentState):
    global quiz_data

    input = state["messages"][-1].content
    template = f"""
다음 조건에 따라 입력 메시지의 목적을 분류하세요.

- 사용자가 요약을 원하면 "summary"를 반환하세요.
- 사용자가 퀴즈 출제를 원하면 "quiz"를 반환하세요.
- 사용자가 퀴즈 정답을 확인하려 하고, 퀴즈 데이터가 존재할 경우에만 "check"를 반환하세요.

현재 퀴즈 데이터 존재 여부: {"있음" if quiz_data else "없음"}

사용자 입력: {input}

반환값 (summary, quiz, check 중 하나):

"""

    llm = ChatOpenAI(model="gpt-4o")

    prompt = PromptTemplate(template=template,
                            input_variables=["input"])
    
    chain = prompt | llm
    response = chain.invoke({"input": input})

    return {"todo": response.content}
    
def summary(state: AgentState):
    global summary_data
    llm = ChatOpenAI(model="gpt-4")
    llm_bind_tools = llm.bind_tools(tools)
    response = llm_bind_tools.invoke(state["messages"])

    summary_data = response.content
    return {"messages": [response], "summary_data": response.content}

def quiz(state: AgentState):
    """
    문제 생성 에이전트
    """
    user_message = state["messages"][-1].content
    global summary_data
    quiz_prompt = """
    당신은 교육 컨텐츠 전문가로, 주어진 주제에 대한 5지선다형 5문제를 생성합니다.
    각 문제는 다음 형식을 따라야 합니다:

    1. 질문: [문제 내용]
       a) [선택지1]
       b) [선택지2]
       c) [선택지3]
       d) [선택지4]
       e) [선택지5]
       정답: [정답 알파벳]
       해설: [문제 해설]

    """
    
    llm = ChatOpenAI(model="gpt-4o")
    
    messages = [
        SystemMessage(content=quiz_prompt),
        HumanMessage(content=f"다음 주제에 대한 문제를 생성해주세요: {user_message}\n\n다음은 관련 요약 내용입니다:\n{summary_data}")
    ]
    
    response = llm.invoke(messages)
    full_quiz_content = response.content
    
    display_prompt = """
    당신은 교육 컨텐츠 편집자입니다. 주어진 문제에서 정답과 해설 부분을 제거하고, 
    문제와 선택지만 포함된 형태로 재구성해주세요.
    """
    
    display_messages = [
        SystemMessage(content=display_prompt),
        HumanMessage(content=f"다음 문제에서 정답과 해설을 제거하고 문제와 선택지만 남겨주세요:\n\n{full_quiz_content}")
    ]
    
    display_response = llm.invoke(display_messages)

    global quiz_data
    quiz_data = full_quiz_content

    return {
        "messages": [display_response],  
        "quiz_data": full_quiz_content    
    }

def check(state: AgentState):
    """
    채점 에이전트
    """
    user_answer = state["messages"][-1].content
    global quiz_data

    check_prompt = """
    당신은 교육 평가 전문가입니다. 사용자가 제출한 답변을 정확하게 채점하고 피드백을 제공합니다.
    채점 과정:
    1. 각 문제의 정답을 확인합니다.
    2. 사용자의 답변과 비교하여 점수를 계산합니다. (문제당 20점)
    3. 오답에 대한 구체적인 피드백과 설명을 제공합니다.
    """
    
    llm = ChatOpenAI(model="gpt-4o")
    
    messages = [
        SystemMessage(content=check_prompt),
        SystemMessage(content=f"다음은 원래 문제입니다:\n\n{quiz_data}"),
        HumanMessage(content=f"사용자 답변:\n{user_answer}\n\n이 답변을 채점해주세요.")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response]
    }


def route_condition(state):
    if state["todo"] == 'summary':
        return "summary"
    elif state["todo"] == 'quiz':
        return "quiz"
    elif state["todo"] == "check":
        return "check"
    else:
        return 
    
def should_use_summary_tool(state):
        last_message = state["messages"][-1]

        if not last_message.tool_calls:
            return "END"
        else:
            return "summary_tools"
        

class Agent():
    def __init__(self):
        self.graph = self.initialize_graph()
        self.state = AgentState(messages=[], todo="")  # todo 초기값 추가

    def initialize_graph(self):
        global tools
        graph = StateGraph(AgentState)
        summary_tool = ToolNode(tools)
        quiz_tool = ToolNode(tools)

        graph.add_node ("route", route)
        graph.add_node ("summary", summary)
        graph.add_node ("quiz", quiz)
        graph.add_node ("check", check)
        graph.add_node("summary_tools", summary_tool)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            route_condition,
            {
                "summary": "summary",
                "quiz": "quiz",
                "check": "check",
                END: END
            }
        )
        graph.add_conditional_edges(
            "summary",
            should_use_summary_tool,
            {
                "END": END,
                "summary_tools": "summary_tools"
            }
        )
        graph.add_edge("summary_tools", "summary")

        graph.add_edge ("quiz", END)
        graph.add_edge ("check", END)

        return graph.compile(checkpointer=MemorySaver())
        #return graph.compile()

    def run(self, user_input) -> Iterator[str]:
        # config 설정
        config = RunnableConfig(
            recursion_limit=7,
            configurable={"thread_id": "1"}
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


