from typing import Annotated, Iterator
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
import io
from langgraph.managed.is_last_step import RemainingSteps
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

def quiz(state: AgentState):
    """
    문제 생성 에이전트
    """
    user_message = state["messages"][-1].content
    
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
        HumanMessage(content=f"다음 주제에 대한 문제를 생성해주세요: {user_message}")
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
    
    return {
        "messages": [display_response],  
        "quiz_data": full_quiz_content    
    }
def check(state: AgentState):
    """
    채점 에이전트
    """
    user_answer = state["messages"][-1].content
    quiz_data = state.get("quiz_data", "퀴즈 데이터가 없습니다.")
    
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


