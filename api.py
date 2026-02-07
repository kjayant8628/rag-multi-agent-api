# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import os
# import asyncio

# # Import your existing working code
# from agentchat1 import answer_one_question
# from agents.planner import PLANNER_SYSTEM_PROMPT
# from agents.researcher import RAG_RESEARCHER_PROMPT
# from agents.critic import CRITIC_SYSTEM_PROMPT
# from agents.external import create_external_agent

# from autogen_agentchat.agents import AssistantAgent
# from autogen_ext.models.openai import OpenAIChatCompletionClient

# load_dotenv()

# app = FastAPI(title="RAG Multi-Agent API", version="1.0.0")

# # Global agent instances (initialized once)
# planner = None
# rag_researcher = None
# external_agent = None
# critic = None


# class QueryRequest(BaseModel):
#     question: str


# class QueryResponse(BaseModel):
#     question: str
#     answer: str
#     status: str
#     route: str = None


# def initialize_agents():
#     """Initialize all agents once at startup"""
#     global planner, rag_researcher, external_agent, critic
    
#     if not os.getenv("GROQ_API_KEY"):
#         raise RuntimeError("GROQ_API_KEY not found in environment variables")
    
#     model_client = OpenAIChatCompletionClient(
#         model="llama-3.3-70b-versatile",
#         api_key=os.getenv("GROQ_API_KEY"),
#         base_url="https://api.groq.com/openai/v1",
#         model_info={
#             "family": "llama",
#             "vision": False,
#             "function_calling": True,
#             "json_output": True,
#             "structured_output": False,
#         },
#     )
    
#     planner = AssistantAgent(
#         name="Planner",
#         model_client=model_client,
#         system_message=PLANNER_SYSTEM_PROMPT,
#     )
    
#     rag_researcher = AssistantAgent(
#         name="RAGResearcher",
#         model_client=model_client,
#         system_message=RAG_RESEARCHER_PROMPT,
#     )
    
#     external_agent = create_external_agent(model_client)
    
#     critic = AssistantAgent(
#         name="Critic",
#         model_client=model_client,
#         system_message=CRITIC_SYSTEM_PROMPT,
#     )
    
#     print("✅ All agents initialized successfully")


# @app.on_event("startup")
# async def startup_event():
#     """Initialize agents when the API starts"""
#     initialize_agents()


# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "message": "RAG Multi-Agent API is running",
#         "version": "1.0.0"
#     }


# @app.get("/health")
# async def health():
#     """Detailed health check"""
#     return {
#         "status": "healthy",
#         "agents_initialized": all([planner, rag_researcher, external_agent, critic]),
#         "groq_api_configured": bool(os.getenv("GROQ_API_KEY"))
#     }


# @app.post("/query", response_model=QueryResponse)
# async def query_endpoint(request: QueryRequest):
#     """
#     Main endpoint: Send a question and get an answer from the multi-agent system
    
#     Example request:
# ```
#     POST /query
#     {
#         "question": "What is machine learning?"
#     }
# ```
#     """
#     if not all([planner, rag_researcher, external_agent, critic]):
#         raise HTTPException(status_code=503, detail="Agents not initialized")
    
#     try:
#         # Use your existing answer_one_question function - NO CHANGES NEEDED!
#         answer, status = await answer_one_question(
#             question=request.question,
#             planner=planner,
#             rag_researcher=rag_researcher,
#             external_agent=external_agent,
#             critic=critic
#         )
        
#         # Extract route from status (e.g., "RAG_0.85" -> "RAG")
#         route = status.split("_")[0] if "_" in status else status
        
#         return QueryResponse(
#             question=request.question,
#             answer=answer,
#             status=status,
#             route=route
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
# Now import your modules
from agentchat1 import answer_one_question
from agents.planner import PLANNER_SYSTEM_PROMPT
from agents.researcher import RAG_RESEARCHER_PROMPT
from agents.critic import CRITIC_SYSTEM_PROMPT
from agents.external import create_external_agent

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

app = FastAPI(title="RAG Multi-Agent API", version="1.0.0")

# Global agent instances (initialized once)
planner = None
rag_researcher = None
external_agent = None
critic = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    status: str
    route: str = None


def initialize_agents():
    """Initialize all agents once at startup"""
    global planner, rag_researcher, external_agent, critic
    

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found in environment variables")

    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "family": "llama",
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": False,
        },
    )
    
    planner = AssistantAgent(
        name="Planner",
        model_client=model_client,
        system_message=PLANNER_SYSTEM_PROMPT,
    )
    
    rag_researcher = AssistantAgent(
        name="RAGResearcher",
        model_client=model_client,
        system_message=RAG_RESEARCHER_PROMPT,
    )
    
    external_agent = create_external_agent(model_client)
    
    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        system_message=CRITIC_SYSTEM_PROMPT,
    )
    
    print("✅ All agents initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize agents when the API starts"""
    initialize_agents()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG Multi-Agent API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agents_initialized": all([planner, rag_researcher, external_agent, critic]),
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY"))
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint: Send a question and get an answer from the multi-agent system
    
    Example request:
```
    POST /query
    {
        "question": "What is machine learning?"
    }
```
    """
    if not all([planner, rag_researcher, external_agent, critic]):
        raise HTTPException(status_code=503, detail="Agents not initialized")
    
    try:
        # Use your existing answer_one_question function - NO CHANGES NEEDED!
        answer, status = await answer_one_question(
            question=request.question,
            planner=planner,
            rag_researcher=rag_researcher,
            external_agent=external_agent,
            critic=critic
        )
        
        # Extract route from status (e.g., "RAG_0.85" -> "RAG")
        route = status.split("_")[0] if "_" in status else status
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            status=status,
            route=route
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
