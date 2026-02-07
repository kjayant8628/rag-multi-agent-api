# from rag.retriever import get_context_for_question
# from dotenv import load_dotenv
# load_dotenv()

# import os
# import asyncio

# from autogen_agentchat.agents import AssistantAgent
# from autogen_ext.models.openai import OpenAIChatCompletionClient

# from agents.planner import PLANNER_SYSTEM_PROMPT, parse_planner_output
# from agents.researcher import (
#     REASONING_RESEARCHER_PROMPT,
#     RAG_RESEARCHER_PROMPT,
# )
# from agents.critic import CRITIC_SYSTEM_PROMPT, parse_critic_output


# # ======================================================
# # Helper: Run one researcher path + critic
# # ======================================================
# async def run_path(question, researcher, critic, use_rag=False):
#     """
#     Always returns: (answer: str, decision: dict)
#     Never returns None.
#     """

#     # ----------------------------
#     # Build researcher task
#     # ----------------------------
#     if use_rag:
#         print("  📚 Retrieving context from vector database...")
#         context = get_context_for_question(question, top_k=3)

#         task = f"""
# You MUST answer the question using ONLY the context below.

# CONTEXT:
# {context}

# QUESTION:
# {question}

# If the answer is not present in the context, say:
# "I cannot answer this question based on the provided documents."
# """
#     else:
#         task = question

#     # ----------------------------
#     # Researcher step
#     # ----------------------------
#     print(f"  🔬 {researcher.name} is working...")
#     researcher_result = await researcher.run(task=task)

#     researcher_msg = next(
#         (m for m in researcher_result.messages if m.source == researcher.name),
#         None
#     )

#     if researcher_msg is None:
#         return (
#             "SYSTEM ERROR: Researcher produced no output",
#             {"verdict": "REJECT", "issues": ["No researcher output"]}
#         )

#     answer = researcher_msg.content

#     # ----------------------------
#     # Critic step
#     # ----------------------------
#     print(f"  ⚖️ Critic is validating...")
#     critic_task = f"""
# QUESTION:
# {question}

# ANSWER:
# {answer}
# """

#     critic_result = await critic.run(task=critic_task)

#     critic_msg = next(
#         (m for m in critic_result.messages if m.source == "Critic"),
#         None
#     )

#     if critic_msg is None:
#         return (
#             answer,
#             {"verdict": "REJECT", "issues": ["No critic output"]}
#         )

#     decision = parse_critic_output(critic_msg.content)

#     # ✅ GUARANTEED RETURN
#     return answer, decision


# # ======================================================
# # Core engine: Answer ONE question
# # ======================================================
# async def answer_one_question(
#     question,
#     planner,
#     reasoning_researcher,
#     rag_researcher,
#     critic,
# ):
#     """
#     Process one question through the multi-agent workflow.
    
#     ALWAYS returns: (answer: str, status: str)
#     """
    
#     print(f"\n{'='*80}")
#     print(f"QUESTION: {question}")
#     print(f"{'='*80}\n")
    
#     # ----------------------------
#     # 1️⃣ PLANNER
#     # ----------------------------
#     print("🧠 PLANNER: Analyzing question...")
#     planner_result = await planner.run(task=question)
#     planner_msg = next(m for m in planner_result.messages if m.source == "Planner")
#     planner_decision = parse_planner_output(planner_msg.content)
    
#     print(f"  Route: {planner_decision['route']}")
#     print(f"  Reason: {planner_decision.get('reason', 'N/A')}\n")

#     # ----------------------------
#     # 2️⃣ ROUTING
#     # ----------------------------
#     if planner_decision["route"] == "OPTION_B":
#         primary = reasoning_researcher
#         secondary = rag_researcher
#     else:
#         primary = rag_researcher
#         secondary = reasoning_researcher

#     use_rag_primary = primary.name == "RAGResearcher"
#     use_rag_secondary = secondary.name == "RAGResearcher"
    
#     print(f"📋 PATH SELECTION:")
#     print(f"  Primary: {primary.name} (RAG: {'Yes' if use_rag_primary else 'No'})")
#     print(f"  Secondary: {secondary.name} (RAG: {'Yes' if use_rag_secondary else 'No'})\n")

#     # ----------------------------
#     # 3️⃣ PRIMARY PATH
#     # ----------------------------
#     print(f"🎯 PRIMARY PATH: {primary.name}")
#     print(f"{'-'*80}\n")

#     answer, decision = await run_path(
#         question=question,
#         researcher=primary,
#         critic=critic,
#         use_rag=use_rag_primary
#     )
    
#     print(f"\nCRITIC VERDICT: {decision['verdict']}\n")

#     if decision["verdict"] == "ACCEPT":
#         print("✅ PRIMARY PATH SUCCEEDED!")
#         print(f"\nFINAL ANSWER (PRIMARY ACCEPTED):")
#         print(f"{'='*80}")
#         print(answer)
#         print(f"{'='*80}\n")
#         return answer, "PRIMARY_ACCEPT"  # ✅ FIX: Return tuple

#     # ----------------------------
#     # 4️⃣ SECONDARY PATH (fallback)
#     # ----------------------------
#     print(f"\n⚠️ PRIMARY PATH FAILED — SWITCHING ROUTE")
#     print(f"🎯 SECONDARY PATH: {secondary.name}")
#     print(f"{'-'*80}\n")

#     answer2, decision2 = await run_path(
#         question=question,
#         researcher=secondary,
#         critic=critic,
#         use_rag=use_rag_secondary
#     )
    
#     print(f"\nCRITIC VERDICT: {decision2['verdict']}\n")

#     if decision2["verdict"] == "ACCEPT":
#         print("✅ SECONDARY PATH SUCCEEDED!")
#         print(f"\nFINAL ANSWER (SECONDARY ACCEPTED):")
#         print(f"{'='*80}")
#         print(answer2)
#         print(f"{'='*80}\n")
#         return answer2, "SECONDARY_ACCEPT"  # ✅ FIX: Return tuple
    
#     # ----------------------------
#     # 5️⃣ FINAL FAILURE
#     # ----------------------------
#     print("\n❌ FINAL FAILURE AFTER REROUTE")
#     print(f"ISSUES: {decision2.get('issues', ['Unknown'])}")
#     print(f"RECOMMENDATION: {decision2.get('recommendation', 'N/A')}\n")
    
#     failure_message = (
#         f"FAILED TO ANSWER.\n"
#         f"ISSUES: {decision2.get('issues', ['Unknown'])}\n"
#         f"RECOMMENDATION: {decision2.get('recommendation', 'N/A')}"
#     )
    
#     return failure_message, "FAILURE"  # ✅ Already returns tuple


# # ======================================================
# # MAIN: Batch driver
# # ======================================================
# async def main():
#     if not os.getenv("GROQ_API_KEY"):
#         raise RuntimeError("GROQ_API_KEY not found")

#     print("\n" + "="*80)
#     print("  🤖 MULTI-AGENT RAG SYSTEM - BATCH PROCESSING")
#     print("="*80 + "\n")

#     # ----------------------------
#     # Model client
#     # ----------------------------
#     print("🔧 Initializing model client...")
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

#     # ----------------------------
#     # Agents (created once)
#     # ----------------------------
#     print("🤖 Creating agents...")
    
#     planner = AssistantAgent(
#         name="Planner",
#         model_client=model_client,
#         system_message=PLANNER_SYSTEM_PROMPT,
#     )

#     reasoning_researcher = AssistantAgent(
#         name="ReasoningResearcher",
#         model_client=model_client,
#         system_message=REASONING_RESEARCHER_PROMPT,
#     )

#     rag_researcher = AssistantAgent(
#         name="RAGResearcher",
#         model_client=model_client,
#         system_message=RAG_RESEARCHER_PROMPT,
#     )

#     critic = AssistantAgent(
#         name="Critic",
#         model_client=model_client,
#         system_message=CRITIC_SYSTEM_PROMPT,
#     )
    
#     print("✅ Agents ready!\n")

#     # ----------------------------
#     # Load questions
#     # ----------------------------
#     print("📋 Loading questions from questions.txt...")
    
#     if not os.path.exists("questions.txt"):
#         print("\n⚠️ questions.txt not found. Creating sample file...")
#         with open("questions.txt", "w", encoding="utf-8") as f:
#             f.write("Why do heavier objects not fall faster in a vacuum?\n")
#             f.write("What is machine learning?\n")
#             f.write("What information is in the documents?\n")
#         print("✅ Sample questions.txt created.\n")
    
#     with open("questions.txt", "r", encoding="utf-8") as f:
#         questions = [q.strip() for q in f if q.strip()]
    
#     if not questions:
#         print("❌ No questions found in questions.txt")
#         return
    
#     print(f"✅ Loaded {len(questions)} question(s)\n")

#     results = []

#     # ----------------------------
#     # Batch processing
#     # ----------------------------
#     print("="*80)
#     print("  🚀 STARTING BATCH PROCESSING")
#     print("="*80)
    
#     for idx, question in enumerate(questions, start=1):
#         print(f"\n{'█'*80}")
#         print(f"  PROCESSING QUESTION {idx}/{len(questions)}")
#         print(f"{'█'*80}")

#         try:
#             answer, status = await answer_one_question(
#                 question,
#                 planner,
#                 reasoning_researcher,
#                 rag_researcher,
#                 critic,
#             )
#         except Exception as e:
#             print(f"\n❌ SYSTEM ERROR: {str(e)}\n")
#             import traceback
#             traceback.print_exc()
#             answer = f"SYSTEM ERROR: {str(e)}"
#             status = "ERROR"

#         results.append((question, answer, status))
        
#         print(f"\n{'='*80}")
#         print(f"  COMPLETED {idx}/{len(questions)} | Status: {status}")
#         print(f"{'='*80}\n")

#     # ----------------------------
#     # Write answers
#     # ----------------------------
#     print("\n📝 Writing results to answers.txt...")
    
#     with open("answers.txt", "w", encoding="utf-8") as f:
#         f.write("="*80 + "\n")
#         f.write("  MULTI-AGENT RAG SYSTEM - ANSWERS\n")
#         f.write("="*80 + "\n\n")
        
#         for i, (q, a, s) in enumerate(results, start=1):
#             f.write(f"QUESTION {i}:\n")
#             f.write(f"{q}\n\n")
#             f.write(f"STATUS: {s}\n\n")
#             f.write(f"ANSWER:\n")
#             f.write(f"{a}\n")
#             f.write("\n" + "-" * 80 + "\n\n")

#     # ----------------------------
#     # Summary statistics
#     # ----------------------------
#     primary_success = sum(1 for _, _, s in results if s == "PRIMARY_ACCEPT")
#     secondary_success = sum(1 for _, _, s in results if s == "SECONDARY_ACCEPT")
#     failures = sum(1 for _, _, s in results if s == "FAILURE")
#     errors = sum(1 for _, _, s in results if s == "ERROR")
    
#     print("\n" + "="*80)
#     print("  📊 BATCH PROCESSING SUMMARY")
#     print("="*80)
#     print(f"\nTotal Questions: {len(questions)}")
#     print(f"  ✅ Primary Success: {primary_success}")
#     print(f"  ✅ Secondary Success: {secondary_success}")
#     print(f"  ❌ Failures: {failures}")
#     print(f"  ⚠️ Errors: {errors}")
    
#     if len(questions) > 0:
#         success_rate = ((primary_success + secondary_success) / len(questions)) * 100
#         print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
#     print(f"\n✅ Results saved to: answers.txt")
#     print("="*80 + "\n")


# if __name__ == "__main__":
#     asyncio.run(main())



from rag.retriever import get_context_for_question
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.planner import PLANNER_SYSTEM_PROMPT, parse_planner_output
from agents.researcher import RAG_RESEARCHER_PROMPT, EXTERNAL_SEARCH_PROMPT
from agents.critic import CRITIC_SYSTEM_PROMPT, parse_critic_output, should_retry
from agents.external import create_external_agent


async def run_path_with_retry(question, researcher, critic, use_rag=False, max_retries=1):
    """
    Unified path for both RAG and EXTERNAL with retry logic.
    """
    retry_count = 0
    
    for attempt in range(max_retries + 1):
        print(f"\n  Attempt {attempt + 1}/{max_retries + 1}")
        
        if use_rag:
            context = get_context_for_question(question, top_k=3)
            task = f"""
Context: {context}

Question: {question}

{'[RETRY: Provide more detail and clarity]' if attempt > 0 else ''}

Answer using only the context above."""
        else:
            task = f"{question}\n\n{'[RETRY: Improve your answer]' if attempt > 0 else ''}"
        
        researcher_result = await researcher.run(task=task)
        researcher_msg = next(
            (m for m in researcher_result.messages if m.source == researcher.name), None
        )
        
        if not researcher_msg:
            return "ERROR: No researcher output", {'verdict': 'REJECT', 'confidence': 0.0}, 0, False
        
        answer = researcher_msg.content
        
        critic_task = f"Question: {question}\n\nAnswer: {answer}\n\nEvaluate this answer."
        critic_result = await critic.run(task=critic_task)
        critic_msg = next(
            (m for m in critic_result.messages if m.source == "Critic"), None
        )
        
        if not critic_msg:
            return answer, {'verdict': 'REJECT', 'confidence': 0.0}, 0, False
        
        decision = parse_critic_output(critic_msg.content)
        confidence = decision.get('confidence', 0.0)
        
        print(f"  Confidence: {confidence:.2f}")
        
        if confidence >= 0.80:
            print(f"  ✅ High confidence - Accept")
            decision['verdict'] = 'ACCEPT'
            return answer, decision, retry_count, True
        
        elif confidence >= 0.50:
            if should_retry(confidence, retry_count):
                retry_count += 1
                print(f"  🔄 Medium confidence - Retry")
                continue
            else:
                print(f"  ⚠️ Accept after retry")
                decision['verdict'] = 'ACCEPT'
                return answer, decision, retry_count, True
        else:
            print(f"  ❌ Low confidence - Reject")
            decision['verdict'] = 'REJECT'
            return answer, decision, retry_count, False
    
    decision['verdict'] = 'ACCEPT'
    return answer, decision, retry_count, True


async def answer_one_question(
    question,
    planner,
    rag_researcher,
    external_agent,
    critic,
):
    """
    Routes question → RAG / EXTERNAL / REJECT
    Returns: (answer: str, status: str)
    """

    print(f"\n{'='*80}")
    print(f"Q: {question}")
    print(f"{'='*80}\n")

    # ==========================
    # 1️⃣ PLANNER
    # ==========================
    try:
        planner_result = await planner.run(task=question)
        planner_msg = next((m for m in planner_result.messages if m.source == "Planner"), None)
        
        if not planner_msg:
            print("⚠️ No planner message found, defaulting to EXTERNAL")
            route = "EXTERNAL"
        else:
            planner_decision = parse_planner_output(planner_msg.content)
            route = planner_decision.get("route", "EXTERNAL")
            print(f"🧠 Planner route: {route}\n")
    except Exception as e:
        print(f"⚠️ Planner error: {e}, defaulting to EXTERNAL")
        route = "EXTERNAL"

    # ==========================
    # 2️⃣ RAG PATH
    # ==========================
    if route == "RAG":
        print("📚 Using RAG path")

        answer, decision, retries, accepted = await run_path_with_retry(
            question=question,
            researcher=rag_researcher,
            critic=critic,
            use_rag=True,
            max_retries=1,
        )

        print(
            f"\nRAG Result: {decision['verdict']} "
            f"(Confidence: {decision.get('confidence', 0):.2f}, Retries: {retries})"
        )

        if accepted and decision["verdict"] == "ACCEPT":
            print(f"\n✅ FINAL ANSWER (RAG | {decision['confidence']:.2f})")
            print(answer)
            return answer, f"RAG_{decision['confidence']:.2f}"

        print("\n⚠️ RAG failed → Falling back to EXTERNAL")

        # Fallback to external
        answer, decision, retries, accepted = await run_path_with_retry(
            question=question,
            researcher=external_agent,
            critic=critic,
            use_rag=False,
            max_retries=1,
        )

        if decision["verdict"] == "ACCEPT":
            print(f"\n✅ FINAL ANSWER (EXTERNAL | {decision['confidence']:.2f})")
            print(answer)
            return answer, f"EXTERNAL_{decision['confidence']:.2f}"

        return "Failed after RAG + EXTERNAL", "FAILURE"

    # ==========================
    # 3️⃣ EXTERNAL PATH
    # ==========================
    elif route == "EXTERNAL":
        print("🌐 Using EXTERNAL path")

        answer, decision, retries, accepted = await run_path_with_retry(
            question=question,
            researcher=external_agent,
            critic=critic,
            use_rag=False,
            max_retries=1,
        )

        print(
            f"\nEXTERNAL Result: {decision['verdict']} "
            f"(Confidence: {decision.get('confidence', 0):.2f})"
        )

        if decision["verdict"] == "ACCEPT":
            print(f"\n✅ FINAL ANSWER (EXTERNAL | {decision['confidence']:.2f})")
            print(answer)
            return answer, f"EXTERNAL_{decision['confidence']:.2f}"

        return "External answer rejected by critic", "FAILURE"

    # ==========================
    # 4️⃣ REJECT
    # ==========================
    else:
        print("🚫 Planner rejected the question")
        return "Rejected by planner", "REJECTED"


async def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found")

    print("\n🤖 Multi-Agent System with Confidence Scoring\n")

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

    if not os.path.exists("questions.txt"):
        with open("questions.txt", "w") as f:
            f.write("Why do heavier objects not fall faster in a vacuum?\n")
            f.write("What is machine learning?\n")
        print("Created sample questions.txt\n")
    
    with open("questions.txt", "r") as f:
        questions = [q.strip() for q in f if q.strip()]
    
    print(f"Processing {len(questions)} questions\n")

    results = []
    for idx, question in enumerate(questions, 1):
        print(f"\n{'█'*80}")
        print(f"Question {idx}/{len(questions)}")
        print(f"{'█'*80}")

        try:
            answer, status = await answer_one_question(
                question, 
                planner, 
                rag_researcher, 
                external_agent, 
                critic
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            answer = f"ERROR: {e}"
            status = "ERROR"

        results.append((question, answer, status))

    with open("answers.txt", "w", encoding="utf-8") as f:
        f.write("Multi-Agent System - Answers with Confidence Scoring\n")
        f.write("="*80 + "\n\n")
        
        for i, (q, a, s) in enumerate(results, 1):
            f.write(f"Q{i}: {q}\n")
            f.write(f"Status: {s}\n")
            f.write(f"Answer: {a}\n")
            f.write("-" * 80 + "\n\n")

    rag_success = sum(1 for _, _, s in results if s.startswith("RAG"))
    external_success = sum(1 for _, _, s in results if s.startswith("EXTERNAL"))
    failed = sum(1 for _, _, s in results if s == "FAILURE")
    rejected = sum(1 for _, _, s in results if s == "REJECTED")
    errors = sum(1 for _, _, s in results if s == "ERROR")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total: {len(questions)}")
    print(f"RAG Success: {rag_success}")
    print(f"External Success: {external_success}")
    print(f"Failed: {failed}")
    print(f"Rejected: {rejected}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {((rag_success + external_success) / len(questions) * 100):.1f}%")
    print(f"\nAnswers saved to answers.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())