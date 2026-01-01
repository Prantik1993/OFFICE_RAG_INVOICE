import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from src.rag.prompts import build_rag_chain

# 1. Define a "Golden Dataset"
# In production, these should be real user queries + expert answers
questions = [
    "What is the purpose of Article 1?",
    "Does this regulation apply to processing outside the Union?",
    "What is the definition of 'personal data'?"
]

ground_truths = [
    ["Article 1 lays down rules relating to the protection of natural persons..."],
    ["Yes, if the processing activities are related to offering goods or services..."],
    ["Any information relating to an identified or identifiable natural person."]
]

def run_eval():
    print("Starting Evaluation...")
    chain = build_rag_chain()
    
    answers = []
    contexts = []
    
    # 2. Run the Chain
    for query in questions:
        # We invoke the chain and capture the result
        response = chain.invoke({"input": query, "chat_history": []})
        answers.append(response)
        
        # Note: To get 'contexts' for Ragas, you might need to modify your chain 
        # to return the raw source documents alongside the answer string.
        # For now, we assume implicit context for simple demo.
        contexts.append(["Content from PDF..."]) # You need to wire this up to actual retrieved docs

    # 3. Score with Ragas (LLM-as-a-Judge)
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    
    print(f"Evaluation Results:\n{results}")

if __name__ == "__main__":
    run_eval()