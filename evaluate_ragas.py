import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Import your RAG components
from rag_chain import chain, retriever

load_dotenv()

# 1. Define your test questions and expected reference answers based on your faq.txt
eval_data = {
    "question": [
        "How do I track my order?",
        "What is your refund policy?",
        "How can I contact customer support?"
    ],
    "ground_truth": [
        "You can track your order using the tracking link provided in your shipping confirmation email.",
        "Refunds are processed within 5-7 business days after we receive your returned item.",
        "You can contact our support team via email or through the live chat feature on our website."
    ]
}

print("Running test queries through your RAG pipeline...")

responses = []
retrieved_contexts = []

# 2. Automatically generate answers and collect contexts from your bot
for q in eval_data["question"]:
    # Get retrieved chunks from ChromaDB
    docs = retriever.invoke(q)
    context_texts = [doc.page_content for doc in docs]
    
    # Get final answer from your Groq LLM chain
    ans = chain.invoke(q)
    
    responses.append(ans)
    retrieved_contexts.append(context_texts)

# 3. Format into a Hugging Face Dataset for RAGAS
dataset = Dataset.from_dict({
    "question": eval_data["question"],
    "answer": responses,
    "contexts": retrieved_contexts,
    "ground_truth": eval_data["ground_truth"]
})

print("Evaluating metrics with RAGAS...")

# 4. Run the evaluation
# Note: RAGAS uses an LLM judge. Ensure your OPENAI_API_KEY or GROQ evaluation setup is active, 
# or use RAGAS default evaluator.
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

# 5. Print out your official RAGAS scores!
print("\n========== RAGAS SCORES ==========")
print(result)
print("==================================")

# Save to a CSV file to show off in your GitHub/Portfolio
df = result.to_pandas()
df.to_csv("ragas_metrics_report.csv", index=False)
print("Saved report to ragas_metrics_report.csv!")