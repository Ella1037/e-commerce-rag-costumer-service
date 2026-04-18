# app/rag.py
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv()

# ── Knowledge base ──────────────────────────────────────────────
faqs = [
    {"q": "How do I return an item?",
     "a": "You can return items within 15 days of delivery. Go to My Orders, select the order, click Return/Refund. Ensure the item is unused and in original packaging."},
    {"q": "When will I receive my refund?",
     "a": "Refunds are processed within 3-7 business days after the seller confirms the return."},
    {"q": "How do I track my order?",
     "a": "Go to My Orders in the app or website. Click on the order to see real-time tracking updates."},
    {"q": "What payment methods are accepted?",
     "a": "Shopee accepts credit/debit cards, ShopeePay, bank transfer, cash on delivery, and installment payments."},
    {"q": "How do I contact the seller?",
     "a": "Go to the product page and click Chat Now to message the seller directly."},
    {"q": "What is Shopee Guarantee?",
     "a": "Shopee Guarantee holds your payment until you confirm receipt in good condition. This protects buyers from fraud."},
    {"q": "How do I use a voucher?",
     "a": "During checkout, click on the voucher field and enter your code. Discount will be applied automatically."},
    {"q": "My order is delayed, what should I do?",
     "a": "Check tracking first. If no update for 7+ days, contact the seller. If unresolved, file a dispute through Shopee Resolution Centre."},
    {"q": "Can I change my delivery address?",
     "a": "You can change the address before the seller ships. Go to My Orders and click Edit Address if available."},
    {"q": "How do Shopee Coins work?",
     "a": "Shopee Coins are earned from purchases. 100 Coins = 1 unit of local currency discount."},
    {"q": "What is the return window?",
     "a": "The standard return window is 15 days from delivery date. Some sellers offer extended return periods."},
    {"q": "How do I get a refund if item is damaged?",
     "a": "Take photos of the damaged item and go to My Orders > Return/Refund > Damaged Item. Upload photos as evidence."},
    {"q": "Can I get a refund without returning the item?",
     "a": "In some cases yes, if the seller agrees or Shopee decides in your favour during dispute resolution."},
    {"q": "How long does shipping take?",
     "a": "Standard shipping takes 3-7 days. Express shipping takes 1-2 days. International orders may take 7-21 days."},
    {"q": "What if seller doesn't respond?",
     "a": "If the seller does not respond within 3 days, you can escalate to Shopee customer support."},
]

# ── Model init (載入一次，module-level singleton) ──────────────
from app.embeddings import onnx_embeddings as embeddings

docs = [Document(
    page_content=f"Question: {f['q']}\nAnswer: {f['a']}",
    metadata={"source": "shopee_faq", "question": f["q"]}
) for f in faqs]

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Loading cross-encoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful Shopee customer service assistant.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {question}
Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Retrievers ───────────────────────────────────────────────────
class CrossEncoderReranker:
    def __init__(self, vs, ce, top_n=15, top_k=3):
        self.vs, self.ce, self.top_n, self.top_k = vs, ce, top_n, top_k
    def retrieve(self, query):
        cands = self.vs.similarity_search(query, k=self.top_n)
        scores = self.ce.predict([(query, d.page_content) for d in cands])
        return [d for _, d in sorted(zip(scores, cands), key=lambda x: -x[0])][:self.top_k]

class HyDERetriever:
    def __init__(self, vs, llm, top_k=3):
        self.vs, self.top_k = vs, top_k
        self.gen = (
            ChatPromptTemplate.from_template(
                "Write a short 2-3 sentence answer to this e-commerce question.\n"
                "Question: {question}\nAnswer:")
            | llm | StrOutputParser()
        )
    def retrieve(self, query):
        hypo = self.gen.invoke({"question": query})
        return self.vs.similarity_search(hypo, k=self.top_k)

reranker     = CrossEncoderReranker(vectorstore, cross_encoder)
hyde_retriever = HyDERetriever(vectorstore, llm)
baseline_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Chains ───────────────────────────────────────────────────────
chain_baseline = (
    {"context": baseline_retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT | llm | StrOutputParser()
)
chain_reranked = (
    {"context": RunnableLambda(reranker.retrieve) | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT | llm | StrOutputParser()
)
chain_hyde = (
    {"context": RunnableLambda(hyde_retriever.retrieve) | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT | llm | StrOutputParser()
)

CHAINS = {
    "baseline": chain_baseline,
    "reranker": chain_reranked,
    "hyde":     chain_hyde,
}

print("RAG engine ready ✓")