import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

def run_app():
    # 1. Setup & Validation
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
        print("Please set your GOOGLE_API_KEY in the .env file.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        print("Vector index not found. Please run 'python ingest.py' first.")
        return
        
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # 2. CONTEXTUALIZE QUESTION (The "Memory" Part 1)
    # This chain re-writes the user's question to be standalone if history exists.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. ANSWER GENERATION (The "Memory" Part 2)
    system_prompt = (
        "You are a friendly and helpful library guide. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "Answer in a polite and warm tone. If you don't know the answer, say so."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 4. STATE MANAGEMENT (Session History)
    # In an interview, mention that we use a dict for simplicity, 
    # but Redis is better for production.
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("--- Welcome to the Memory-Enabled Library Guide AI! ---")
    print("(Type 'exit' to quit)")

    session_id = "default_user" # In real app, this would be a unique ID per user

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        print("Librarian is thinking...")
        try:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nLibrarian: {response['answer']}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_app()
