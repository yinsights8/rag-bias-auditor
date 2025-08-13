from langchain.chains import RetrievalQA

class RAGAnswerGenerator:
    """
    Generates answers using an LLM in a retrieval-augmented setup.
    Wraps LangChain's RetrievalQA for modular pipeline integration.
    """

    def __init__(self, llm, retriever, prompt_template=None):
        """
        Args:
            llm: LangChain-compatible LLM instance (e.g. ChatOpenAI, ChatTogether)
            retriever: A retriever instance (e.g., from RAGRetrieverLoader)
            prompt_template: Optional custom prompt (LangChain PromptTemplate or None)
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.qa_chain = self._setup_chain()

    def _setup_chain(self):
        """Initializes the RetrievalQA chain."""
        if self.prompt_template is not None:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
        else:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever
            )

    def generate_answer(self, question, return_context=False):
        """
        Generates an answer for the given question.

        Args:
            question: User query (str)
            return_context: If True, returns both answer and context used (default: False)

        Returns:
            If return_context is False: str (answer)
            If return_context is True: dict {'answer': ..., 'context': ...}
        """
        result = self.qa_chain({"query": question})
        if return_context:
            # This assumes chain returns a 'source_documents' key (standard for RetrievalQA)
            return {
                "answer": result["result"],
                "context": [doc.page_content for doc in result.get("source_documents", [])]
            }
        return result["result"]

# --- Example Usage ---

# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=...)

# rag_answer_gen = RAGAnswerGenerator(llm=llm, retriever=bm25_retriever)
# answer = rag_answer_gen.generate_answer("Why did Scotland seek independence from the UK?")
# print(answer)

# To get both answer and retrieved context:
# full_output = rag_answer_gen.generate_answer("Your query...", return_context=True)
