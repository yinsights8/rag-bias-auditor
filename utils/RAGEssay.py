from langchain.prompts import PromptTemplate
import os
from utils.logger import logging

# ==== RAG Essay Writer ====
class RAGEssayWriter:
    def __init__(self, retriever, llm, role_descriptions=None):
        self.retriever = retriever
        self.llm = llm
        self.role_descriptions = role_descriptions or {}

    def write_essay(self, topic, role=None, k=5):
        docs = self.retriever.get_relevant_documents(topic, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        logging.info(f"Context length: {len(docs)}, k : {k}")
        system_prompt = f"{self.role_descriptions[role][1]}\n" if role and role in self.role_descriptions else ""
        prompt = PromptTemplate(
            template="{system_prompt}\nGiven the following evidence:\n{context}\nWrite a short essay in response to: \"{topic}\"",
            input_variables=["system_prompt", "context", "topic"]
        )
        chain = prompt | self.llm
        essay = chain.invoke({
            "system_prompt": system_prompt,
            "context": context,
            "topic": topic
        })
        class Essay:  # for compatibility with .content
            def __init__(self, content): self.content = content
        return Essay(essay.content if hasattr(essay, 'content') else essay)
