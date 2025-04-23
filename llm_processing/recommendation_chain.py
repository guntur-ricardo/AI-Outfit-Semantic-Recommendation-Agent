import os
import logging

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logger = logging.getLogger(__name__)


class RecommendationChain:
    """
    Wraps a LangChain-based chat pipeline to refine semantic search hits
    into human-friendly fashion recommendations using RunnableSequence.
    """

    def __init__(self, model_name: str = None, temperature: float = 0.7):
        # Determine model name from env or default
        if model_name is None:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        logger.info("Initializing ChatOpenAI with model '%s'", model_name)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        # System instructions
        system_template = (
            "You are a fashion recommendation assistant. "
            "Given a user's query and a list of recommended products with titles, descriptions, similarity scores and average rating, "
            "produce concise, human-friendly outfit suggestions or product recommendations. "
            "The average ratings are from 0 to 5 and represent Amazon review ratings, if the rating is over 3, mention that it is a well reviewed product on Amazon."
            "Similarity scores are internal metrics that represent how similar the product is to the user's query, only use this to decide which "
            "product to mention first but do not display the score."
        )
        # User prompt template
        human_template = (
            "User query: {query}\n\n"
            "Products:\n{products}\n\n"
            "Please reply with top recommendations, referencing product titles."
        )

        # Build the composite chat prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        # Create a RunnableSequence: prompt followed by LLM
        self.chain = self.prompt | self.llm

    def run(self, query: str, products: list[dict]) -> BaseMessage:
        """
        Formats product metadata (including average_rating) and invokes the chat pipeline.
        Returns the assistant's response text.
        """
        logger.debug("Formatting %d products for query '%s'", len(products), query)
        products_str = []
        for p in products:
            title = p.get('title', 'Unknown')
            desc = p.get('description', '')
            score = p.get('score', 0.0)
            rating = p.get('average_rating', 'N/A')
            products_str.append(
                f"- {title}: {desc} (score: {score:.2f}, rating: {rating})"
            )
        formatted = "\n".join(products_str)

        logger.info("Invoking recommendation chain for query '%s'", query)
        response = self.chain.invoke({"query": query, "products": formatted})
        logger.debug("RecommendationChain response: %s", response)
        return response


# Shared instance to import in application
recommendation_chain = RecommendationChain()
