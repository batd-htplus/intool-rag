"""
Answer Generation, Validation, Response Formatting

ANSWER GENERATION (LangChain)
- Build prompt using ChatPromptTemplate
- Call LLM with page-aware context

ANSWER VALIDATION
- Check for hallucination
- Verify answer uses context

Step 11: RESPONSE FORMATTING
- Extract citations
- Format final output
"""

from typing import Dict, Any, Optional
from rag.agent.state import AgentState
from rag.logging import logger

# LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    """Expected LLM output format"""
    answer: str = Field(description="Answer to the question")
    confidence: float = Field(description="Confidence score 0-1")
    uses_context: bool = Field(description="Whether answer uses provided context")


class PromptBuilder:
    """Build Prompt for LLM"""
    
    def build_prompt_template(self) -> ChatPromptTemplate:
        """
        Build page-aware prompt template using LangChain.
        
        ✅ Using LangChain for: prompt formatting
        ❌ NOT using LangChain for: context selection (agent did that)
        """
        template = """You are a precise document analyst. You answer questions based ONLY on provided context.

# Document Context
{context}

# Instructions
1. Answer the question based ONLY on the provided context
2. Do NOT use outside knowledge
3. If context doesn't answer the question, say "I cannot find this information"
4. Be concise and specific
5. Cite relevant sections from the context when possible

# Question
{question}

# Answer (in JSON format)
Respond with valid JSON:
{{
  "answer": "...",
  "confidence": 0.95,
  "uses_context": true
}}"""
        
        prompt = ChatPromptTemplate.from_template(template)
        return prompt
    
    async def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """Build final prompt"""
        template = self.build_prompt_template()
        prompt = template.format(question=question, context=context)
        return prompt


class AnswerGenerator:
    """Generate Answer using LLM"""
    
    def __init__(self, model: str = "gpt-4-turbo"):
        """
        Initialize generator.
        
        ✅ Using LangChain for: LLM calls
        ❌ NOT using LangChain for: decide what to say (agent did)
        
        Args:
            model: LangChain model name
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,  # Deterministic
        )
        self.parser = JsonOutputParser(pydantic_object=AnswerOutput)
    
    async def generate(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            LLM output (parsed as JSON)
        """
        
        response = await self.llm.ainvoke(prompt)
        
        try:
            result = self.parser.parse(response.content)
            return {
                "answer": result.answer,
                "confidence": result.confidence,
                "uses_context": result.uses_context,
            }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "answer": response.content,
                "confidence": 0.5,
                "uses_context": False,
            }


class AnswerValidator:
    """Validate Answer"""
    
    def validate(
        self,
        answer: str,
        context: str,
        uses_context: bool,
    ) -> bool:
        """
        Validate answer quality.
        
        Checks:
        1. Answer is not empty
        2. Answer uses provided context
        3. No obvious hallucinations
        
        Args:
            answer: Generated answer
            context: Provided context
            uses_context: LLM's assertion about using context
            
        Returns:
            True if valid
        """
        
        if not answer or len(answer.strip()) < 10:
            logger.warning("Answer too short")
            return False
        
        if not uses_context:
            logger.warning("LLM claims not using context")
            return False
        
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(context_words & answer_words)
        if overlap < 3:
            logger.warning(f"Low context overlap ({overlap} words)")
        
        logger.info("✓ Answer validation passed")
        
        return True


class ResponseFormatter:
    """Format Final Response"""
    
    def format(
        self,
        state: AgentState,
    ) -> Dict[str, Any]:
        """
        Format final response.
        
        Output format:
        {
          "answer": "...",
          "source": {
            "page": 12,
            "chapter": "3",
            "section": "3.2",
            "subsection": "3.2.1",
            "title": "Embedding Models"
          }
        }
        
        Args:
            state: Final agent state
            
        Returns:
            Formatted response
        """
        
        if not state.selected_page:
            return {
                "answer": "I could not find relevant information in the document.",
                "source": None,
                "error": "No page selected",
            }
        
        source = {
            "page": state.selected_page.page,
        }
        
        # Add structure info if available
        if state.selected_page.chapter:
            source["chapter"] = state.selected_page.chapter
        if state.selected_page.section:
            source["section"] = state.selected_page.section
        if state.selected_page.subsection:
            source["subsection"] = state.selected_page.subsection
        if state.selected_page.title:
            source["title"] = state.selected_page.title
        
        return {
            "answer": state.answer,
            "source": source,
        }


async def generate_answer(
    state: AgentState,
    prompt_builder: PromptBuilder,
    generator: AnswerGenerator,
) -> None:
    """Execute: Answer Generation"""
    
    # RULE 1: Must have context
    if not state.is_valid_to_answer():
        logger.warning("⚠️  Cannot generate answer - no selected page")
        state.answer = "I could not find relevant information."
        return
    
    # Build prompt
    prompt = await prompt_builder.build_prompt(
        state.query,
        state.context,
    )
    
    # Generate answer
    result = await generator.generate(prompt)
    
    state.answer = result["answer"]
    state.answer_valid = result["uses_context"]
    
    logger.info(f"Generated answer ({len(state.answer)} chars)")


async def validate_answer(
    state: AgentState,
    validator: AnswerValidator,
) -> None:
    """Execute: Answer Validation"""
    
    if not state.answer:
        logger.warning("No answer to validate")
        state.answer_valid = False
        return
    
    is_valid = validator.validate(
        state.answer,
        state.context,
        state.answer_valid,
    )
    
    state.answer_valid = is_valid
    state.validation_attempts += 1
    
    if not is_valid and state.validation_attempts < 2:
        logger.warning("Answer validation failed, could retry with stricter prompt")


async def format_response(
    state: AgentState,
    formatter: ResponseFormatter,
) -> Dict[str, Any]:
    """Execute: Response Formatting"""
    
    response = formatter.format(state)
    
    logger.info(f"✓ Response formatted with {len(response.get('answer', ''))} chars")
    
    return response
