"""
Classification service for intelligent message routing and classification.
"""

from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import json


class YAMLTemplateParser(BaseOutputParser):
    """Custom parser to extract YAML template from LLM response"""
    
    def parse(self, text: str) -> dict:
        try:
            # Try to extract JSON from the response
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]
                result = json.loads(json_str)
                return result
            else:
                # Fallback: look for template name in text
                template_mapping = {
                    'savings': 'savings_accounts.yaml',
                    'credit': 'credit_cards.yaml', 
                    'loan': 'loans.yaml',
                    'remittance': 'remittances_ofw.yaml',
                    'digital': 'digital_banking.yaml',
                    'account': 'account_services.yaml',
                    'banking': 'general_banking.yaml',
                    'investment': 'config.yaml',  # No specific investment template yet
                    'invest': 'config.yaml'
                }
                
                text_lower = text.lower()
                for keyword, template in template_mapping.items():
                    if keyword in text_lower:
                        return {
                            "template": template,
                            "category": keyword,
                            "confidence": 0.7,
                            "method": "keyword_fallback"
                        }
                
                return {
                    "template": "config.yaml",
                    "category": "general",
                    "confidence": 0.5,
                    "method": "default_fallback"
                }
        except Exception as e:
            print(f"Parser error: {e}")
            return {
                "template": "config.yaml",
                "category": "general", 
                "confidence": 0.3,
                "method": "error_fallback"
            }


async def classify_with_langchain_agent(message: str, api_key: str) -> dict:
    """
    Use LangChain agent to intelligently classify customer inquiry and select appropriate YAML template
    """
    try:
        # Initialize Groq LLM through LangChain
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-8b-8192",
            temperature=0.1
        )
        
        # Create classification prompt
        classification_prompt = PromptTemplate(
            input_variables=["message", "available_templates"],
            template="""You are a banking classification agent. Analyze the customer message and select the most appropriate YAML template for generating a response.

Available templates and their purposes:
- savings_accounts.yaml: Savings account inquiries, opening accounts, requirements, features
- credit_cards.yaml: Credit card applications, features, requirements, issues  
- loans.yaml: Personal loans, business loans, requirements, applications
- remittances_ofw.yaml: Money transfers, OFW remittances, international transfers
- digital_banking.yaml: Online banking, mobile app, digital services
- account_services.yaml: Account management, balance inquiries, statements
- general_banking.yaml: Banking questions, branch locations, contact info, banking procedures
- config.yaml: General conversation, greetings, small talk, non-banking topics

Customer message: "{message}"

IMPORTANT CLASSIFICATION RULES:
- Use config.yaml for: greetings (hi, hello), casual conversation (how are you), small talk, non-banking questions
- Use general_banking.yaml for: banking questions that don't fit specific categories, branch info, general banking procedures
- Use specific templates for: targeted banking product inquiries

Analyze the message and respond with ONLY a JSON object in this exact format:
{{
    "template": "template_name.yaml",
    "category": "category_name", 
    "confidence": 0.95,
    "method": "langchain_agent"
}}

Choose the template that best matches the customer's intent."""
        )
        
        # Create the chain
        chain = LLMChain(
            llm=llm,
            prompt=classification_prompt,
            output_parser=YAMLTemplateParser()
        )
        
        # Get available templates
        available_templates = [
            "savings_accounts.yaml", "credit_cards.yaml", "loans.yaml", 
            "remittances_ofw.yaml", "digital_banking.yaml", "account_services.yaml",
            "general_banking.yaml", "config.yaml"
        ]
        
        # Run the classification
        result = await chain.arun(
            message=message,
            available_templates=", ".join(available_templates)
        )
        
        # Ensure result has required fields
        if not isinstance(result, dict):
            result = {"template": "config.yaml", "category": "general", "confidence": 0.5, "method": "fallback"}
            
        return result
        
    except Exception as e:
        print(f"LangChain classification error: {e}")
        return {
            "template": "config.yaml",
            "category": "general",
            "confidence": 0.3,
            "method": "error_fallback"
        }


def should_use_rag(template_name: str) -> bool:
    """
    Determine if we should use RAG based on the template
    Use RAG only for banking-related topics, not for general conversation
    """
    rag_templates = [
        "savings_accounts.yaml",
        "credit_cards.yaml", 
        "loans.yaml",
        "remittances_ofw.yaml",
        "digital_banking.yaml",
        "account_services.yaml",
        "general_banking.yaml"  # Include general banking in RAG
    ]
    
    return template_name in rag_templates
