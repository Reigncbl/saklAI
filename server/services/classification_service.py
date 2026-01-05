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
                # Enhanced template mapping with more specific keywords
                template_mapping = {
                    # Primary keywords for better matching
                    'savings': 'savings_accounts.yaml',
                    'account': 'savings_accounts.yaml',
                    'deposit': 'savings_accounts.yaml',
                    'time deposit': 'savings_accounts.yaml',
                    'interest': 'savings_accounts.yaml',
                    
                    'credit': 'credit_cards.yaml',
                    'card': 'credit_cards.yaml', 
                    'rewards': 'credit_cards.yaml',
                    'cashback': 'credit_cards.yaml',
                    
                    'loan': 'loans.yaml',
                    'borrow': 'loans.yaml',
                    'financing': 'loans.yaml',
                    'mortgage': 'loans.yaml',
                    
                    'remittance': 'remittances_ofw.yaml',
                    'transfer': 'remittances_ofw.yaml',
                    'send money': 'remittances_ofw.yaml',
                    'ofw': 'remittances_ofw.yaml',
                    
                    'digital': 'digital_banking.yaml',
                    'online': 'digital_banking.yaml',
                    'mobile': 'digital_banking.yaml',
                    'app': 'digital_banking.yaml',
                    
                    'balance': 'account_services.yaml',
                    'statement': 'account_services.yaml',
                    'transaction': 'account_services.yaml',
                    
                    'branch': 'general_banking.yaml',
                    'location': 'general_banking.yaml',
                    'banking': 'general_banking.yaml',
                    'hours': 'general_banking.yaml',
                    
                    # Conversational keywords
                    'hello': 'config.yaml',
                    'hi': 'config.yaml',
                    'good morning': 'config.yaml',
                    'how are you': 'config.yaml',
                    'thanks': 'config.yaml'
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
            model_name="moonshotai/kimi-k2-instruct",
            temperature=0.0,  # Zero temperature for consistent classification
            max_tokens=150    # Reduced tokens for faster classification
        )
        
        # Create classification prompt
        classification_prompt = PromptTemplate(
            input_variables=["message", "available_templates"],
            template="""You are a high-precision banking classification agent. Analyze the customer message and select the MOST APPROPRIATE YAML template for generating a response.

TEMPLATE MAPPING:
- savings_accounts.yaml: savings accounts, checking accounts, deposits, account opening, minimum balance, interest rates, account features
- credit_cards.yaml: credit card applications, card features, rewards, cashback, annual fees, credit limits, card comparisons  
- loans.yaml: personal loans, business loans, auto loans, home loans, loan applications, requirements, interest rates
- remittances_ofw.yaml: money transfers, remittances, sending money abroad, OFW services, international transfers
- digital_banking.yaml: online banking, mobile app, digital services, BPI online, mobile banking, tech support
- account_services.yaml: account balance, statements, account management, transactions, account issues
- general_banking.yaml: branch locations, banking hours, contact information, general banking procedures, bank services
- config.yaml: greetings (hi, hello, good morning), casual conversation (how are you), small talk, non-banking topics

CLASSIFICATION RULES:
1. EXACT KEYWORD MATCHING:
   - "savings" OR "account opening" OR "minimum balance" → savings_accounts.yaml
   - "credit card" OR "rewards" OR "cashback" OR "annual fee" → credit_cards.yaml  
   - "loan" OR "borrow" OR "financing" → loans.yaml
   - "remittance" OR "transfer money" OR "send money" OR "OFW" → remittances_ofw.yaml
   - "online banking" OR "mobile app" OR "digital" → digital_banking.yaml
   - "balance" OR "statement" OR "transaction" → account_services.yaml
   - "branch" OR "location" OR "hours" OR "contact" → general_banking.yaml
   - "hello" OR "hi" OR "good morning" OR "how are you" → config.yaml

2. CONTEXT ANALYSIS:
   - Time deposits, CDs, savings plans → savings_accounts.yaml
   - Card applications, rewards programs → credit_cards.yaml
   - Loan applications, payment schedules → loans.yaml
   - International transfers, overseas → remittances_ofw.yaml

3. DEFAULT FALLBACK:
   - Banking-related but unclear → general_banking.yaml
   - Non-banking topics → config.yaml

Customer message: "{message}"

Respond with ONLY a JSON object in this EXACT format:
{{
    "template": "template_name.yaml",
    "category": "category_name", 
    "confidence": 0.95,
    "method": "langchain_agent"
}}

Be precise - the template selection directly impacts response quality."""
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
