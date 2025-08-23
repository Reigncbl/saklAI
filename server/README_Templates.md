# BPI Intelligent RAG Template System

## Overview

The BPI RAG system now includes intelligent inquiry classification that automatically detects the type of customer inquiry and selects the most appropriate response template. This allows the agent to provide specialized, contextual responses based on the customer's specific banking needs.

## Available Templates

### 1. Savings Accounts (`savings_accounts.yaml`)

**Triggers:** Questions about savings accounts, deposits, minimum balances, interest rates
**Specializes in:**

- Account type recommendations (#SaveUp, #MySaveUp, Jumpstart, Saver-Plus, Regular Savings)
- Minimum balance requirements and waivers
- Interest rates and earning requirements
- Digital vs traditional banking features
- Foreign currency options for OFWs

### 2. Credit Cards (`credit_cards.yaml`)

**Triggers:** Questions about credit cards, rewards, applications, fees, limits
**Specializes in:**

- Card recommendations (Platinum, Gold, Blue, Signature, etc.)
- Income requirements for different card tiers
- Rewards programs and earning rates
- Annual fees and fee waivers
- Lifestyle benefits and privileges

### 3. Loans (`loans.yaml`)

**Triggers:** Questions about personal loans, mortgages, auto loans, financing
**Specializes in:**

- Loan product recommendations
- Interest rates and payment terms
- Eligibility criteria and requirements
- Processing fees and timelines
- Documentation requirements

### 4. Remittances & OFW Services (`remittances_ofw.yaml`)

**Triggers:** Questions about money transfers, foreign currency, OFW services
**Specializes in:**

- Padala remittance services
- Foreign currency accounts
- OFW-specific products and benefits
- Exchange rates and fees
- International banking features

### 5. Digital Banking (`digital_banking.yaml`)

**Triggers:** Questions about online banking, mobile apps, digital services
**Specializes in:**

- BPI Mobile App setup and features
- Online banking access and security
- Digital payments and transfers
- QR code payments
- Troubleshooting digital issues

### 6. Account Services (`account_services.yaml`)

**Triggers:** Questions about account balance, transactions, ATM issues, card problems
**Specializes in:**

- Account balance and transaction inquiries
- ATM card issues and replacements
- Blocked account resolution
- Transaction disputes
- Account maintenance and updates

### 7. Investments (`investing_funds.yaml`)

**Triggers:** Questions about mutual funds, UITF, investment products
**Specializes in:**

- Investment product recommendations
- Fund performance and features
- Investment strategies
- Portfolio management

### 8. General Banking (`general_banking.yaml`)

**Triggers:** General questions, branch info, customer service
**Specializes in:**

- General product overviews
- Branch locations and services
- Customer support information
- Basic banking procedures

## How to Use

### Automatic Classification (Recommended)

```json
{
  "user_id": "customer123",
  "message": "I want to apply for a credit card with good rewards",
  "prompt_type": "auto"
}
```

The system will:

1. Analyze the customer's message
2. Classify the inquiry type
3. Select the appropriate template
4. Generate contextual suggestions based on BPI PDF content

### Manual Template Selection

```json
{
  "user_id": "customer123",
  "message": "Customer inquiry text",
  "prompt_type": "credit_cards"
}
```

Available manual prompt types:

- `savings_accounts`
- `credit_cards`
- `loans`
- `remittances_ofw`
- `digital_banking`
- `account_services`
- `investments`
- `general_banking`

## API Response Format

```json
{
  "status": "success",
  "user_id": "customer123",
  "message": "Customer inquiry",
  "template_used": "credit_cards.yaml",
  "classification": {
    "detected_category": "credit_cards",
    "confidence": 0.8,
    "method": "llm_based"
  },
  "suggestions": [
    {
      "analysis": "customer_intent: new_application",
      "category": "Card Recommendation",
      "suggestion": "Based on your interest in rewards..."
    }
  ]
}
```

## Benefits

1. **Contextual Responses**: Each template is specialized for specific banking topics
2. **Automatic Detection**: No need to manually specify the inquiry type
3. **Consistent Format**: All responses follow the same structured format
4. **Comprehensive Coverage**: Templates cover all major BPI products and services
5. **Fallback Support**: Graceful handling of unclear or complex inquiries
6. **PDF-Based Knowledge**: All responses are grounded in actual BPI product data

## Template Development

Each template includes:

- **Specialized Role**: Agent acts as a specific type of BPI specialist
- **Product Knowledge**: Deep understanding of relevant BPI products
- **Response Structure**: Consistent JSON format with analysis, category, and suggestions
- **Key Areas**: Specific topics to address for each inquiry type
- **Tone Guidelines**: Appropriate communication style for the context

This system ensures that customers receive expert-level, contextual assistance tailored to their specific banking needs while maintaining accuracy through the BPI PDF knowledge base.
