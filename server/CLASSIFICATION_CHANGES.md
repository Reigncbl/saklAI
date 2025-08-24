# Conservative Classification System - Using config.yaml as Default

## Overview

The system has been updated to use `config.yaml` as the default template and only switch to specialized templates when there's a clear, strong indication that the user's query is specifically about that domain.

## Changes Made

### 1. **Conservative Classification Logic**

- **Increased confidence threshold** from 0.3 to 0.6 for keyword-based classification
- **Added default fallback** to `config.yaml` for general inquiries
- **Modified LLM classification** to be more conservative
- **Strengthened requirements** for specialized template selection

### 2. **Classification Behavior**

#### **Uses config.yaml (Default) for:**

- General greetings: "Hello", "Hi", "Good morning"
- Vague requests: "I need help", "Can you assist me?"
- General inquiries: "What services do you offer?"
- Unclear messages: "I have a question"
- Mixed/unclear intents

#### **Uses Specialized Templates Only When:**

- **Strong keyword matches** (confidence > 0.6)
- **Explicit domain mentions**: "credit card", "savings account", "loan"
- **Specific problems**: "My ATM card is blocked", "I can't login to the app"
- **Clear intent**: "I want to apply for...", "How do I open..."

### 3. **Updated Classification Categories**

```python
# Thresholds:
- Keyword confidence > 0.6 → Use specialized template
- Keyword confidence 0.4-0.6 + LLM support → Use specialized template
- Keyword confidence < 0.4 OR unclear → Use config.yaml
```

### 4. **Template Selection Priority**

1. **config.yaml** (Default)

   - General banking assistance
   - Mixed or unclear queries
   - Greetings and basic help

2. **Specialized Templates** (Only when confident)
   - `savings_accounts.yaml` - Clear savings account questions
   - `credit_cards.yaml` - Clear credit card questions
   - `loans.yaml` - Clear loan questions
   - `account_services.yaml` - Clear account problems
   - `digital_banking.yaml` - Clear app/online issues
   - `remittances_ofw.yaml` - Clear remittance questions

### 5. **Benefits of This Approach**

✅ **Consistent Experience**: Most users get the general banking assistant  
✅ **Specialized Help When Needed**: Clear domain-specific queries get expert help  
✅ **Fallback Safety**: Unclear queries default to helpful general assistance  
✅ **Better for Mixed Languages**: Taglish and unclear intents handled gracefully  
✅ **Reduced Classification Errors**: Less chance of wrong template selection

### 6. **Example Behavior**

| User Message                        | Template Used         | Reason                  |
| ----------------------------------- | --------------------- | ----------------------- |
| "Hello"                             | config.yaml           | General greeting        |
| "I need help"                       | config.yaml           | Vague request           |
| "What services do you offer?"       | config.yaml           | General inquiry         |
| "I want to apply for a credit card" | credit_cards.yaml     | Clear intent + keywords |
| "My ATM card is blocked"            | account_services.yaml | Specific problem        |
| "Paano mag-save?"                   | config.yaml           | Unclear/mixed language  |
| "How do I open a savings account?"  | savings_accounts.yaml | Clear intent + keywords |

### 7. **Server Response Format**

```json
{
  "status": "success",
  "template_used": "config.yaml",
  "classification": {
    "detected_category": "general_config",
    "confidence": 0.5,
    "method": "default_fallback"
  },
  "suggestions": [...]
}
```

This approach ensures that users get helpful, consistent responses while still providing specialized assistance when they have specific banking needs.
