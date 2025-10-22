# Changes to chat_agent.py - LangChain Structured Output Implementation

## Summary
Updated `chat_agent.py` to use LangChain's structured output capabilities with the `ChatDartmouth` LLM, replacing the problematic OpenAI API call pattern with a proper LangChain chain.

## Key Changes

### 1. Added LangChain Imports
```python
from langchain_dartmouth.llms import ChatDartmouth
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
```

### 2. Updated `run_episode()` Function

**Before:**
```python
def run_episode(model="llam", ...):
    chat = ChatDartmouth(dartmouth_chat_api_key=api_key, ...)
    
    # Inside loop:
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation.extend([{"role": "user", "content": HUMAN_PROMPT.format(history=history)}])
    
    response = client.responses.parse(...)  # This was broken
    response = chat.invoke(conversation)    # This didn't give structured output
    
    output = response.output_parsed  # This doesn't exist on BaseMessage
```

**After:**
```python
def run_episode(model="llama-3.3-70b", ...):
    # Create LLM
    chat = ChatDartmouth(dartmouth_chat_api_key=api_key, ...)
    
    # Create parser for structured output
    parser = PydanticOutputParser(pydantic_object=Action)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\n{format_instructions}"),
        ("user", HUMAN_PROMPT)
    ])
    
    # Create chain
    chain = prompt | chat | parser
    
    # Inside loop:
    try:
        output = chain.invoke({
            "history": history,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        print(f"\nError getting structured output: {e}")
        # Fallback to MONITOR action
        output = Action(analyse="Error occurred, monitoring network", 
                       action="MONITOR", host="nohost")
```

### 3. Updated Model List
Changed from OpenAI-specific models to Dartmouth Chat supported models:
```python
models = [
    "llama-3.3-70b",
    "gpt-4o",
    "gpt-4o-mini",
]
```

### 4. Fixed API Key Usage
Changed from `api_key` (OpenAI) to `chat_key` (Dartmouth):
```python
run_episode(
    model=model,
    temperature=temp,
    api_key=chat_key,  # Now uses Dartmouth Chat API key
    ...
)
```

## How It Works

1. **PydanticOutputParser**: Converts your `Action` Pydantic model into format instructions that the LLM can understand
2. **ChatPromptTemplate**: Structures the conversation with system and user messages, including format instructions
3. **Chain**: Combines prompt → LLM → parser into a single pipeline using the `|` operator
4. **Structured Output**: The parser automatically converts the LLM's text response into a validated `Action` object

## Benefits

- ✅ **Type Safety**: Guaranteed to return an `Action` object or raise an exception
- ✅ **Validation**: Pydantic validates all fields match the expected types and constraints
- ✅ **Error Handling**: Try/except block with fallback action prevents crashes
- ✅ **Modularity**: Clean separation of prompt, LLM, and parsing logic
- ✅ **LangChain Ecosystem**: Can now use LangChain's additional features (memory, tools, etc.)

## Testing

To test the updated implementation:
```bash
python3 chat_agent.py
```

Make sure you have your Dartmouth Chat API key in `dartmouth_key.env`.

## Dependencies

Ensure you have installed:
```bash
pip install langchain langchain-core langchain-dartmouth
```
