QWEN=(--llm --llm-endpoint http://localhost:11434/v1/chat/completions --llm-model qwen3.5:9b --llm-max-tokens 4096 --llm-extra '{"reasoning_effort": "none"}')
LLM=(--llm --llm-endpoint http://localhost:11434/v1/chat/completions --llm-model hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M)

GEM=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
FLASH=(--llm --llm-endpoint $GEM --llm-model gemini-3.8-flash --llm-key-env GEMINI_API_KEY --llm-max-tokens 8192 --llm-extra '{"reasoning_effort": "low"}')
FLASH_HI=(--llm --llm-endpoint $GEM --llm-model gemini-3.8-flash --llm-key-env GEMINI_API_KEY --llm-max-tokens 32768 --llm-extra '{"reasoning_effort": "high"}')
LITE=(--llm --llm-endpoint $GEM --llm-model gemini-3.8-flash-lite --llm-key-env GEMINI_API_KEY --llm-max-tokens 8192 --llm-extra '{"reasoning_effort": "low"}')

QWEN_CTX=(--context-model qwen3.5:9b --context-endpoint http://localhost:11434/v1/chat/completions --context-extra '{"reasoning_effort": "none"}' --context-max-tokens 4096)