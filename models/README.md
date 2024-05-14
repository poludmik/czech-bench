
# LLM integrations

There are currently 5 available model classes to use for the evaluation, all interfaced via [Langchain](https://github.com/langchain-ai/langchain). 

The model to be evaluated can be selected in the [*eval_config.yml*](*eval_config.yml*) file using the `model_name` parameter. The choices currently available are:

- [chat_openai](chat_openai.py) - OpenAI's API accessing their GPT models. Requires the OPENAI_API_KEY environment variable to be properly set.  
Supported parameters:  
    - `model_id` - specific chat model to be evaluated, defaults to 'gpt-3.5-turbo-0125'  
    - `temperature` - output sampling temperature, defaults to 0
    - `max_tokens` - maximum of tokens to generate, defaults to 512

- [chat_anthropic](chat_anthropic.py) - Anthropics's API accessing their Claude models. Requires the ANTHROPIC_API_KEY environment variable to be properly set.  
Supported parameters:  
    - `model_id` - specific chat model to be evaluated, defaults to 'claude-3-haiku-20240307'  
    - `temperature` - output sampling temperature, defaults to 0
    - `max_tokens` - maximum of tokens to generate, defaults to 512

- [ollama_raw](ollama_raw.py) - Custom integration of the [Ollama](https://github.com/ollama/ollama) local LLM runtime. An already-running Ollama instance with the requested model pre-pulled is required. Please note that running a model through Ollama may result in inferior performance.  
Supported parameters:  
    - `model_id` - model to be loaded and evaluated, defaults to 'llama2'
    - `base_url` - URL of the running Ollama server, defaults to 'http<span>://localhost:11434'
    - `temperature` - output sampling temperature, defaults to 0

- [auto_hf](auto_hf.py) - An integration of the [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) abstractor from the Transformers library. Can automatically pull the selected model from Hugging Face. Only models compatible with the AutoModelForCausalLM and AutoModelForSeq2SeqLM classes are recommended.  
Supported parameters:  
    - `model_id` - Hugging Face ID of the selected model, or local path
    - `causal` - required to properly set the output format for causal decoder-only models. Set to False when loading an encoder-decoder based model, such as the T5. Defaults to True.
    - `do_sample` - enables output sampling instead of greedy decoding, defaults to False
    - `max_new_tokens` - maximum number of tokens to generate, defaults to 512
    - `precision` - determines the torch_dtype parameter of the model. Use 'fp16' for `torch.float16`, 'bf16' for `torch.bfloat16`, 'fp32' for `torch.float32`, or 'auto' for automatic selection based on model parameters. Additional data types are now supported via the Quanto library: 'fp8', 'int8', 'int4', and 'int2'. Defaults to 'auto'.
    - `device_map` - corresponds to the `device_map` parameter of the Transformers pipeline. Defaults to 'auto'.
    - `**kwargs` - all additional keyword arguments will be passed directly to the model's loading function. These can include `temperature`, `load_in_8bit`, `load_in_4bit`, etc.

- [llama3_chat](llama3_chat.py) - Custom integration of the LLama3 Instruct models, compliant with their dedicated prompt structure. The models can be automatically pulled from Hugging Face, but require having set the HF_TOKEN environment variable and having agreed to Meta's usage terms.  
Supported parameters:  
    - `model_id` - Hugging Face ID of the selected model ('meta-llama/Meta-Llama-3-8B-Instruct' or 'meta-llama/Meta-Llama-3-70B-Instruct'), or local path
    - `do_sample` - enables output sampling instead of greedy decoding, defaults to False
    - `max_new_tokens` - maximum number of tokens to generate, defaults to 512
    - `precision` - determines the torch_dtype parameter of the model. Use 'fp16' for `torch.float16`, 'bf16' for `torch.bfloat16`, 'fp32' for `torch.float32`, or 'auto' for automatic selection based on model parameters. Additional data types are now supported via the Quanto library: 'fp8', 'int8', 'int4', and 'int2'. Defaults to 'auto'.
    - `device_map` - corresponds to the `device_map` parameter of the Transformers pipeline. Defaults to 'auto'.
    - `**kwargs` - all additional keyword arguments will be passed directly to the model's loading function. These can include `temperature`, `load_in_8bit`, `load_in_4bit`, etc.

Supported model parameters can be set using the `model_parameters` dictionary inside [*eval_config.yml*](*eval_config.yml*).

## Custom LLM integration

If your model cannot be loaded using one of the options above, feel free to implement a custom loader. The main constraint is you need to create an object compliant with Langchain's [LLM](https://python.langchain.com/docs/modules/model_io/llms/custom_llm/) or [ChatModel](https://python.langchain.com/docs/modules/model_io/chat/custom_chat_model/) structure and return it using the `get_llm` function. You can take inspiration from the custom [ollama_raw](ollama_raw.py) and [llama3_chat](llama3_chat.py) loaders.

Provided your model is implemented in *my_model.py* inside this folder, it can be loaded for evaluation by setting the `model_name` parameter inside the config file to 'my_model'.