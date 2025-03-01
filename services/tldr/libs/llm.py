from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider




sys_prompt = """
### Role  
You are an expert text summarization model designed to generate concise, informative, and engaging news summaries.  

### Task  
Generate a structured summary and relevant tags from a given news article.  

### Input Format  
You will receive a JSON object with the following fields:  
- **headline (string)**: The title of the news article.  
- **article_text (string)**: The main content of the article.  

### Output Format  
Return a JSON object with:  
- **summary (string)**: A concise, engaging, and informative summary of the article.  
- **tags (list of strings)**: A list of 3–5 relevant tags for categorization.  

### Summary Guidelines  
- Length: **250–300 characters** (about 2–3 sentences).  
- Free of spelling, grammatical, and punctuation errors.  
- Capture the article’s **most critical information**.  
- Maintain a **neutral tone**, avoiding bias, opinion, or emotional language.  
- Include **relevant keywords naturally** for SEO.  

### Tagging Guidelines  
- Tags should be **concise, descriptive, and relevant**.  
- They should help with **discovery, filtering, and categorization**.  
 
#### **Input**  
**headline**: {headline}
**article_text**: {article_text}
"""


class LLM:
    def __init__(
        self,
        model_path,
        temperature=0.7,
        max_tokens=512,
        context_window=2048,
        n_gpu_layers=0,
        n_batch=1,
    ):
        # Initialize Llama model
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=context_window,
        )

        # Initialize provider
        self.provider = LlamaCppPythonProvider(self.llm)

        # Configure provider settings
        self.settings = self.provider.get_provider_default_settings()
        self.settings.temperature = temperature
        self.settings.max_tokens = max_tokens
        self.settings.stream = True  # Enable streaming

        # Initialize the agent
        self.agent = LlamaCppAgent(
            self.provider,
            system_prompt=sys_prompt,
            predefined_messages_formatter_type=MessagesFormatterType.LLAMA_3,
            debug_output=False,
        )

    async def generate_response(self, prompt):
        """
        Generate a response using the LlamaCppAgent.
        :param prompt: The user query or prompt string.
        :return: An asynchronous generator of responses.
        """
        print(f"Generating response...")

        response = self.agent.get_chat_response(
            prompt,
            returns_streaming_generator=True,
            llm_sampling_settings=self.settings,
        )

        for chunk in response:
            yield chunk  # Convert sync generator to async
