from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import FIMRequest
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline
import torch


local_path = 'mistral_models/Codestral-22B-v0.1'

tokenizer_code = MistralTokenizer.v3()
model_code = Transformer.from_folder(local_path)

model_id = "MaziyarPanahi/Phi-3-mini-4k-instruct-v0.3"

model_chat = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2"
)

tokenizer_chat = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

streamer = TextStreamer(tokenizer_chat)


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
    """
    Welcome to the Wingman v1 demo! This virtual assistant is designed to help you code. It is optimized for Python, but should work with any language supported by Codestral!\n
    To get started, prompt the chatbot with a question. The conversational agent can help you figure out where to start. \n
    When you're ready to generate code, enter your input with the following structure: \n
    ```CODE def |function name|(|args*|)\n## |description of code functionality| <split> return |object type|``` \n
    For example, we could enter:
    ```CODE def addNumbers(num1, num2)\n## return the sum of the args <split> return integer``` \n
    Then, we can test our code by running the snippet! all we need to do is seperate the generated code from our test with 5 # symbols: ##### \n
    We can interact with and view the chat history in the chatbot window. This process will get automatically smoothed in the near future. Cheers!  
    """)
        with gr.Row():
            chatbot = gr.Chatbot()
            codebox = gr.Code(interactive = True)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox()
                submit_button = gr.Button('Submit')
                runcode = gr.Button('Run code snippet!')
                clear = gr.ClearButton([msg, chatbot])
    
    
    def respond(message, chat_history):
        # if chat_history
        if 'CODE' in message:
            message = message.split('CODE ')[1]
            prefix = message.split(' <split> ')[0].split('##')[0]
            suffix = message.split(' <split> ')[1]
            bot_message = FIMRequest(prompt=prefix, suffix=suffix)
            tokens = tokenizer_code.encode_fim(bot_message).tokens
            out_tokens, _ = generate([tokens], model_code, max_tokens=256, temperature=0.0, eos_id=tokenizer_code.instruct_tokenizer.tokenizer.eos_id)
            result = tokenizer_code.decode(out_tokens[0])
            middle = result.split(suffix)[0].strip()
            message = f'''
            The function prefix is: {prefix}
            The goal of this function is to: {message.split(' <split> ')[0].split('##')[1]}
            If it is succesful, it will return: {suffix}'''
            out = f'''\n```\n{prefix}  {middle}\n    {suffix}'''
            chat_history.append((message.replace('#', ''), out))
            return "", chat_history, out.split('```')[1]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant who always responds with detailed assistance"},
                {"role": "user", "content": message},
                    ]
            terminators = [
                tokenizer_chat.eos_token_id, # this should be <|im_end|>
                tokenizer_chat.convert_tokens_to_ids("<|assistant|>"), # sometimes model stops generating at <|assistant|>
                tokenizer_chat.convert_tokens_to_ids("<|end|>") # sometimes model stops generating at <|end|>
            ]

            pipe = pipeline(
                "text-generation",
                model=model_chat,
                tokenizer=tokenizer_chat,
            )

            generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
                "streamer": streamer,
                "eos_token_id": terminators,
            }

            output = pipe(messages, **generation_args)
            output[0]['generated_text']
            chat_history.append((message, output[0]['generated_text']))
            return "", chat_history, ''
    
    def execute(code, chat_history):
        test = code.split('#####')[1]
        # try:
        #     eval(code)
        # except:
        exec(code)
        out = f'The result of the test ```{test}``` is:\n{eval(test)}'
        chat_history.append((f"Ran code:\n ```\n{code}", out)) 
        return chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot, codebox])
    submit_button.click(respond, [msg, chatbot], [msg, chatbot, codebox])
    runcode.click(execute, [codebox, chatbot], [chatbot])

demo.launch(share = True)

