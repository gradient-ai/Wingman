from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import FIMRequest
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import time
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline
import torch
import PIL

local_path = 'mistral_models/Codestral-22B-v0.1'

tokenizer_code = MistralTokenizer.v3()
model_code = Transformer.from_folder(local_path)
print('Loaded Codestral...')

# model_id = "MaziyarPanahi/Phi-3-mini-4k-instruct-v0.3"
model_id = "xtuner/llava-phi-3-mini-hf"

# model_chat = AutoModelForCausalLM.from_pretrained(
#     "MaziyarPanahi/Phi-3-mini-4k-instruct-v0.3",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True,
#     # attn_implementation="flash_attention_2"
# )

from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaProcessor.from_pretrained(model_id)

tokenizer_chat = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

streamer = TextIteratorStreamer(tokenizer_chat)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x["path"],), None))  
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
print('Loaded Phi...')
print('Launching UI')

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("<center><font size=8>Wingman🪽</center>\n<center><font size=4>Chat & run code with your LLM powered coding buddy!</center>")
        with gr.Row():
            chatbot = gr.Chatbot()
            codebox = gr.Code(interactive = True)
        with gr.Row():
            with gr.Column():
                msg = gr.MultimodalTextbox(label = 'Chat', interactive=True, file_types=["image"], placeholder="Enter message or upload file to chat!", show_label=True)
                submit_button = gr.Button('Send message to Chatbot')
            with gr.Column():
                runcode = gr.Button('Run code snippet')
                clear = gr.ClearButton([msg, chatbot])
        with gr.Row():
            gr.Textbox(label = "How to use Wingman",
                        value = """Wingman uses two LLMs (Phi and Codestral) in pair to operate. You can chat with the assistant like any LLM to ask about various things. Use it to stage any questions.\n When you're ready to generate code, enter your input with the following structure:\n\nCODE def |function name|(|args*|):\n ## |description of code functionality| <split> return |object type|\n\n For example, we could enter:\n\nCODE def addNumbers(num1, num2):\n ## return the sum of the args <split> return integer\n\nThen, we can test our code by running the snippet! all we need to do is seperate the generated code from our test with 5 # symbols: ##### \nWe can interact with and view the chat history in the chatbot window. Context understanding to be added, paste previous message in to have model understand it. This process will get automatically smoothed in the near future. Cheers!""")
                
    
    def respond(message, chat_history):
        empty = {'text':''}
        if 'def' in message['text']:
            text = 'def'+message['text'].split('def')[1]
            prefix = message['text'].split(' <split> ')[0].split('##')[0]
            suffix = message['text'].split(' <split> ')[1]
            bot_message = FIMRequest(prompt=prefix, suffix=suffix)
            tokens = tokenizer_code.encode_fim(bot_message).tokens
            out_tokens, _ = generate([tokens], model_code, max_tokens=256, temperature=0.0, eos_id=tokenizer_code.instruct_tokenizer.tokenizer.eos_id)
            result = tokenizer_code.decode(out_tokens[0])
            middle = result.split(suffix)[0].strip()
            message = f'''
            The function prefix is: {prefix}
            The goal of this function is to: {text.split(' <split> ')[0].split('##')[1]}
            If it is succesful, it will return: {suffix}'''
            out = f'''\n```\n{prefix}  {middle}\n    {suffix}'''
            chat_history.append((message.replace('#', ''), out))
            yield empty, chat_history, out.split('```')[1]
        else:
            terminators = [
                tokenizer_chat.eos_token_id, # this should be <|im_end|>
                tokenizer_chat.convert_tokens_to_ids("<|assistant|>"), # sometimes model stops generating at <|assistant|>
                tokenizer_chat.convert_tokens_to_ids("<|end|>") # sometimes model stops generating at <|end|>
            ]
            if message['files']!= []:
                model_id = "xtuner/llava-phi-3-mini-hf"
                pipe = pipeline("image-to-text", model=model_id, device=0)
                for x in message["files"]:
                    message['text'] = "<image> "+message['text']
                    # print(type(x))
                    # print(message['text'])
                    image = PIL.Image.open(x)
                    # print(type(image))
                    chat_history.append(((x,), None))
                thread = Thread(target=pipe(image, prompt=message['text'], generate_kwargs={"max_new_tokens": 200, "streamer":streamer}))
                thread.start()
                generated_text = ""
                count = 0
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful coding assistant who always responds with detailed assistance"},
                    {"role": "user", "content": message['text']},
                        ]
                terminators = [
                    tokenizer_chat.eos_token_id, # this should be <|im_end|>
                    tokenizer_chat.convert_tokens_to_ids("<|assistant|>"), # sometimes model stops generating at <|assistant|>
                    tokenizer_chat.convert_tokens_to_ids("<|end|>") # sometimes model stops generating at <|end|>
                ]
                model_id = "MaziyarPanahi/Phi-3-mini-4k-instruct-v0.3"
                pipe = pipeline("text-generation", model=model_id, device=0)
                thread = Thread(target=pipe(text_inputs=messages, max_new_tokens= 200, streamer=streamer,eos_token_id=terminators))
                thread.start()
                generated_text = ""
                count = 0
            
            for new_text in streamer:
                if count == 0:
                    generated_text = generated_text.strip('<s><image>').rstrip('<|end|>').lstrip(message['text']).lstrip('<|end|>').rstrip('<|end|>')
                    generated_text += new_text
                    print(generated_text)
                    chat_history.append((message['text'], "".join(generated_text.split('assistant')[-1])))
                    count += 1
                    time.sleep(.05)
                    yield empty, chat_history, ''
                else:
                    del chat_history[-1]
                    generated_text += new_text
                    if "|>" != "".join(generated_text.split('assistant')[-1]):
                        generated_text = generated_text.strip('<s><image>').rstrip('<|end|>').lstrip(message['text']).lstrip('<|end|>').rstrip('<|end|>')
                        chat_history.append((message['text'], "".join(generated_text.split('assistant')[-1])))
                        time.sleep(.05)
                        yield empty, chat_history, ''
                    else:
                        pass
            
        
    
    def execute(code, chat_history):
        test = code.split('#####')[1]
        exec(code, globals())
        out = f'The result of the test ```{test}``` is:\n{eval(test)}'
        chat_history.append((f"Ran code:\n ```\n{code}", out)) 
        return chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot, codebox])
    submit_button.click(respond, [msg, chatbot], [msg, chatbot, codebox])
    runcode.click(execute, [codebox, chatbot], [chatbot])

demo.launch(share = True)

