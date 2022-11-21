import modules.scripts as scripts
import gradio as gr
import os

from modules import devices, sd_hijack, sd_models, shared
from modules.shared import opts, cmd_opts, state, config_filename
from modules.hypernetworks import hypernetwork

import ldm.modules.attention
import ldm.modules.diffusionmodules.model
from ldm.util import default

import torch
from torch import einsum
from torch.nn.functional import silu
from einops import repeat, rearrange

old_hypernetworks = [] #keeps track of the last loaded hypernetwork parameters so we don't reload if we don't need to
shared.opts.hypernetwork_obj_list = []

#Custom hypernetwork loading:
def load_hypernetwork_custom(filename):
    hypernet = hypernetwork.Hypernetwork()
    hypernet.load(filename)
    return hypernet

os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)
hypernetworks_list = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)
def reload_hypernetworks_list():
    global hypernetworks_list
    hypernetworks_list = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)

def apply_multi_hypernetworks(hypernetwork_obj_list, context): #NO LAYER OPTION
    context_k = context
    context_v = context
    for hypernetwork_obj in hypernetwork_obj_list:
        hypernetwork_layers = (hypernetwork_obj.layers if hypernetwork_obj is not None else {})
        hypernetwork_layers = hypernetwork_layers.get(context.shape[2], None)
        if hypernetwork_layers is None:
            context_k = context_k
            context_v = context_v
            continue
        context_k = hypernetwork_layers[0](context_k)
        context_v = hypernetwork_layers[1](context_v)
    return context_k, context_v

#TODO: Create multiple versions of this function with various optimizations (Xformers, etc...)
def attention_CrossAttention_forward_custom(obj, x, context=None, mask=None):
    h = obj.heads
    q = obj.to_q(x)
    context = default(context, x)
    #New custom shared attribute for storing our loaded hypernetworks across all files:
    hypernetwork_obj_list = shared.opts.hypernetwork_obj_list
    
    context_k, context_v = apply_multi_hypernetworks(hypernetwork_obj_list, context)

    k = obj.to_k(context_k)
    v = obj.to_v(context_v)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * obj.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return obj.to_out(out)
    
class Script(scripts.Script):  
    def title(self):
        #return "Hypernetworks EX V3"
        return "Multiple Hypernetworks"
        #this script overrides optimizations
        #logic path:
        #StableDiffusionModelHijack:
        #hijack method -> apply_optimizations()
        #apply_optimizations() -> (various crossattention forward methods)
        #attention methods -> apply_hypernetwork()

        #personally I would redo how hypernets are loaded, but this is a quick fix

    def show(self, is_img2img):
        return (not is_img2img)

    def ui(self, is_img2img):
        with gr.Accordion("Hypernetworks List", open=False):
            #gr.Markdown(value = "### Hypernetworks:")
            #gr.Markdown(value = "----")
            gr.Markdown(value = "  <br>  ".join(hypernetworks_list.keys()))
        hypernetworks = gr.Textbox(label="Hypernetworks",lines=1)
        hypernetworks_strength = gr.Textbox(label="Hypernetwork strengths",lines=1)
        return [hypernetworks, hypernetworks_strength]

    def run(self, p, hypernetworks, hypernetworks_strength): #p = processing object
        #process hypernetworks:
        #shared.opts.hypernetwork_obj_list = []

        reload_hypernetworks_list()
        #DEBUG:
        #print(hypernetworks)
        #print(hypernetworks_list)

        hypernetworks = [x.strip() for x in hypernetworks.split(',')]
        hypernetworks_strength = hypernetworks_strength.split(",")
        hypernetworks_strength = [float(element) for element in hypernetworks_strength]
        #print(hypernetworks_strength)
        hypernetworks = list(zip(hypernetworks, hypernetworks_strength))
        #print()
        #print("FINAL:")
        #print(hypernetworks)

        global old_hypernetworks #bad practice lol, may not be necessary
        #shared.opts.hypernetwork_obj_list index corresponds to index in old_hypernetworks
        skip = []
        skip_x = []
        skip_y = []

        skip_ind = []
        for old in range(len(old_hypernetworks)):
            for new in range(len(hypernetworks)):
                if hypernetworks[new] == old_hypernetworks[old]:
                    skip_ind.append((old,new))
        for new,old in skip_ind:
            skip.append((new, shared.opts.hypernetwork_obj_list[old]))
            
        if len(skip) > 0:
            #print("Skipping:")
            #print(skip)
            skip_x, skip_y = zip(*skip)
            #print(skip_x) #indices to skip
            #print(skip_y) #hypernetwork objects
        else:
            print("Found nothing to skip...")

        #print("Clearing old hypernetwork objects...")
        shared.opts.hypernetwork_obj_list = []

        #shared.opts.hypernetwork_obj_list
                    

        #I HAVE NO IDEA WHAT I'M DOING

        #Alias our model:
        model = p.sd_model

        #Undo hijack:
        sd_hijack.model_hijack.undo_hijack(model)

        #Bugfix: no longer reload model weights on every generation
        #checkpoint_info = sd_models.select_checkpoint()
        #sd_models.load_model_weights(model, checkpoint_info)

        #Now we hijack again (CUSTOM HIJACK)
        #This part of the code runs a custom hijack very similar to StableDiffusionModelHijack.hijack(self, m) in sd_hijack.py
        
        model_embeddings = model.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(model_embeddings.token_embedding, sd_hijack.model_hijack)
        model.cond_stage_model = sd_hijack.FrozenCLIPEmbedderWithCustomWords(model.cond_stage_model, sd_hijack.model_hijack)
        sd_hijack.model_hijack.clip = model.cond_stage_model
        sd_hijack.undo_optimizations()
        ldm.modules.diffusionmodules.model.nonlinearity = silu
        counter = 0
        for hypernetwork_data in hypernetworks:
            if counter in skip_x: #Skips loading hypernetworks that are already loaded
                shared.opts.hypernetwork_obj_list.append(skip_y[counter])
                counter += 1
                continue
            strength = hypernetwork_data[1]
            hypernetwork_name = hypernetwork_data[0]
            hypernetwork_obj = load_hypernetwork_custom(hypernetworks_list[hypernetwork_name])

            #This is an override for setting hypernetwork module strength: (we can't with the current functions)
            print("Setting layer strength to: ", strength)
            layers = (hypernetwork_obj.layers).values()
            for layer in layers:
                for module in layer:
                    module.multiplier = strength
            shared.opts.hypernetwork_obj_list.append(hypernetwork_obj)
            counter+=1

        #DEBUG:
        #print("Hypernetwork obj list:")
        #print(shared.opts.hypernetwork_obj_list)

        #hypernetwork_obj_list now contains all hypernetwork objects we plan to use
        
        #Here we override the CrossAttention forward method with our custom one:
        ldm.modules.attention.CrossAttention.forward = attention_CrossAttention_forward_custom

        #NOTE:
        #ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.xformers_attnblock_forward
        #We don't actually need to modify the AttnBlock method it seems, TODO Investigate this further
        #No hypernetworks get loaded, may need to write a custom function to optimize?

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        sd_hijack.model_hijack.layers = flatten(model)

        #This was at the end of the hijack function, not sure if it's needed?
        if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
            model.to(devices.device)

        old_hypernetworks = hypernetworks
