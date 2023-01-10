import modules.scripts as scripts
import gradio as gr
import os

from modules import devices, sd_hijack, sd_hijack_optimizations, sd_models, shared
from modules import sd_hijack_clip, sd_hijack_open_clip #NEW
from modules import sd_hijack_unet, sd_hijack_xlmr, xlmr

from modules import processing, shared, sd_samplers, images, devices
from modules.ui import plaintext_to_html

from modules.shared import opts, cmd_opts, state, config_filename

import importlib
try:
    hypernetwork = importlib.import_module("extensions.Hypernetwork-MonkeyPatch-Extension.patches.hypernetwork")
    print('Hypernetwork-MonkeyPatch-Extension found!')
    monkeypatch_found = True
except ImportError:
    from modules.hypernetworks import hypernetwork
    print('Hypernetwork-MonkeyPatch-Extension not found')
    monkeypatch_found = False

from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingTxt2Img)

from modules.processing import Processed

import ldm.modules.attention
import ldm.modules.diffusionmodules.model
from ldm.util import default

import torch
from torch import einsum
from torch.nn.functional import silu
from einops import repeat, rearrange


import math
import sys
import traceback
from collections import namedtuple

Cached = namedtuple("Cached", ["noise", "cfg_scale", "steps", "latent", "original_prompt", "original_negative_prompt", "sigma_adjustment"])

if shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers:
    try:
        import xformers.ops
        shared.xformers_available = True
    except Exception:
        print("Cannot import xformers", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

old_hypernetworks = [] #keeps track of the last loaded hypernetwork parameters so we don't reload if we don't need to
shared.opts.hypernetwork_obj_list = []
proc = None

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
        if monkeypatch_found:
            context_k = hypernetwork_layers[0](context_k, hypernetwork_layers[0].multiplier)
            context_v = hypernetwork_layers[1](context_v, hypernetwork_layers[1].multiplier)
        else:
            context_k = hypernetwork_layers[0](context_k)
            context_v = hypernetwork_layers[1](context_v)
    return context_k, context_v

def apply_optimizations_custom(sd_hijack):
    sd_hijack.undo_optimizations()
    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th
    
    if cmd_opts.force_enable_xformers or (cmd_opts.xformers and shared.xformers_available and torch.version.cuda and (6, 0) <= torch.cuda.get_device_capability(shared.device) <= (9, 0)):
        print("Applying xformers cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = xformers_attention_forward_custom
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.xformers_attnblock_forward
        optimization_method = 'xformers'
    elif cmd_opts.opt_sub_quad_attention:
        print("Applying sub-quadratic cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = sub_quad_attention_forward_custom
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sub_quad_attnblock_forward
        optimization_method = 'sub-quadratic'
    elif cmd_opts.opt_split_attention_v1:
        print("Applying v1 cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1_custom
        optimization_method = 'V1'
    elif not cmd_opts.disable_opt_split_attention and (cmd_opts.opt_split_attention_invokeai or not cmd_opts.opt_split_attention and not torch.cuda.is_available()):
        print("Applying cross attention optimization (InvokeAI).")
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI_custom
        optimization_method = 'InvokeAI'
    elif not cmd_opts.disable_opt_split_attention and (cmd_opts.opt_split_attention or torch.cuda.is_available()):
        print("Applying cross attention optimization (Doggettx).")
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_custom
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.cross_attention_attnblock_forward
        optimization_method = 'Doggettx'

    return optimization_method

#hijack_optimizations CUSTOM:

def xformers_attention_forward_custom(obj, x, context=None, mask=None):
    h = obj.heads
    q_in = obj.to_q(x)
    context = default(context, x)

    #New custom shared attribute for storing our loaded hypernetworks across all files:
    hypernetwork_obj_list = shared.opts.hypernetwork_obj_list
    
    context_k, context_v = apply_multi_hypernetworks(hypernetwork_obj_list, context)

    k_in = obj.to_k(context_k)
    v_in = obj.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    return obj.to_out(out)

def split_cross_attention_forward_v1_custom(obj, x, context=None, mask=None):
    h = obj.heads

    q_in = obj.to_q(x)
    context = default(context, x)

    #New custom shared attribute for storing our loaded hypernetworks across all files:
    hypernetwork_obj_list = shared.opts.hypernetwork_obj_list
    
    context_k, context_v = apply_multi_hypernetworks(hypernetwork_obj_list, context)

    k_in = obj.to_k(context_k)
    v_in = obj.to_v(context_v)
    del context, context_k, context_v, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
    for i in range(0, q.shape[0], 2):
        end = i + 2
        s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
        s1 *= obj.scale

        s2 = s1.softmax(dim=-1)
        del s1

        r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
        del s2
    del q, k, v

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return obj.to_out(r2)

def split_cross_attention_forward_invokeAI_custom(obj, x, context=None, mask=None):
    h = obj.heads

    q = obj.to_q(x)
    context = default(context, x)

    #New custom shared attribute for storing our loaded hypernetworks across all files:
    hypernetwork_obj_list = shared.opts.hypernetwork_obj_list
    
    context_k, context_v = apply_multi_hypernetworks(hypernetwork_obj_list, context)

    k = obj.to_k(context_k) * obj.scale
    v = obj.to_v(context_v)
    del context, context_k, context_v, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    r = sd_hijack_optimizations.einsum_op(q, k, v)
    return obj.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))

def split_cross_attention_forward_custom(obj, x, context=None, mask=None):
    h = obj.heads

    q_in = obj.to_q(x)
    context = default(context, x)

    #New custom shared attribute for storing our loaded hypernetworks across all files:
    hypernetwork_obj_list = shared.opts.hypernetwork_obj_list
    
    context_k, context_v = apply_multi_hypernetworks(hypernetwork_obj_list, context)

    k_in = obj.to_k(context_k)
    v_in = obj.to_v(context_v)

    k_in *= obj.scale

    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                           f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

        s2 = s1.softmax(dim=-1, dtype=q.dtype)
        del s1

        r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
        del s2

    del q, k, v

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return obj.to_out(r2)
def sub_quad_attention_forward_custom(obj, x, context=None, mask=None):
    assert mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."

    h = obj.heads

    q = obj.to_q(x)
    context = default(context, x)

    #New custom shared attribute for storing our loaded hypernetworks across all files:
    hypernetwork_obj_list = shared.opts.hypernetwork_obj_list
    
    context_k, context_v = apply_multi_hypernetworks(hypernetwork_obj_list, context)

    k = obj.to_k(context_k)
    v = obj.to_v(context_v)
    del context, context_k, context_v, x

    q = q.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    k = k.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    v = v.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)

    x = sd_hijack_optimizations.sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=obj.training)

    x = x.unflatten(0, (-1, h)).transpose(1,2).flatten(start_dim=2)

    out_proj, dropout = obj.to_out
    x = out_proj(x)
    x = dropout(x)

    return x
    
def reset_script():
    global proc
    if proc != None:
        checkpoint_info = sd_models.select_checkpoint()
        proc.sd_model = sd_models.load_model(checkpoint_info)
        shared.opts.hypernetwork_obj_list = []
    print("RESET!")

def on_ui_settings():
	section = ('multiple-hypernetworks', "Multiple Hypernetworks")

class MultipleHypernetworks(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
    
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

    #def show(self, is_img2img):
        #return (not is_img2img)
        #return(True) #show on all tabs
    def show(self, is_img2img):
        return scripts.AlwaysVisible
        #return(True) #show on all tabs

    def ui(self, is_img2img):
        global proc #global processing object, terrible solution
        with gr.Accordion("Multiple Hypernetworks", open=False):
            with gr.Row():
                with gr.Accordion("Hypernetworks List", open=False):
                    #gr.Markdown(value = "### Hypernetworks:")
                    #gr.Markdown(value = "----")
                    gr.Markdown(value = "  <br>  ".join(hypernetworks_list.keys()))
            hypernetworks = gr.Textbox(label="Hypernetworks",lines=1)
            hypernetworks_strength = gr.Textbox(label="Hypernetwork strengths",lines=1)
            btn = gr.Button(value="Reset")
            btn.click(reset_script)
        self.run_callback = False
        return [hypernetworks, hypernetworks_strength]

    #def on_script_unloaded():
        #model = shared.sd_model
        #hijack:
        #sd_hijack.model_hijack.hijack(model)
        

    
    """ def process(
		self,
		p: StableDiffusionProcessing,
        **kwargs
    ):
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            global old_hypernetworks
            temp_hypernetworks = []
            temp_hypernetworks_strength = []
            for i in range(len(old_hypernetworks)):
                temp_hypernetworks.append(old_hypernetworks[i][0])
                temp_hypernetworks_strength.append(old_hypernetworks[i][1])
            self.hypernetwork = str(temp_hypernetworks)
            self.hypernetwork_strength = str(temp_hypernetworks_strength) """
    
    #def process(self, p):
        #global old_hypernetworks
        #p.extra_generation_params["Multiple Hypernetworks"] = str(old_hypernetworks)


    def process(self, p, hypernetworks, hypernetworks_strength): #p = processing object
        global proc
        global old_hypernetworks #bad practice lol, may not be necessary

        proc = p
        #Alias our model:
        model = p.sd_model

        reload_hypernetworks_list()
        #DEBUG:
        #print(hypernetworks)
        #print(hypernetworks_list)

        #check for blank boxes:
        if((hypernetworks.strip() == "") and (hypernetworks_strength.strip() == "")):
            if len(old_hypernetworks)>0: #clear out old hypernets
                print("Empty hypernetworks, clearing: " + str(old_hypernetworks))
                shared.opts.hypernetwork_obj_list = []
                old_hypernetworks = []
                #reset_script() #SLOW!
                sd_hijack.model_hijack.hijack(model)
                #sd_hijack.apply_optimizations() #redundant?

            #processed = processing.process_images(proc)
            #return(processed)
        else:
            hypernetworks = [x.strip() for x in hypernetworks.split(',')]
            hypernetworks_strength = hypernetworks_strength.split(",")
            hypernetworks_strength = [float(element) for element in hypernetworks_strength]
            #print(hypernetworks_strength)
            hypernetworks = list(zip(hypernetworks, hypernetworks_strength)) #contains hypernet strings and strengths

            #print("FINAL:")
            #print(hypernetworks)

            
            #shared.opts.hypernetwork_obj_list index corresponds to index in old_hypernetworks
            skip = []

            for old_ind in range(len(old_hypernetworks)):
                for new_ind in range(len(hypernetworks)):
                    if hypernetworks[new_ind][0] == old_hypernetworks[old_ind][0]: #if names identical
                        skip.append((new_ind, shared.opts.hypernetwork_obj_list[old_ind])) #[0] = new hypernet index, [1] = old hypernet object

            #print("Clearing old hypernetwork objects...")
            shared.opts.hypernetwork_obj_list = []

            #shared.opts.hypernetwork_obj_list


            #Undo hijack:
            sd_hijack.model_hijack.undo_hijack(model)

            #Now we hijack again (CUSTOM HIJACK)
            #This part of the code runs a custom hijack very similar to StableDiffusionModelHijack.hijack(self, m) in sd_hijack.py
            if type(model.cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
                model_embeddings = model.cond_stage_model.roberta.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(model_embeddings.word_embeddings, sd_hijack.model_hijack)
                model.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(model.cond_stage_model, sd_hijack.model_hijack)
            
            elif type(model.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
                model_embeddings = model.cond_stage_model.transformer.text_model.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(model_embeddings.token_embedding, sd_hijack.model_hijack)
                model.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(model.cond_stage_model, sd_hijack.model_hijack)

            elif type(model.cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
                model.cond_stage_model.model.token_embedding = sd_hijack.EmbeddingsWithFixes(model.cond_stage_model.model.token_embedding, sd_hijack.model_hijack)
                model.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(model.cond_stage_model, sd_hijack.model_hijack)
            

            sd_hijack.model_hijack.clip = model.cond_stage_model

            
            #self.optimization_method = apply_optimizations()
            #Apply optimizations
            #sd_hijack.undo_optimizations() #must do this before custom function since we don't want to pass sd_hijack reference... may change later
            for hypernetwork_ind in range(0, len(hypernetworks)):
                #Skips loading hypernetworks that are already loaded
                skipped = False
                for i in skip:
                    if i[0] == hypernetwork_ind:
                        strength = hypernetworks[hypernetwork_ind][1]
                        hypernetwork_obj = i[1]
                        skipped = True
                        break
                if not skipped: #if not skipped, load hypernetwork
                    hypernetwork_data = hypernetworks[hypernetwork_ind]
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

            sd_hijack.optimization_method = apply_optimizations_custom(sd_hijack)
            
            

            #DEBUG:
            #print("Hypernetwork obj list:")
            #print(shared.opts.hypernetwork_obj_list)

            #hypernetwork_obj_list now contains all hypernetwork objects we plan to use
            
            #apply_optimizations_custom()

            #NOTE:
            #ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.xformers_attnblock_forward
            #We don't actually need to modify the AttnBlock method it seems, TODO Investigate this further
            #No hypernetworks get loaded, may need to write a custom function to optimize?
            sd_hijack.fix_checkpoint() #NEW

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
            #print("PROCESSING INFO:")
            #print(proc.generation_params)
            #print(proc.extra_generation_params)]
            
            #proc.extra_generation_params["Multiple Hypernetworks"] = str(hypernetworks).strip
            
            #GENERATING METADATA:
            temp_hypernetworks = []
            temp_hypernetworks_strength = []
            for i in range(len(old_hypernetworks)):
                temp_hypernetworks.append(str(old_hypernetworks[i][0]))
                temp_hypernetworks_strength.append(str(old_hypernetworks[i][1]))
            temp_hypernetworks = str(", ".join(temp_hypernetworks))
            temp_hypernetworks_strength = str(", ".join(temp_hypernetworks_strength))
            #print("TESTING:")
            #print(temp_hypernetworks)
            #print(temp_hypernetworks_strength)

            #plaintext to html wont work here...
            proc.extra_generation_params["Hypernetworks"] = temp_hypernetworks
            proc.extra_generation_params["Hypernetworks Strengths"] = temp_hypernetworks_strength
            #processed = processing.process_images(proc)
            self.run_callback = True
            #return(processed)
            #return(proc)

        

    def postprocess(self, *args):
        self.run_callback = False
        return

