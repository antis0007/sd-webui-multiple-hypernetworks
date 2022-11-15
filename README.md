# Multiple Hypernetworks
Script that allows the use of multiple hypernetworks at once in AUTOMATIC1111's Stable Diffusion webui

## What is this?

This is a script I wrote to hack in the ability to apply multiple Hypernetworks at once in [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Additional Info:

### What's a Hypernetwork?
Hypernetworks are essentially small neural networks that can modify an image with a trained style, without taking away from your tokens / max prompt length like an embedding. They allow for fine tuning a model without touching any of its weights.

### How does this work?
It overrides the hijack, optimization and CrossAttention forward functions in order to apply multiple hypernetworks sequentially, with different weights. The way that the WebUI is structured at the moment, I don't think this could be added as a main feature without some major changes.

Honestly I'm not sure that this works exactly like I think it does, but please try to contact me if you would like to help improve my code.

### Warning!
Images generated with this script will **NOT** have the hypernetwork metadata saved to the image, meaning you must save the hypernetworks/strengths seperately if you want to reproduce exact images later.

This script will not load hypernetworks globally into the opts/shared opts, since there is no reason to, and it just breaks a lot of the image saving.

## Showcase / Usage:
![image](https://user-images.githubusercontent.com/31860133/202029527-d4b2b853-cb22-473e-8e4f-ee01efb9166d.png)

To use this script, simply drag the script into your scripts directory inside stable-diffusion-webui
> stable-diffusion-webui\scripts

Once you do that, restart your webui and select the script:
> Multiple Hypernetworks

Open the "Hypernetworks List" tab with the arrow, and you'll see all the names of your hypernetworks, to use them all you need to do is copy the names into the "Hypernetworks" textbox as a **comma seperated list**.

You will also need to provide an additional corresponding list of values for strength ranging from 0 to 1 in the "Hypernetwork strengths" textbox.

Example:
> anime_3(402c9025) ,  anime_2(813ae0d8) 

> 0.4 , 0.8

*List Requirements:*
- *Must be the same length*
- *Must not be empty*

Have Fun!

---

### Disclaimer:
I am absolutely not an expert on Stable Diffusion or even Machine Learning in general, just someone who's interested in hypernetworks and experimenting. I can't give any guarantee that this script will continue working with future changes, it ultimately hijacks and performs unintended operations through some custom functions and some existing ones.


