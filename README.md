# Multiple Hypernetworks Extension
Extension that allows the use of multiple hypernetworks at once in [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

### EXTENSION UPDATE:
- Updated to an extension, now the Multiple Hypernetworks box will appear at the bottom of the UI without needing to select it as a script!
- Old version of the script was archived to a seperate branch
- Updated custom hijacks to be in line with the main auto1111 code
- Fixed the need for the RESET button! It's still in this release as I'm worried about bugs related to porting from script to an extension, but now it will automatically remove applied hypernetworks when the hypernetworks list box is left blank!


Please report any bugs you encounter with this version!


## Showcase / Usage:
![image](https://user-images.githubusercontent.com/31860133/211479385-57a890cb-8b7d-41f7-9460-2c265378710e.png)


To use this extension, simply clone this repo into your extension directory inside stable-diffusion-webui
> stable-diffusion-webui\extensions

Once you do that, restart your webui and the Multiple Hypernetworks dropdown menu should appear!

Open the "Hypernetworks List" tab with the arrow, and you'll see all the names of your hypernetworks, to use them all you need to do is copy the names into the "Hypernetworks" textbox as a **comma seperated list**.

You will also need to provide an additional corresponding list of values for strength ranging from 0 to 1 in the "Hypernetwork strengths" textbox.

Example:
> anime_3(402c9025) ,  anime_2(813ae0d8) 

> 0.4 , 0.8

*The lists of hypernetworks and strengths provided must be the same length*

Have Fun!


## Additional Info:

### What's a Hypernetwork?
Hypernetworks are essentially small neural networks that can modify an image with a trained style, without taking away from your tokens / max prompt length like an embedding. They allow for fine tuning a model without touching any of its weights.

### How does this work?
It overrides the hijack, optimization and CrossAttention forward functions in order to apply multiple hypernetworks sequentially, with different weights.

Please contact me if you have any ideas to contribute or help improve this project.




---

### Disclaimer:
I am absolutely not an expert on Stable Diffusion or even Machine Learning in general, just someone who's interested in hypernetworks and experimenting. I can't give any guarantee that this extension will continue working with future versions of the webui.
