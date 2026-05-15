# Does GATOR Use the Reference View - Identify with Visual Anagrams

In this tutorial we will test whether gator relies on the second image or if it's not useful at all.

Main idea: we need to pictures img1 and img2 such that they both look realistic and img1 can be transformed into img2 by just shuffling patches. 
We already tried to do so with [visual anagrams](../visual_anagrams/README.md) but the generation appears to be unrealistic.
Here we notice that some parts of image can be more or less uniform (e.g. a wall or a table), and we can generate a small object (1-4 patches) such that after moving it on several patches the image still look realistic.

We used GhatGPT to edit images (it follows the prompt better than Flux). 
Example edited images are stored in [docs/diffusion_editing/images_edit](./images_edit/) 


