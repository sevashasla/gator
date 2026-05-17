# Does GATOR Use the Reference View?

## Visual Anagrams

The method does not work for our task, but one could check the approach we tried [here](./visual_anagrams/README.md)

## Identify with Visual Anagrams

### Idea

We need pictures `img1` and `img2` such that they both look realistic and `img1` can be transformed into `img2` by shuffling patches and that `img1` != `img2` (we shuffle non-equal patches).
We notice that some parts of image can be more or less uniform (e.g. a wall or a table), and we can generate a small object (area of 1-4 patches) at this uniform position, such that after moving object patches at another place, the resulting image still would look realistic. 
Let the first image is `img1` and the second image (after moving some patches) is `img2`. Let `img_s = reshuffle_patches(img1)`. 

Then we would like the model output be:

```python3
gator(img_s, reference=img1) ≈ img1 
gator(img_s, reference=img2) ≈ img2 
```

Then it means that our model utilizes the second view.

### Dataset Collection

We used ChatGPT to edit images, they are stored in [docs/diffusion_editing/images_edit](./images_edit/). We further process them in [GIMP](https://www.gimp.org) to ensure that after swapping patches the result look seamless. 

### Evaluation

After that we run `gator(img_s, reference=img1)` and `gator(img_s, reference=img2)` and calculate metrics (accuracy, mean distance to the correct position) using only patches containing the inserted object.

One should run:

```bash
python3 gator/scripts/gator_multiview_usage/by_twin_images.py \
    --ckpt-path /path/to/ckpt.pth \
    --exp-name exp-name-used-to-produce-ckpt \
    --dataset-config-path ./docs/diffusion_editing/twoview_check.yml 
```

The method would print the accuracy and save images after predicting correct positions.


## Zeroing

In this example we pass zeros instead of the reference image.

...
