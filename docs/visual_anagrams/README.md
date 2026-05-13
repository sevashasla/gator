# Does GATOR Use the Reference View - Identify with Visual Anagrams

In this tutorial we will test whether gator relies on the second image or if it's not useful at all.

Main idea: generate img1 and img2, such that both of them look realistic, and img1 can be transformed into img2 by reshuffling patches. 
Let img_s = reshuffle_patches(img1). 
Then we would like the model output be:

```python3
gator(img_s, reference=img1) ≈ img1 
gator(img_s, reference=img2) ≈ img2 
```

1. Install visual_anagrams

```bash
git clone https://github.com/dangeng/visual_anagrams
cd visual_anagrams
```

2. Get access to [DeepFloyd](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) following the [visual_anagrams documentation](https://github.com/dangeng/visual_anagrams#deepfloyd)

3. Generate patch permutation illusions
```bash
python3 ../gator/docs/visual_anagrams/generate.sh
```

4. Unfortunately, the generated images do not look good since the border between patches is visible

![generation example](./example.png)