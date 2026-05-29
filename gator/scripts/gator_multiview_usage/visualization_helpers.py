import numpy as np
import cv2

def draw_rectangle(
    image,
    rectangle: tuple[int, int, int, int],
    output_size=224,
    num_patches=14,
    line_color=(0, 255, 0),
    line_thickness=1,
):
    """
    image: (H, W, C)
    rectangle: (y_min, x_min, h, w) in the grid of patches
    output_size: int, the size of the output image
    num_patches: int, the number of patches in one dimension
    line_color: tuple, the color of the rectangle
    line_thickness: int, the thickness of the rectangle

    Example usage:
    
    ```python
    from gator.scripts.gator_multiview_usage.swap_rectangles import swap_two_rectangles

    with open("/home/skorokho/coding/gator/docs/diffusion_editing/twoview_check.yml") as f:
        cfg = yaml.safe_load(f)

    image_info = cfg["images"][-1]
    print(image_info["name"])
    image_path = Path("/home/skorokho/coding/gator/docs/diffusion_editing/images_edit", image_info["name"])
    image = cv2.imread(str(image_path))
    image_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_w_rect = image_orig.copy()
    image_w_rect = draw_rectangle(image_w_rect, image_info["rectangle1"], line_color=(255, 0, 0))
    image_w_rect = draw_rectangle(image_w_rect, image_info["rectangle2"], line_color=(0, 255, 0))

    image_swap = swap_two_rectangles(image_orig, image_info["rectangle1"], image_info["rectangle2"])

    images = [image_orig, image_w_rect, image_swap]

    plt.figure(figsize=(12, 4))
    for i, img in enumerate(images):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
    ```
    """
    image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    patch_size = output_size // num_patches

    y_min, x_min, h, w = rectangle
    y_max = y_min + h
    x_max = x_min + w

    y_min *= patch_size
    x_min *= patch_size
    y_max *= patch_size
    x_max *= patch_size

    cv2.line(
        image,
        (x_min, y_min),
        (x_max, y_min),
        line_color,
        line_thickness,
    )

    cv2.line(
        image,
        (x_max, y_min),
        (x_max, y_max),
        line_color,
        line_thickness,
    )

    cv2.line(
        image,
        (x_max, y_max),
        (x_min, y_max),
        line_color,
        line_thickness,
    )

    cv2.line(
        image,
        (x_min, y_max),
        (x_min, y_min),
        line_color,
        line_thickness,
    )

    return image


def draw_patch_grid(
    image,
    output_size=224*2,
    num_patches=14,
    line_color=(0, 255, 0),
    text_color=(0, 255, 0),
    line_thickness=1,
    font_scale=0.3,
    font_thickness=1,
):
    """
    Resize an image to 224x224, draw a 14x14 patch grid,
    and write each patch's (y, x) index inside it.

    Args:
        image: Input image as a NumPy array, loaded with cv2.
        output_size: Final image size. Default is 224.
        num_patches: Number of patches per side. Default is 14.
        line_color: Grid line color in BGR.
        text_color: Text color in BGR.
        line_thickness: Thickness of grid lines.
        font_scale: Font scale for patch labels.
        font_thickness: Thickness of text.

    Returns:
        Annotated image as a NumPy array.

    Example usage:
    ```python
    image_name = "img24.png"
    img = cv2.imread(f"/home/skorokho/coding/gator/docs/diffusion_editing/images_edit/{image_name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = draw_patch_grid(img)
    plt.imshow(img2)
    ```
    """

    image = cv2.resize(image, (output_size, output_size))

    patch_size = output_size // num_patches

    # Draw 13 vertical and 13 horizontal internal grid lines
    for i in range(1, num_patches):
        pos = i * patch_size

        # Vertical line
        cv2.line(
            image,
            (pos, 0),
            (pos, output_size),
            line_color,
            line_thickness,
        )

        # Horizontal line
        cv2.line(
            image,
            (0, pos),
            (output_size, pos),
            line_color,
            line_thickness,
        )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Write (y, x) index in each patch
    for y in range(num_patches):
        for x in range(num_patches):
            label = f"({y},{x})"

            text_x = x * patch_size + 2
            text_y = y * patch_size + patch_size // 2

            cv2.putText(
                image,
                label,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

    return image