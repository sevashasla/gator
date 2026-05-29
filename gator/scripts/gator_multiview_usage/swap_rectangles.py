import cv2
import numpy as np
import numpy.typing as npt

def _validate_rectangle(top_y, left_x, h, w, name, num_patches):
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} has invalid size: height={h}, width={w}")

    if not (
        0 <= top_y < num_patches
        and 0 <= left_x < num_patches
        and top_y + h <= num_patches
        and left_x + w <= num_patches
    ):
        raise ValueError(
            f"{name} is out of bounds: "
            f"(top_y={top_y}, left_x={left_x}, height={h}, width={w})"
        )

def _rectangles_intersect(a, b):
    ay, ax, ah, aw = a
    by, bx, bh, bw = b

    return not (
        ay + ah <= by
        or by + bh <= ay
        or ax + aw <= bx
        or bx + bw <= ax
    )

def _rectangle_bounds(rectangle, patch_size):
    top_y, left_x, h, w = rectangle

    y1 = top_y * patch_size
    y2 = (top_y + h) * patch_size
    x1 = left_x * patch_size
    x2 = (left_x + w) * patch_size

    return y1, y2, x1, x2

def swap_two_rectangles(
    image: npt.NDArray[np.uint8],
    rectangle_1: tuple[int, int, int, int],
    rectangle_2: tuple[int, int, int, int],
    output_size: int = 224,
    num_patches: int = 14,
) -> npt.NDArray[np.uint8]:
    if output_size % num_patches != 0:
        raise ValueError(
            f"output_size must be divisible by num_patches, got "
            f"output_size={output_size}, num_patches={num_patches}"
        )

    patch_size = output_size // num_patches
    image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)

    r1_y, r1_x, h1, w1 = rectangle_1
    r2_y, r2_x, h2, w2 = rectangle_2

    if (h1, w1) != (h2, w2):
        raise ValueError(
            f"rectangle_1 and rectangle_2 must have the same shape, got "
            f"{(h1, w1)} and {(h2, w2)}"
        )

    _validate_rectangle(r1_y, r1_x, h1, w1, "rectangle_1", num_patches)
    _validate_rectangle(r2_y, r2_x, h2, w2, "rectangle_2", num_patches)

    if _rectangles_intersect(rectangle_1, rectangle_2):
        raise ValueError("rectangle_1 and rectangle_2 must not intersect")

    y1a, y2a, x1a, x2a = _rectangle_bounds(rectangle_1, patch_size)
    y1b, y2b, x1b, x2b = _rectangle_bounds(rectangle_2, patch_size)

    result = image.copy()
    result[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
    result[y1b:y2b, x1b:x2b] = image[y1a:y2a, x1a:x2a]

    return result
