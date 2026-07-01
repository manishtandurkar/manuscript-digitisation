import cv2
import numpy as np

# Load target
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
# Load original
orig = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg", cv2.IMREAD_GRAYSCALE)

# We want to find a matching configuration.
# Let's run a grid of Sauvola parameters on the original image,
# resize the output to different shapes, and compute similarity to target.
# Since the target shape is 436 x 632, let's check if the target has a border or is cropped.
# Let's try resizing the binarised original to all sizes around (436, 581) or similar.
# Wait! Let's look at the shape of the target again: 436x632.
# Let's test window_size and k values for Sauvola.

from skimage.filters import threshold_sauvola

window_sizes = [15, 25, 35, 51, 61]
k_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Let's search over sub-regions of target.
# Let's try sliding the resized binarized image over target and checking maximum matching pixels.
best_sim = 0
best_params = None

for ws in window_sizes:
    for k in k_values:
        thresh = threshold_sauvola(orig, window_size=ws, k=k)
        binary = (orig < thresh).astype(np.uint8) * 255
        
        # Try both polarities
        for polarity in [True, False]:
            src = binary if polarity else cv2.bitwise_not(binary)
            
            # Let's try resizing to target height (436) and keeping aspect ratio (which would be 436 * 400 / 300 = 581 width)
            # Or try resizing to various widths
            for w in [509, 531, 581, 632]:
                h = 436
                src_r = cv2.resize(src, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Check maximum matching overlay on target (y=0, x=0 to W-w)
                for x_offset in range(0, target.shape[1] - w + 1, 10):
                    target_sub = target[:, x_offset:x_offset+w]
                    # Since target might not be perfectly binary, threshold it at 127
                    _, target_bin = cv2.threshold(target_sub, 127, 255, cv2.THRESH_BINARY)
                    _, src_bin = cv2.threshold(src_r, 127, 255, cv2.THRESH_BINARY)
                    
                    match = np.sum(src_bin == target_bin)
                    sim = match / (h * w)
                    if sim > best_sim:
                        best_sim = sim
                        best_params = (ws, k, polarity, w, x_offset)
                        print(f"New Best: sim={sim:.4f} | ws={ws}, k={k}, polarity={polarity}, width={w}, offset={x_offset}")
