import numpy as np
from json_parse import overlay_mask_on_image

# Create dummy image and mask
h, w = 300, 300
image = (np.linspace(0, 255, h*w).reshape(h, w) % 256).astype(np.uint8)
mask = np.zeros((h, w), dtype=np.uint8)
mask[60:220, 80:240] = 255

# Create overlay without showing, then save
ax = overlay_mask_on_image(image, mask, alpha=0.35, show=False)
ax.figure.savefig('overlay_test.png')
print('Saved overlay_test.png')
