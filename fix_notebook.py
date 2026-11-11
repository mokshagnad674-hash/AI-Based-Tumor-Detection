import json

# Read the notebook
with open('Unet.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Iterate through cells and apply fixes
for cell in notebook.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # 1. Fix Image Normalization
            if 'if img.max() > 1:' in line:
                indent = line.split('if')[0]
                source[i] = indent + 'for c in range(img.shape[0]):\n'
            elif 'img /= 255.0' in line:
                indent = line.split('img')[0][:-4] # back up to the if statement's indent
                source[i] = (
                    indent + '    mask_c = img[c] > 0\n' +
                    indent + '    if mask_c.any():\n' +
                    indent + '        mean = img[c][mask_c].mean()\n' +
                    indent + '        std = img[c][mask_c].std()\n' +
                    indent + '        if std > 0:\n' +
                    indent + '            img[c][mask_c] = (img[c][mask_c] - mean) / std\n'
                )
            
            # 2. Fix Mask Channel Extraction (Whole Tumor)
            elif 'mask[:,:,0]' in line:
                source[i] = line.replace('mask[:,:,0]', 'np.max(mask, axis=2)')
            
            # 3. Fix Visualization Clipping Issue
            elif 'img.max() + 1e-8' in line:
                source[i] = line.replace('img.max() + 1e-8', 'img.max() - img.min() + 1e-8)\n' + line.split('img_np')[0] + 'img_np = np.clip(img_np, 0, 1')
            
            # 4. Improve Mask Visualization Contrast
            elif "plt.imshow(pred, cmap='gray')" in line:
                source[i] = line.replace('pred', 'pred > 0.5')

# Save the updated notebook
with open('Unet.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Successfully updated Unet.ipynb with the required fixes!")
