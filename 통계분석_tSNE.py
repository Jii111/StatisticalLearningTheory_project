# pip install threadpoolctl==3.1.0

image_dir = "../data/train"
annotation_file="../data/train/_annotations.coco.json"
coco=COCO(annotation_file)

data = []
for ann_id in coco.getAnnIds():
    ann = coco.loadAnns(ann_id)[0]
    img_info = coco.loadImgs(ann['image_id'])[0]
    data.append({
        'image_id': ann['image_id'],
        'category_id': ann['category_id'],
        'category_name': coco.loadCats(ann['category_id'])[0]['name'],
        'bbox': ann['bbox'],
        'image_path': os.path.join(image_dir, img_info['file_name'])
    })
df = pd.DataFrame(data)

images = [resize(io.imread(img), (100, 100)) for img in df['image_path']]
images_array = np.array(images)
images_flat = images_array.reshape(images_array.shape[0], -1)

tsne = TSNE(n_components=2, random_state=42)
images_tsne = tsne.fit_transform(images_flat)

cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                       data=np.column_stack((images_tsne, 
                                            df['category_id'])))
cps_df.loc[:, 'target'] = cps_df.target.astype(int)
dogs_map = {1: 'Chihuahua',
               2: 'Maltese',
               3: 'Pomeranian',
               4: 'Shih-Tzu',
               5: 'Standard_Poodle'}
cps_df.loc[:, 'target'] = cps_df.target.map(dogs_map)

grid = sns.FacetGrid(cps_df, hue="target")
grid.map(plt.scatter, 'CP1', 'CP2').add_legend()
